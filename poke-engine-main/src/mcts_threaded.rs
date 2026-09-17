use crate::engine::evaluate::evaluate;
use crate::engine::generate_instructions::generate_instructions_from_move_pair;
use crate::engine::state::MoveChoice;
use crate::instruction::StateInstructions;
use crate::mcts::{MctsResult, MctsSideResult, RootPolicy};
use crate::state::State;
use dashmap::DashMap;
use rand::prelude::*;
use rand::rng;
use std::sync::atomic::{AtomicI8, AtomicU32, Ordering};
use std::sync::{Arc, OnceLock};
use std::thread;
use std::time::{Duration, Instant};

const MCTS_MAX_ITERATIONS_PER_TREE: u32 = 10_000_000;
const MCTS_DAMAGE_BRANCH_DEPTH: u8 = 2;
const SCORE_SCALE: f32 = 400.0;
const VIRTUAL_LOSS_VISITS: u32 = 3;

// Node map type alias for clarity.
// key: (parent node address, s1_move_index, s2_move_index)
// value: the branch (weighted list of outcome nodes for that move pair)
type ChildMap = DashMap<(usize, usize, usize), SharedBranch>;

fn sigmoid(x: f32) -> f32 {
    // Tuned so that ~200 points is very close to 1.0
    1.0 / (1.0 + (-0.0125 * x).exp())
}

fn prior_at(prior: Option<&[f32]>, i: usize, fallback: f32) -> f32 {
    match prior {
        Some(p) if i < p.len() && p[i].is_finite() && p[i] >= 0.0 => p[i],
        _ => fallback,
    }
}

pub struct MoveNode {
    move_choice: MoveChoice,
    total_score: AtomicU32,
    visits: AtomicU32,
    /// Set once at node construction, never mutated afterwards -- a plain f32 needs no
    /// atomic wrapper. See `mcts::MoveNode::prior` for the single-threaded twin of this
    /// field; kept in sync by hand since the two search implementations don't share a Node
    /// type (a pre-existing duplication in this crate, not introduced by this patch).
    prior: f32,
}

impl MoveNode {
    fn new(move_choice: MoveChoice, prior: f32) -> Self {
        Self {
            move_choice,
            total_score: AtomicU32::new(0),
            visits: AtomicU32::new(0),
            prior,
        }
    }

    fn add_virtual_loss(&self) {
        self.visits.fetch_add(VIRTUAL_LOSS_VISITS, Ordering::AcqRel);
    }

    fn remove_virtual_loss(&self) {
        self.visits.fetch_sub(VIRTUAL_LOSS_VISITS, Ordering::AcqRel);
    }

    fn add_result(&self, score: f32) {
        self.total_score
            .fetch_add((score * SCORE_SCALE).round() as u32, Ordering::AcqRel);
        self.visits.fetch_add(1, Ordering::AcqRel);
    }

    fn total_score_f32(&self) -> f32 {
        self.total_score.load(Ordering::Acquire) as f32 / SCORE_SCALE
    }

    fn visits_u32(&self) -> u32 {
        self.visits.load(Ordering::Acquire)
    }

    fn average_score(&self) -> f32 {
        let visits = self.visits_u32();
        if visits == 0 {
            0.0
        } else {
            self.total_score_f32() / visits as f32
        }
    }

    fn ucb1(&self, parent_visits: u32) -> f32 {
        let visits = self.visits.load(Ordering::Acquire);
        if visits == 0 {
            return f32::INFINITY;
        }
        let average_score = self.total_score_f32() / visits as f32;
        let exploration = 2.0 * (parent_visits as f32).ln().max(0.0) / visits as f32;
        average_score + exploration.sqrt()
    }

    /// See `mcts::MoveNode::puct` -- identical formula, atomic-backed reads.
    fn puct(&self, parent_visits: u32, c_puct: f32) -> f32 {
        let visits = self.visits_u32();
        if visits == 0 {
            return f32::INFINITY;
        }
        let q = self.total_score_f32() / visits as f32;
        let exploration =
            c_puct * self.prior.max(1e-6) * (parent_visits as f32).sqrt() / (1.0 + visits as f32);
        q + exploration
    }

    /// See `mcts::MoveNode::regret_matching_weight`.
    fn regret_matching_weight(&self, root_value: f32, root_visits: u32, c_puct: f32) -> f32 {
        let regret = if root_visits == 0 {
            0.0
        } else {
            (self.average_score() - root_value).max(0.0)
        };
        let n = (root_visits.max(1) as f32).sqrt();
        regret + (c_puct / n) * self.prior.max(1e-6)
    }
}

pub struct SharedNodeOptions {
    s1: Vec<MoveNode>,
    s2: Vec<MoveNode>,
}

impl SharedNodeOptions {
    fn new(s1_options: Vec<MoveChoice>, s2_options: Vec<MoveChoice>) -> Self {
        Self::new_with_prior(s1_options, s2_options, None, None)
    }

    fn new_with_prior(
        s1_options: Vec<MoveChoice>,
        s2_options: Vec<MoveChoice>,
        s1_prior: Option<&[f32]>,
        s2_prior: Option<&[f32]>,
    ) -> Self {
        let n1 = s1_options.len().max(1);
        let n2 = s2_options.len().max(1);
        let u1 = 1.0 / n1 as f32;
        let u2 = 1.0 / n2 as f32;
        Self {
            s1: s1_options
                .into_iter()
                .enumerate()
                .map(|(i, mc)| MoveNode::new(mc, prior_at(s1_prior, i, u1)))
                .collect(),
            s2: s2_options
                .into_iter()
                .enumerate()
                .map(|(i, mc)| MoveNode::new(mc, prior_at(s2_prior, i, u2)))
                .collect(),
        }
    }
}


pub struct SharedBranch {
    nodes: Arc<[Node]>,
    total_weight: f32,
}

impl SharedBranch {
    fn sample<R: Rng + ?Sized>(&self, rng: &mut R) -> *const Node {
        if self.nodes.len() <= 1 || self.total_weight <= 0.0 {
            return &self.nodes[0];
        }
        let mut threshold = rng.random_range(0.0..self.total_weight);
        for node in self.nodes.iter() {
            threshold -= node.instructions.percentage.max(0.0);
            if threshold <= 0.0 {
                return node;
            }
        }
        &self.nodes[self.nodes.len() - 1]
    }
}

struct PathStep {
    parent: *const Node,
    child: *const Node,
    s1_index: usize,
    s2_index: usize,
}

pub struct Node {
    root: bool,
    instructions: StateInstructions,
    depth: u8,
    times_visited: AtomicU32,
    virtual_losses: AtomicI8,
    options: OnceLock<SharedNodeOptions>,
    /// See `mcts::RootPolicy`. Always `RootPolicy::disabled()` except on the root node --
    /// `select_move_pair` only consults it when `self.root` is true.
    root_policy: RootPolicy,
}

impl Node {
    #[allow(dead_code)] // kept as a documented, zero-prior entry point for any future direct caller
    fn new_root(s1_options: Vec<MoveChoice>, s2_options: Vec<MoveChoice>) -> Arc<Self> {
        Self::new_root_with_prior(s1_options, s2_options, None, None, RootPolicy::disabled())
    }

    fn new_root_with_prior(
        s1_options: Vec<MoveChoice>,
        s2_options: Vec<MoveChoice>,
        s1_prior: Option<&[f32]>,
        s2_prior: Option<&[f32]>,
        root_policy: RootPolicy,
    ) -> Arc<Self> {
        let node = Arc::new(Self {
            root: true,
            instructions: StateInstructions::default(),
            depth: 0,
            times_visited: AtomicU32::new(0),
            virtual_losses: AtomicI8::new(0),
            options: OnceLock::new(),
            root_policy,
        });
        let _ = node.options.set(SharedNodeOptions::new_with_prior(
            s1_options, s2_options, s1_prior, s2_prior,
        ));
        node
    }

    fn new_child(instructions: StateInstructions, depth: u8) -> Self {
        Self {
            root: false,
            instructions,
            depth,
            times_visited: AtomicU32::new(0),
            virtual_losses: AtomicI8::new(0),
            options: OnceLock::new(),
            root_policy: RootPolicy::disabled(),
        }
    }

    fn as_key(&self) -> usize {
        self as *const Node as usize
    }

    fn ensure_options(&self, state: &State) -> &SharedNodeOptions {
        self.options.get_or_init(|| {
            let (s1, s2) = state.get_all_options();
            SharedNodeOptions::new(s1, s2)
        })
    }

    fn select_move_pair<R: Rng + ?Sized>(&self, state: &State, rng: &mut R) -> (usize, usize) {
        let options = self.ensure_options(state);
        let parent_visits = self
            .times_visited
            .load(Ordering::Acquire)
            .saturating_add(self.virtual_losses.load(Ordering::Acquire).max(0) as u32)
            .max(1);
        (
            self.maximize_ucb_for_side(&options.s1, parent_visits, rng),
            self.maximize_ucb_for_side(&options.s2, parent_visits, rng),
        )
    }

    fn selection<R: Rng + ?Sized>(
        root: &Arc<Node>,
        state: &mut State,
        rng: &mut R,
        children: &ChildMap,
        path: &mut Vec<PathStep>,
    ) -> (*const Node, usize, usize) {
        // raw pointers walk both the root (a standalone Arc<Node>) and children
        // (Nodes living inside a branch's Arc<[Node]>) uniformly. every node is
        // owned by children/root for the whole search, so the pointers stay
        // valid
        let mut current: *const Node = Arc::as_ptr(root);
        loop {
            let node = unsafe { &*current };
            let (s1_index, s2_index) = node.select_move_pair(state, rng);
            let options = node.options.get().expect("options set during selection");

            let key = (node.as_key(), s1_index, s2_index);
            match children.get(&key) {
                Some(branch) => {
                    let child = branch.sample(rng);

                    // drop the DashMap ref before mutating state to avoid
                    // holding the lock any longer than necessary. the sampled
                    // node stays alive via the branch's Arc<[Node]> in the
                    // ChildMap
                    drop(branch);

                    let child_ref = unsafe { &*child };
                    options.s1[s1_index].add_virtual_loss();
                    options.s2[s2_index].add_virtual_loss();
                    child_ref.virtual_losses.fetch_add(1, Ordering::AcqRel);
                    state.apply_instructions(&child_ref.instructions.instruction_list);
                    path.push(PathStep {
                        parent: current,
                        child,
                        s1_index,
                        s2_index,
                    });
                    current = child;
                }
                None => {
                    // this is the leaf, stop selection
                    return (current, s1_index, s2_index);
                }
            }
        }
    }

    fn side_value(side_options: &[MoveNode]) -> (f32, u32) {
        let mut total_score = 0.0f32;
        let mut total_visits = 0u32;
        for n in side_options {
            total_score += n.total_score_f32();
            total_visits += n.visits_u32();
        }
        if total_visits == 0 {
            (0.0, 0)
        } else {
            (total_score / total_visits as f32, total_visits)
        }
    }

    fn maximize_ucb_for_side<R: Rng + ?Sized>(
        &self,
        side_options: &[MoveNode],
        parent_visits: u32,
        rng: &mut R,
    ) -> usize {
        if self.root && self.root_policy.active() {
            return self.select_root_side(side_options, parent_visits, rng);
        }
        side_options
            .iter()
            .enumerate()
            .max_by(|(_, a), (_, b)| {
                a.ucb1(parent_visits)
                    .partial_cmp(&b.ucb1(parent_visits))
                    .unwrap_or(std::cmp::Ordering::Equal)
            })
            .map(|(i, _)| i)
            .unwrap_or(0)
    }

    /// See `mcts::Node::select_root_side` -- same two modes (PUCT argmax / regret-matching
    /// sample), atomic-backed reads instead of the single-threaded plain fields.
    fn select_root_side<R: Rng + ?Sized>(
        &self,
        side_options: &[MoveNode],
        parent_visits: u32,
        rng: &mut R,
    ) -> usize {
        let n = side_options.len();
        if n == 0 {
            return 0;
        }
        if !self.root_policy.regret_matching {
            return side_options
                .iter()
                .enumerate()
                .max_by(|(_, a), (_, b)| {
                    a.puct(parent_visits, self.root_policy.c_puct)
                        .partial_cmp(&b.puct(parent_visits, self.root_policy.c_puct))
                        .unwrap_or(std::cmp::Ordering::Equal)
                })
                .map(|(i, _)| i)
                .unwrap_or(0);
        }

        let (value, visits) = Self::side_value(side_options);
        let weights: Vec<f32> = side_options
            .iter()
            .map(|node| node.regret_matching_weight(value, visits, self.root_policy.c_puct))
            .collect();
        let total: f32 = weights.iter().sum();
        if total <= 0.0 {
            return rng.random_range(0..n);
        }
        let mut threshold = rng.random_range(0.0..total);
        for (i, w) in weights.iter().enumerate() {
            threshold -= w;
            if threshold <= 0.0 {
                return i;
            }
        }
        n - 1
    }

    /// looks up or creates the child branch for `(s1_index, s2_index)` and
    /// returns one sampled child, applying virtual loss bookkeeping.  Returns
    /// `None` when the node should not be expanded (battle over, both-None).
    fn expand<R: Rng + ?Sized>(
        &self,
        state: &mut State,
        s1_index: usize,
        s2_index: usize,
        rng: &mut R,
        children: &ChildMap,
    ) -> Option<*const Node> {
        let options = self
            .options
            .get()
            .expect("options initialised before expand");
        let s1_move = &options.s1[s1_index].move_choice;
        let s2_move = &options.s2[s2_index].move_choice;

        if (state.battle_is_over() != 0.0 && !self.root)
            || (s1_move == &MoveChoice::None && s2_move == &MoveChoice::None)
        {
            return None;
        }

        let should_branch_on_damage = self.depth < MCTS_DAMAGE_BRANCH_DEPTH;
        let instructions =
            generate_instructions_from_move_pair(state, s1_move, s2_move, should_branch_on_damage);

        let mut total_weight = 0.0f32;
        let nodes = instructions
            .into_iter()
            .map(|instr| {
                total_weight += instr.percentage.max(0.0);
                Node::new_child(instr, self.depth.saturating_add(1))
            })
            .collect::<Arc<[Node]>>();
        let branch = SharedBranch {
            nodes,
            total_weight,
        };

        let key = (self.as_key(), s1_index, s2_index);
        // entry() on DashMap is atomic per-shard: only one thread will
        // construct the branch; all others get the winner's branch.
        let branch_ref = children.entry(key).or_insert(branch);

        Some(branch_ref.sample(rng))
    }

    fn rollout(&self, state: &State, root_eval: f32) -> f32 {
        let battle_is_over = state.battle_is_over();
        if battle_is_over == 0.0 {
            sigmoid(evaluate(state) - root_eval)
        } else if battle_is_over == -1.0 {
            0.0
        } else {
            battle_is_over
        }
    }

    // walk `path` in reverse, updating visit counts and scores,
    // removes virtual losses, and reverse-applying instructions to restore `state` to how it
    // was in the root
    fn backpropagate(path: &[PathStep], leaf: &Node, score: f32, state: &mut State) {
        leaf.times_visited.fetch_add(1, Ordering::AcqRel);

        for step in path.iter().rev() {
            let (parent, child) = unsafe { (&*step.parent, &*step.child) };
            let options = parent.options.get().expect("path parent has options");
            options.s1[step.s1_index].add_result(score);
            options.s1[step.s1_index].remove_virtual_loss();
            options.s2[step.s2_index].add_result(1.0 - score);
            options.s2[step.s2_index].remove_virtual_loss();
            parent.times_visited.fetch_add(1, Ordering::AcqRel);
            child.virtual_losses.fetch_sub(1, Ordering::AcqRel);
            state.reverse_instructions(&child.instructions.instruction_list);
        }
    }
}

fn mcts_iteration<R: Rng + ?Sized>(
    root: &Arc<Node>,
    state: &mut State,
    root_eval: f32,
    rng: &mut R,
    children: &ChildMap,
    path: &mut Vec<PathStep>,
) {
    path.clear();

    let (leaf, s1_index, s2_index) = Node::selection(root, state, rng, children, path);
    let leaf = unsafe { &*leaf };

    let options = leaf.options.get().expect("options set during selection");
    options.s1[s1_index].add_virtual_loss();
    options.s2[s2_index].add_virtual_loss();
    let expanded = leaf.expand(state, s1_index, s2_index, rng, children);
    match expanded {
        Some(child) => {
            let child = unsafe { &*child };
            child.virtual_losses.fetch_add(1, Ordering::AcqRel);
            state.apply_instructions(&child.instructions.instruction_list);
            path.push(PathStep {
                parent: leaf,
                child,
                s1_index,
                s2_index,
            });

            let score = child.rollout(state, root_eval);

            Node::backpropagate(path, child, score, state);
        }

        // if expansion returns None,
        // the battle is either over or both sides have no valid moves
        // so no child is added to the tree
        // we do a rollout on the leaf and backpropagate without adding a child to the tree
        None => {
            // remove the virtual loss we added before expansion, since we're not actually expanding
            options.s1[s1_index].remove_virtual_loss();
            options.s2[s2_index].remove_virtual_loss();

            let score = leaf.rollout(state, root_eval);

            Node::backpropagate(path, leaf, score, state);
        }
    }
}

enum SearchLimit {
    Time,
    Iterations(u32),
}

fn run_mcts_loop(
    root: &Arc<Node>,
    root_eval: f32,
    children: Arc<ChildMap>,
    worker_state: &mut State,
    started_iterations: Arc<AtomicU32>,
    deadline: Instant,
    search_limit: SearchLimit,
) {
    let mut rng = rng();
    let mut path = Vec::with_capacity(16);
    let mut current_iterations = started_iterations.load(Ordering::Acquire);
    loop {
        for _ in 0..1000 {
            mcts_iteration(
                &root,
                worker_state,
                root_eval,
                &mut rng,
                &children,
                &mut path,
            );
            current_iterations = started_iterations.fetch_add(1, Ordering::AcqRel);
        }
        if current_iterations >= MCTS_MAX_ITERATIONS_PER_TREE {
            break;
        }
        match search_limit {
            SearchLimit::Time => {
                if Instant::now() >= deadline {
                    break;
                }
            }
            SearchLimit::Iterations(max_iterations) => {
                if current_iterations >= max_iterations {
                    break;
                }
            }
        }
    }
}

pub fn perform_mcts_shared_tree(
    state: &mut State,
    side_one_options: Vec<MoveChoice>,
    side_two_options: Vec<MoveChoice>,
    max_time: Duration,
    max_iterations: u32,
    worker_count: usize,
) -> MctsResult {
    perform_mcts_shared_tree_with_prior(
        state,
        side_one_options,
        side_two_options,
        max_time,
        max_iterations,
        worker_count,
        None,
        None,
        RootPolicy::disabled(),
    )
}

/// Same as `perform_mcts_shared_tree`, with an optional learned prior over each side's root
/// options -- see `mcts::perform_mcts_with_prior`, which this mirrors. This is the function
/// Laplace's `EnginePlayer` actually reaches at `threads=8`, so it (not the single-threaded
/// `mcts::perform_mcts_with_prior`) is the one that matters for the shipped ladder config.
pub fn perform_mcts_shared_tree_with_prior(
    state: &mut State,
    side_one_options: Vec<MoveChoice>,
    side_two_options: Vec<MoveChoice>,
    max_time: Duration,
    max_iterations: u32,
    worker_count: usize,
    side_one_prior: Option<Vec<f32>>,
    side_two_prior: Option<Vec<f32>>,
    root_policy: RootPolicy,
) -> MctsResult {
    let root_eval = evaluate(state);
    let deadline = Instant::now() + max_time;
    let root = Node::new_root_with_prior(
        side_one_options,
        side_two_options,
        side_one_prior.as_deref(),
        side_two_prior.as_deref(),
        root_policy,
    );
    let started_iterations = Arc::new(AtomicU32::new(0));

    // global map shared by all threads.
    let children: Arc<ChildMap> = Arc::new(DashMap::with_capacity(1 << 16));

    thread::scope(|scope| {
        for _ in 0..worker_count {
            let root = root.clone();
            let started_iterations = started_iterations.clone();
            let children = children.clone();
            let mut worker_state = state.clone();
            let search_limit = if max_iterations > 0 {
                SearchLimit::Iterations(max_iterations)
            } else {
                SearchLimit::Time
            };
            scope.spawn(move || {
                run_mcts_loop(
                    &root,
                    root_eval,
                    children,
                    &mut worker_state,
                    started_iterations,
                    deadline,
                    search_limit,
                );
            });
        }
    });

    let options = root.options.get().expect("root options initialized");
    MctsResult {
        s1: options
            .s1
            .iter()
            .map(|v| MctsSideResult {
                move_choice: v.move_choice,
                total_score: v.total_score_f32(),
                visits: v.visits.load(Ordering::Acquire),
            })
            .collect(),
        s2: options
            .s2
            .iter()
            .map(|v| MctsSideResult {
                move_choice: v.move_choice,
                total_score: v.total_score_f32(),
                visits: v.visits.load(Ordering::Acquire),
            })
            .collect(),
        iteration_count: root.times_visited.load(Ordering::Acquire),
    }
}
