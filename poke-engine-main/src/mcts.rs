use crate::engine::evaluate::evaluate;
use crate::engine::generate_instructions::generate_instructions_from_move_pair;
use crate::engine::state::MoveChoice;
use crate::instruction::StateInstructions;
use crate::state::State;
use rand::prelude::*;
use rand::rng;
use std::collections::HashMap;
use std::time::Duration;

fn sigmoid(x: f32) -> f32 {
    // Tuned so that ~200 points is very close to 1.0
    1.0 / (1.0 + (-0.0125 * x).exp())
}

/// Root-only learned-prior configuration (Jaxcalibur-inspired Phase 1/6).
///
/// poke-engine's tree otherwise has no way to hear from an external policy net: nodes below
/// the root are hypothetical futures nobody has featurized, so there is nothing not-uniform
/// to hand them. The root is the one place a caller (laplace's `EnginePlayer`) has already
/// built full board features for, so that is the only place this engine accepts a prior.
///
/// Defaulting `c_puct` to 0.0 (see `RootPolicy::disabled`) makes every formula below reduce
/// to plain UCB1 -- existing callers that don't pass a prior get byte-identical search
/// behaviour to before this file changed.
#[derive(Clone, Copy, Debug)]
pub struct RootPolicy {
    /// 0.0 disables the prior/regret-matching path entirely (falls back to argmax UCB1).
    pub c_puct: f32,
    /// Sample the root's move stochastically each iteration (regret-matching-style), rather
    /// than the fixed argmax UCB1 selection every non-root node still uses.
    pub regret_matching: bool,
}

impl RootPolicy {
    pub fn disabled() -> Self {
        RootPolicy { c_puct: 0.0, regret_matching: false }
    }
    pub fn active(&self) -> bool {
        self.c_puct > 0.0
    }
}

#[derive(Debug)]
pub struct Node {
    pub root: bool,
    pub parent: *mut Node,
    pub times_visited: u32,

    // represents the instructions & s1/s2 moves that led to this node from the parent
    pub instructions: StateInstructions,
    pub s1_choice: u8,
    pub s2_choice: u8,

    // represents the total score and number of visits for this node
    // de-coupled for s1 and s2
    pub s1_options: Option<Vec<MoveNode>>,
    pub s2_options: Option<Vec<MoveNode>>,

    // Only ever non-default on the root (see `RootPolicy`). Kept as a plain `Copy` field
    // rather than a reference so `Node` doesn't need a lifetime parameter threaded through
    // every existing call site.
    root_policy: RootPolicy,
}

impl Node {
    fn new() -> Node {
        Node {
            root: false,
            parent: std::ptr::null_mut(),
            instructions: StateInstructions::default(),
            times_visited: 0,
            s1_choice: 0,
            s2_choice: 0,
            s1_options: None,
            s2_options: None,
            root_policy: RootPolicy::disabled(),
        }
    }

    unsafe fn populate(&mut self, s1_options: Vec<MoveChoice>, s2_options: Vec<MoveChoice>) {
        self.populate_with_prior(s1_options, s2_options, None, None)
    }

    /// Same as `populate`, but attaches a per-option prior probability (Jaxcalibur-style
    /// policy-net output) when one is supplied. `s1_prior`/`s2_prior` must be the same
    /// length as their option list and are matched by INDEX (not by move name) -- the caller
    /// (poke-engine-py's `mcts()`) is responsible for building that vector in the same order
    /// `state.root_get_all_options()` returns, since that's the only ordering available
    /// before this function runs. A missing or wrong-length prior degrades to uniform,
    /// never to a panic or a silently-misaligned prior.
    unsafe fn populate_with_prior(
        &mut self,
        s1_options: Vec<MoveChoice>,
        s2_options: Vec<MoveChoice>,
        s1_prior: Option<&[f32]>,
        s2_prior: Option<&[f32]>,
    ) {
        let n1 = s1_options.len().max(1);
        let n2 = s2_options.len().max(1);
        let uniform1 = 1.0 / n1 as f32;
        let uniform2 = 1.0 / n2 as f32;

        let s1_options_vec: Vec<MoveNode> = s1_options
            .iter()
            .enumerate()
            .map(|(i, x)| MoveNode {
                move_choice: x.clone(),
                total_score: 0.0,
                visits: 0,
                prior: prior_at(s1_prior, i, uniform1),
            })
            .collect();
        let s2_options_vec: Vec<MoveNode> = s2_options
            .iter()
            .enumerate()
            .map(|(i, x)| MoveNode {
                move_choice: x.clone(),
                total_score: 0.0,
                visits: 0,
                prior: prior_at(s2_prior, i, uniform2),
            })
            .collect();

        self.s1_options = Some(s1_options_vec);
        self.s2_options = Some(s2_options_vec);
    }

    /// Root value estimate used by the regret-matching formula: the pooled average score
    /// across a side's options so far, i.e. an empirical V(root) for that side. Before any
    /// visits exist this is undefined, so callers treat 0 visits as "no regret signal yet"
    /// and fall back to the plain prior/exploration term (see `MoveNode::regret_matching`).
    fn side_value(side_map: &[MoveNode]) -> (f32, u32) {
        let (total_score, total_visits) = side_map
            .iter()
            .fold((0.0f32, 0u32), |(ts, tv), n| (ts + n.total_score, tv + n.visits));
        if total_visits == 0 {
            (0.0, 0)
        } else {
            (total_score / total_visits as f32, total_visits)
        }
    }

    pub fn maximize_ucb_for_side(&self, side_map: &[MoveNode], rng: &mut impl Rng) -> usize {
        if self.root && self.root_policy.active() {
            return self.select_root_side(side_map, rng);
        }
        let mut choice = 0;
        let mut best_ucb1 = f32::MIN;
        for (index, node) in side_map.iter().enumerate() {
            let this_ucb1 = node.ucb1(self.times_visited);
            if this_ucb1 > best_ucb1 {
                best_ucb1 = this_ucb1;
                choice = index;
            }
        }
        choice
    }

    /// Root-only selection. Two modes, both gated on `self.root_policy.active()`
    /// (c_puct > 0), so this method is never reached unless a caller opted in:
    ///
    ///   * `regret_matching = false` (default when a prior is supplied): PUCT, i.e. the
    ///     existing UCB1 argmax with the exploration term additionally weighted by the
    ///     learned prior: `Q + c_puct * P(a) * sqrt(N) / (1 + n)`. Still deterministic
    ///     argmax, so this is the lower-risk of the two -- it only ever biases which action
    ///     UCB1 was already going to favour, it can't introduce new variance in which
    ///     action gets chosen for a given visit-count profile.
    ///   * `regret_matching = true`: samples the root move each iteration proportional to
    ///     `max(Q(a) - V, 0) + (c_puct / sqrt(N)) * P(a)`, per Jaxcalibur's write-up
    ///     ("Architecture" -> "Search"). This is the more faithful reproduction of the
    ///     described mechanism but changes the qualitative behaviour of root visit
    ///     accumulation (a mixed strategy instead of an optimism-driven argmax), so it is
    ///     opt-in and should be A/B'd against the PUCT mode, not assumed better.
    fn select_root_side(&self, side_map: &[MoveNode], rng: &mut impl Rng) -> usize {
        let n = side_map.len();
        if n == 0 {
            return 0;
        }
        let parent_visits = self.times_visited.max(1);

        if !self.root_policy.regret_matching {
            let mut choice = 0;
            let mut best = f32::MIN;
            for (i, node) in side_map.iter().enumerate() {
                let score = node.puct(parent_visits, self.root_policy.c_puct);
                if score > best {
                    best = score;
                    choice = i;
                }
            }
            return choice;
        }

        let (value, visits) = Self::side_value(side_map);
        let weights: Vec<f32> = side_map
            .iter()
            .map(|node| node.regret_matching_weight(value, visits, self.root_policy.c_puct))
            .collect();
        let total: f32 = weights.iter().sum();
        if total <= 0.0 {
            // No regret signal yet (first visits) and a degenerate prior: fall back to
            // uniform random rather than always picking index 0, so early exploration still
            // covers every option.
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

    pub unsafe fn selection(
        &mut self,
        state: &mut State,
        children: &mut HashMap<(usize, usize, usize), Box<[Node]>>,
        rng: &mut impl Rng,
    ) -> (*mut Node, usize, usize) {
        if self.s1_options.is_none() {
            let (s1_options, s2_options) = state.get_all_options();
            self.populate(s1_options, s2_options);
        }

        let s1_mc_index = self.maximize_ucb_for_side(self.s1_options.as_ref().unwrap(), rng);
        let s2_mc_index = self.maximize_ucb_for_side(self.s2_options.as_ref().unwrap(), rng);
        let key = (self as *mut Node as usize, s1_mc_index, s2_mc_index);
        match children.get_mut(&key) {
            Some(child_vector) => {
                let child_vec_ptr = child_vector as *mut Box<[Node]>;
                let chosen_child = self.sample_node(child_vec_ptr, rng);
                state.apply_instructions(&(*chosen_child).instructions.instruction_list);
                (*chosen_child).selection(state, children, rng)
            }
            None => (self as *mut Node, s1_mc_index, s2_mc_index),
        }
    }

    unsafe fn sample_node(&self, move_vector: *mut Box<[Node]>, rng: &mut impl Rng) -> *mut Node {
        let nodes = &mut **move_vector;

        let total_weight: f32 = nodes
            .iter()
            .map(|n| n.instructions.percentage.max(0.0))
            .sum();

        let mut threshold = rng.random_range(0.0..total_weight);

        for node in nodes.iter_mut() {
            threshold -= node.instructions.percentage.max(0.0);
            if threshold <= 0.0 {
                return node as *mut Node;
            }
        }

        // fallback: return last node (handles float rounding issues that can come up)
        &mut nodes[nodes.len() - 1] as *mut Node
    }

    pub unsafe fn expand(
        &mut self,
        state: &mut State,
        s1_move_index: usize,
        s2_move_index: usize,
        children: &mut HashMap<(usize, usize, usize), Box<[Node]>>,
        rng: &mut impl Rng,
    ) -> *mut Node {
        let s1_move = &self.s1_options.as_ref().unwrap()[s1_move_index].move_choice;
        let s2_move = &self.s2_options.as_ref().unwrap()[s2_move_index].move_choice;
        // if the battle is over or both moves are none there is no need to expand
        if (state.battle_is_over() != 0.0 && !self.root)
            || (s1_move == &MoveChoice::None && s2_move == &MoveChoice::None)
        {
            return self as *mut Node;
        }
        let should_branch_on_damage = self.root || (*self.parent).root;
        let mut new_instructions =
            generate_instructions_from_move_pair(state, s1_move, s2_move, should_branch_on_damage);
        let mut this_pair_vec = Vec::with_capacity(new_instructions.len());
        for state_instructions in new_instructions.drain(..) {
            let mut new_node = Node::new();
            new_node.parent = self;
            new_node.instructions = state_instructions;
            new_node.s1_choice = s1_move_index as u8;
            new_node.s2_choice = s2_move_index as u8;
            this_pair_vec.push(new_node);
        }

        // sample a node from the new instruction list.
        // this is the node that the rollout will be done on.
        // into_boxed_slice drops the Vec's spare capacity and, more importantly,
        // makes it a type that cannot be resized, which ensures the node
        // addresses are stable for the children map keys
        let mut boxed = this_pair_vec.into_boxed_slice();
        let new_node_ptr = self.sample_node(&mut boxed, rng);
        state.apply_instructions(&(*new_node_ptr).instructions.instruction_list);

        let key = (self as *mut Node as usize, s1_move_index, s2_move_index);
        children.insert(key, boxed);
        new_node_ptr
    }

    pub unsafe fn backpropagate(&mut self, score: f32, state: &mut State) {
        self.times_visited += 1;
        if self.root {
            return;
        }

        let parent_s1_movenode =
            &mut (*self.parent).s1_options.as_mut().unwrap()[self.s1_choice as usize];
        parent_s1_movenode.total_score += score;
        parent_s1_movenode.visits += 1;

        let parent_s2_movenode =
            &mut (*self.parent).s2_options.as_mut().unwrap()[self.s2_choice as usize];
        parent_s2_movenode.total_score += 1.0 - score;
        parent_s2_movenode.visits += 1;

        state.reverse_instructions(&self.instructions.instruction_list);
        (*self.parent).backpropagate(score, state);
    }

    pub fn rollout(&mut self, state: &mut State, root_eval: &f32) -> f32 {
        let battle_is_over = state.battle_is_over();
        if battle_is_over == 0.0 {
            let eval = evaluate(state);
            sigmoid(eval - root_eval)
        } else {
            if battle_is_over == -1.0 {
                0.0
            } else {
                battle_is_over
            }
        }
    }
}

#[derive(Debug)]
pub struct MoveNode {
    pub move_choice: MoveChoice,
    pub total_score: f32,
    pub visits: u32,
    /// Learned-prior probability for this option, in [0, 1]. 1/n (uniform) unless a caller
    /// supplied a real prior for the root -- see `Node::populate_with_prior`.
    pub prior: f32,
}

/// Shared by both prior-consuming paths: index into an optional prior slice, defaulting to
/// `fallback` (uniform) when absent or short. Never panics on a mismatched-length prior --
/// see `populate_with_prior`'s doc comment for why that has to be true (the prior comes from
/// a separate Python-side computation of "same order as root_get_all_options()", which is
/// trusted but not verified, and a length mismatch should degrade gracefully, not crash the
/// ladder mid-battle).
fn prior_at(prior: Option<&[f32]>, i: usize, fallback: f32) -> f32 {
    match prior {
        Some(p) if i < p.len() && p[i].is_finite() && p[i] >= 0.0 => p[i],
        _ => fallback,
    }
}

impl MoveNode {
    pub fn ucb1(&self, parent_visits: u32) -> f32 {
        if self.visits == 0 {
            return f32::INFINITY;
        }
        let score = (self.total_score / self.visits as f32)
            + (2.0 * (parent_visits as f32).ln() / self.visits as f32).sqrt();
        score
    }

    /// AlphaZero-style PUCT: same exploitation term as `ucb1`, but the exploration term is
    /// scaled by this option's prior instead of being prior-free. `c_puct` plays the same
    /// role as AlphaZero's constant of the same name. Falls back to `f32::INFINITY` for an
    /// unvisited option exactly like `ucb1` does, so every option is still tried at least
    /// once regardless of how small its prior is -- a confident-but-wrong prior can bias
    /// *order* of exploration, never fully starve an option of its first visit.
    pub fn puct(&self, parent_visits: u32, c_puct: f32) -> f32 {
        if self.visits == 0 {
            return f32::INFINITY;
        }
        let q = self.total_score / self.visits as f32;
        let exploration = c_puct * self.prior.max(1e-6) * (parent_visits as f32).sqrt()
            / (1.0 + self.visits as f32);
        q + exploration
    }

    pub fn average_score(&self) -> f32 {
        if self.visits == 0 {
            return 0.0;
        }
        self.total_score / self.visits as f32
    }

    /// `max(Q(a) - V, 0) + (c_puct / sqrt(N)) * P(a)`, the regret-matching-flavoured root
    /// sampling weight. `root_value`/`root_visits` come from `Node::side_value` (the
    /// pooled/empirical V for this side); `root_visits == 0` means nothing has been sampled
    /// yet, in which case regret is undefined and this returns just the prior term so the
    /// very first iterations are prior-guided rather than zero-weighted.
    pub fn regret_matching_weight(&self, root_value: f32, root_visits: u32, c_puct: f32) -> f32 {
        let regret = if root_visits == 0 {
            0.0
        } else {
            (self.average_score() - root_value).max(0.0)
        };
        let n = (root_visits.max(1) as f32).sqrt();
        regret + (c_puct / n) * self.prior.max(1e-6)
    }
}


#[derive(Clone)]
pub struct MctsSideResult {
    pub move_choice: MoveChoice,
    pub total_score: f32,
    pub visits: u32,
}

impl MctsSideResult {
    pub fn average_score(&self) -> f32 {
        if self.visits == 0 {
            return 0.0;
        }
        let score = self.total_score / self.visits as f32;
        score
    }
}

pub struct MctsResult {
    pub s1: Vec<MctsSideResult>,
    pub s2: Vec<MctsSideResult>,
    pub iteration_count: u32,
}

fn mcts_iteration(
    root_node: &mut Node,
    state: &mut State,
    root_eval: &f32,
    children: &mut HashMap<(usize, usize, usize), Box<[Node]>>,
    rng: &mut impl Rng,
) {
    let (mut new_node, s1_move, s2_move) = unsafe { root_node.selection(state, children, rng) };
    new_node = unsafe { (*new_node).expand(state, s1_move, s2_move, children, rng) };
    let rollout_result = unsafe { (*new_node).rollout(state, root_eval) };
    unsafe { (*new_node).backpropagate(rollout_result, state) }
}

enum SearchLimit {
    Time(Duration),
    Iterations(u32),
}

fn run_mcts_loop(
    root_node: &mut Node,
    state: &mut State,
    root_eval: &f32,
    children: &mut HashMap<(usize, usize, usize), Box<[Node]>>,
    limit: SearchLimit,
) {
    let mut rng = rng();
    let start_time = std::time::Instant::now();
    loop {
        for _ in 0..1000 {
            mcts_iteration(root_node, state, root_eval, children, &mut rng);
        }
        if root_node.times_visited >= 10_000_000 {
            break;
        }
        match limit {
            SearchLimit::Time(max_time) => {
                if start_time.elapsed() >= max_time {
                    break;
                }
            }
            SearchLimit::Iterations(n) => {
                if root_node.times_visited >= n {
                    break;
                }
            }
        }
    }
}

pub fn perform_mcts(
    state: &mut State,
    side_one_options: Vec<MoveChoice>,
    side_two_options: Vec<MoveChoice>,
    max_time: Duration,
    max_iterations: u32,
) -> MctsResult {
    perform_mcts_with_prior(
        state,
        side_one_options,
        side_two_options,
        max_time,
        max_iterations,
        None,
        None,
        RootPolicy::disabled(),
    )
}

/// Same as `perform_mcts`, with an optional learned prior over each side's root options and
/// a `RootPolicy` controlling how it's used (see `RootPolicy`, `MoveNode::puct`,
/// `MoveNode::regret_matching_weight`). `perform_mcts` above is unchanged and calls straight
/// through to this with priors disabled, so existing callers keep today's exact behaviour.
pub fn perform_mcts_with_prior(
    state: &mut State,
    side_one_options: Vec<MoveChoice>,
    side_two_options: Vec<MoveChoice>,
    max_time: Duration,
    max_iterations: u32,
    side_one_prior: Option<Vec<f32>>,
    side_two_prior: Option<Vec<f32>>,
    root_policy: RootPolicy,
) -> MctsResult {
    let mut root_node = Node::new();
    unsafe {
        root_node.populate_with_prior(
            side_one_options,
            side_two_options,
            side_one_prior.as_deref(),
            side_two_prior.as_deref(),
        );
    }
    root_node.root = true;
    root_node.root_policy = root_policy;
    let mut children: HashMap<(usize, usize, usize), Box<[Node]>> = HashMap::new();

    let root_eval = evaluate(state);
    let search_limit = if max_iterations > 0 {
        SearchLimit::Iterations(max_iterations)
    } else {
        SearchLimit::Time(max_time)
    };
    run_mcts_loop(
        &mut root_node,
        state,
        &root_eval,
        &mut children,
        search_limit,
    );

    let result = MctsResult {
        s1: root_node
            .s1_options
            .as_ref()
            .unwrap()
            .iter()
            .map(|v| MctsSideResult {
                move_choice: v.move_choice.clone(),
                total_score: v.total_score,
                visits: v.visits,
            })
            .collect(),
        s2: root_node
            .s2_options
            .as_ref()
            .unwrap()
            .iter()
            .map(|v| MctsSideResult {
                move_choice: v.move_choice.clone(),
                total_score: v.total_score,
                visits: v.visits,
            })
            .collect(),
        iteration_count: root_node.times_visited,
    };

    result
}

