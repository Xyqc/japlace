"""Offline scheduling regressions, including the real process/Rust boundary."""

import unittest
from types import SimpleNamespace
from unittest.mock import patch

from poke_engine import State

from laplace.agent.engine_search import EnginePlayer, _search_world, search_cpu_layout
from laplace.cli.ladder import build_agent


class SearchBudgetTests(unittest.TestCase):
    def setUp(self):
        self.agent = object.__new__(EnginePlayer)
        self.agent.N_DETERMINIZATIONS = 8
        self.agent.SEARCH_TIME_MS = 150
        self.agent.adaptive_search = True
        self.battle = SimpleNamespace(
            battle_tag="battle-test", opponent_team={"one": object()},
            opponent_active_pokemon=SimpleNamespace(fainted=False, moves={}),
        )

    def test_breadth_allocation(self):
        self.assertEqual(self.agent._turn_search_budget(self.battle), (16, 75))
        self.battle.opponent_active_pokemon.moves = {"tackle": object()}
        self.assertEqual(self.agent._turn_search_budget(self.battle), (8, 150))

    def test_revealed_team_and_fainted_active_use_normal_depth(self):
        self.battle.opponent_team = dict.fromkeys(range(4))
        self.assertEqual(self.agent._turn_search_budget(self.battle), (8, 150))
        self.battle.opponent_team = {}
        self.battle.opponent_active_pokemon.fainted = True
        self.assertEqual(self.agent._turn_search_budget(self.battle), (8, 150))

    def test_fixed_budget_opt_out_and_minimum_budget(self):
        self.agent.adaptive_search = False
        self.assertEqual(self.agent._turn_search_budget(self.battle), (8, 150))
        self.agent.adaptive_search = True
        self.agent.N_DETERMINIZATIONS = 1
        self.agent.SEARCH_TIME_MS = 1
        self.assertEqual(self.agent._turn_search_budget(self.battle), (2, 1))

    def test_cpu_layout_is_bounded(self):
        for cpus in (1, 2, 3, 4, 6, 8, 32):
            with patch("laplace.agent.engine_search.os.sched_getaffinity",
                       return_value=set(range(cpus)), create=True):
                workers, threads = search_cpu_layout()
                self.assertLessEqual(workers * threads, min(cpus, 8))
                self.assertGreaterEqual(workers, 1)
                self.assertGreaterEqual(threads, 1)

    def test_ladder_keeps_tuned_config_and_accepts_parallel_opt_in(self):
        # This branch's ladder keeps its own tuned sequential config (450 ms/world); the
        # adaptive/parallel mode from upstream is opt-in through build_agent's **extra.
        config = build_agent(None, cls=lambda **kwargs: kwargs)
        self.assertEqual(config["search_time_ms"], 450)
        self.assertEqual(config["threads"], 8)
        self.assertFalse(config.get("parallel_worlds", False))
        self.assertFalse(config.get("adaptive_search", False))
        opt_in = build_agent(None, cls=lambda **kwargs: kwargs, parallel_worlds=True,
                             adaptive_search=True, search_workers=4, threads=2)
        self.assertTrue(opt_in["parallel_worlds"])
        self.assertTrue(opt_in["adaptive_search"])
        self.assertEqual((opt_in["search_workers"], opt_in["threads"]), (4, 2))

    def test_sequential_search_also_honors_adaptive_budget(self):
        agent = EnginePlayer(start_listening=False, n_determinizations=2,
                             search_time_ms=40, threads=1, adaptive_search=True)
        with patch.object(agent, "_opp_used_since_switch", return_value=set()), \
                patch.object(agent.metagame, "build_state", side_effect=lambda *a, **k: State()), \
                patch("laplace.agent.engine_search.monte_carlo_tree_search",
                      return_value=SimpleNamespace(side_one=[], total_visits=1)) as search:
            agent._pooled_policy(self.battle)
        self.assertEqual(search.call_count, 4)
        self.assertTrue(all(call.args[1] == 20 for call in search.call_args_list))

    def test_parallel_policy_uses_adaptive_budget_and_reuses_pool(self):
        agent = EnginePlayer(start_listening=False, n_determinizations=2,
                             search_time_ms=40, threads=1, search_workers=2,
                             parallel_worlds=True, adaptive_search=True)
        battle = SimpleNamespace(
            battle_tag="battle-test", opponent_team={},
            opponent_active_pokemon=SimpleNamespace(fainted=False, moves={}),
        )
        try:
            with patch.object(agent, "_opp_used_since_switch", return_value=set()), \
                    patch.object(agent.metagame, "build_state", side_effect=lambda *a, **k: State()):
                policy = agent._pooled_policy(battle)
                self.assertTrue(policy)
                self.assertEqual(len(agent._worlds), 4)
                self.assertEqual(agent._search_budget["time_ms"], 20)
                pool = agent._pool()
                self.assertIs(pool, agent._pool())
                raw = pool.submit(_search_world, State().to_string(), 20, 1).result(timeout=30)
                self.assertGreater(raw[2], 0)
        finally:
            if agent._executor is not None:
                agent._executor.shutdown(wait=True, cancel_futures=True)


if __name__ == "__main__":
    unittest.main()
