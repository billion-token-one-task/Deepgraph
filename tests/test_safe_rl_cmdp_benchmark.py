import unittest

import numpy as np


class SafeRLCMDPBenchmarkTests(unittest.TestCase):
    def test_cmdp_environment_has_expected_shapes(self):
        from benchmarks.safe_rl_cmdp.envs import make_cmdp

        for name in (
            "risky_shortcut",
            "delayed_safety",
            "resource_gathering",
            "randomized_bandit",
            "stochastic_bridge",
            "tight_budget_chain",
            "systematic_chain_s3_g70_c30",
            "systematic_chain_s4_g82_c35",
            "systematic_random_s4_a3_g78_c40",
            "systematic_random_s5_a2_g90_c45",
        ):
            env = make_cmdp(name, seed=0)

            self.assertEqual(env.name, name)
            self.assertEqual(env.transitions.shape, (env.n_states, env.n_actions, env.n_states))
            self.assertEqual(env.rewards.shape, (env.n_states, env.n_actions))
            self.assertEqual(env.costs.shape, (env.n_states, env.n_actions))
            self.assertAlmostEqual(float(env.start.sum()), 1.0)
            self.assertGreater(env.gamma, 0.0)
            self.assertLess(env.gamma, 1.0)
            self.assertGreater(env.cost_limit, 0.0)

    def test_systematic_cmdp_family_varies_size_discount_and_actions(self):
        from benchmarks.safe_rl_cmdp.envs import make_cmdp

        env_small = make_cmdp("systematic_chain_s3_g70_c30", seed=0)
        env_three_action = make_cmdp("systematic_random_s4_a3_g78_c40", seed=0)

        self.assertEqual(env_small.n_states, 3)
        self.assertEqual(env_small.n_actions, 2)
        self.assertAlmostEqual(env_small.gamma, 0.70)
        self.assertAlmostEqual(env_small.cost_limit, 0.30)
        self.assertEqual(env_three_action.n_actions, 3)
        self.assertEqual(env_three_action.n_states, 4)

    def test_deterministic_policy_enumeration_covers_tabular_policy_space(self):
        from benchmarks.safe_rl_cmdp.envs import make_cmdp
        from benchmarks.safe_rl_cmdp.solvers import enumerate_deterministic_policies

        env = make_cmdp("risky_shortcut", seed=0)
        policies = enumerate_deterministic_policies(env)

        self.assertEqual(len(policies), env.n_actions ** env.n_states)
        self.assertTrue(all(policy.shape == (env.n_states,) for policy in policies))

    def test_policy_evaluation_returns_discounted_reward_and_cost(self):
        from benchmarks.safe_rl_cmdp.envs import make_cmdp
        from benchmarks.safe_rl_cmdp.solvers import evaluate_policy

        env = make_cmdp("risky_shortcut", seed=0)
        policy = np.zeros(env.n_states, dtype=int)
        result = evaluate_policy(env, policy)

        self.assertIn("reward", result)
        self.assertIn("cost", result)
        self.assertTrue(np.isfinite(result["reward"]))
        self.assertTrue(np.isfinite(result["cost"]))

    def test_reward_only_can_violate_constraint_and_preference_cone_improves_safe_return(self):
        from benchmarks.safe_rl_cmdp.envs import make_cmdp
        from benchmarks.safe_rl_cmdp.solvers import preference_cone_policy, reward_only_policy

        env = make_cmdp("risky_shortcut", seed=0)

        reward_only = reward_only_policy(env)
        candidate = preference_cone_policy(env, penalties=[0.0, 1.0, 3.0, 8.0])

        self.assertGreater(reward_only["metrics"]["constraint_violation"], 0.0)
        self.assertGreater(candidate["metrics"]["safe_return"], reward_only["metrics"]["safe_return"])
        self.assertLessEqual(candidate["metrics"]["constraint_violation"], reward_only["metrics"]["constraint_violation"])

    def test_occupancy_enumeration_returns_feasible_frontier_policy(self):
        from benchmarks.safe_rl_cmdp.envs import make_cmdp
        from benchmarks.safe_rl_cmdp.solvers import occupancy_enumeration

        env = make_cmdp("delayed_safety", seed=1)

        result = occupancy_enumeration(env)

        self.assertEqual(result["status"], "ok")
        self.assertTrue(result["metrics"]["feasible"])
        self.assertLessEqual(result["metrics"]["constraint_violation"], 1e-9)

    def test_occupancy_lp_can_return_randomized_policy_better_than_deterministic_frontier(self):
        from benchmarks.safe_rl_cmdp.envs import FiniteCMDP
        from benchmarks.safe_rl_cmdp.solvers import occupancy_enumeration, occupancy_lp_optimal

        env = FiniteCMDP(
            name="requires_randomization",
            seed=0,
            transitions=np.array([[[1.0], [1.0]]]),
            rewards=np.array([[0.0, 1.0]]),
            costs=np.array([[0.0, 1.0]]),
            start=np.array([1.0]),
            gamma=0.5,
            cost_limit=1.0,
        )

        deterministic = occupancy_enumeration(env)
        randomized = occupancy_lp_optimal(env)

        self.assertTrue(randomized["metrics"]["feasible"])
        self.assertGreater(randomized["metrics"]["reward"], deterministic["metrics"]["reward"])
        self.assertEqual(randomized["policy_type"], "stationary_randomized")
        self.assertGreater(randomized["metrics"]["policy_entropy"], 0.0)

    def test_deterministic_policy_entropy_is_zero(self):
        from benchmarks.safe_rl_cmdp.envs import make_cmdp
        from benchmarks.safe_rl_cmdp.solvers import reward_only_policy

        result = reward_only_policy(make_cmdp("risky_shortcut", seed=0))

        self.assertEqual(result["policy_type"], "stationary_deterministic")
        self.assertEqual(result["metrics"]["policy_entropy"], 0.0)

    def test_occupancy_lp_reports_numerical_residuals(self):
        from benchmarks.safe_rl_cmdp.envs import make_cmdp
        from benchmarks.safe_rl_cmdp.solvers import occupancy_lp_optimal

        result = occupancy_lp_optimal(make_cmdp("stochastic_bridge", seed=0))

        self.assertLess(result["metrics"]["lp_flow_residual"], 1e-8)
        self.assertLess(result["metrics"]["lp_cost_residual"], 1e-8)
        self.assertLess(result["metrics"]["lp_objective_gap"], 1e-8)

    def test_harness_runs_methods_for_multiple_seeds(self):
        from benchmarks.safe_rl_cmdp.harness import run_safe_rl_benchmark

        payload = run_safe_rl_benchmark({
            "datasets": ["risky_shortcut"],
            "methods": [
                "reward_only",
                "lagrangian_penalty_1.00",
                "lagrangian_grid_best",
                "preference_cone_policy",
                "occupancy_lp_optimal",
            ],
            "seeds": [0, 1],
            "primary_metric": "safe_return",
        })

        rows = payload["rows"]
        self.assertEqual(payload["schema_version"], 1)
        self.assertEqual(payload["capability"], "safe_rl_cmdp")
        self.assertEqual(len(rows), 10)
        self.assertEqual({row["status"] for row in rows}, {"ok"})
        self.assertEqual({row["seed"] for row in rows}, {0, 1})
        self.assertIn("constraint_violation", rows[0]["metrics"])

        by_method = {row["method"]: row for row in rows if row["seed"] == 0}
        self.assertEqual(
            by_method["lagrangian_grid_best"]["selection_protocol"],
            "oracle_grid_selected_safe_return",
        )
        self.assertEqual(
            by_method["preference_cone_policy"]["selection_protocol"],
            "oracle_grid_selected_safe_return",
        )
        self.assertEqual(
            by_method["occupancy_lp_optimal"]["baseline_role"],
            "exact_lp_reference",
        )

    def test_lp_validation_includes_analytic_randomized_one_state_check(self):
        from benchmarks.safe_rl_cmdp.artifacts import build_lp_validation

        validation = build_lp_validation({
            "datasets": ["randomized_bandit"],
            "seeds": [0],
            "safety_penalty": 6.0,
        })

        checks = validation["analytic_randomized_checks"]
        self.assertEqual(checks[0]["status"], "ok")
        self.assertLess(checks[0]["reward_gap"], 1e-8)
        self.assertLess(checks[0]["cost_gap"], 1e-8)

    def test_lp_validation_compares_randomized_lp_to_fixed_penalty_candidate(self):
        from benchmarks.safe_rl_cmdp.artifacts import build_lp_validation

        validation = build_lp_validation({
            "datasets": ["randomized_bandit"],
            "seeds": [0],
            "safety_penalty": 6.0,
            "candidate_method": "lagrangian_penalty_4.00",
        })

        row = validation["deterministic_reference_comparisons"][0]
        self.assertEqual(row["candidate_method"], "lagrangian_penalty_4.00")
        self.assertIn("candidate_reward", row)
        self.assertIn("lp_vs_candidate_reward_gap", row)
        self.assertIn("lp_vs_deterministic_feasible_reward_gap", row)


if __name__ == "__main__":
    unittest.main()
