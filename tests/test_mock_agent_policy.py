from pathlib import Path

from eval.eval_agent import build_scenario, evaluate_run, run_scenario


def test_mock_agent_policy_s1_to_s5(tmp_path: Path) -> None:
    for name in ["S1", "S2", "S3", "S4", "S5"]:
        scenario = build_scenario(name)
        run = run_scenario(scenario, tmp_path)
        assert evaluate_run(scenario, run), (name, [entry.tool for entry in run.trace.entries])
