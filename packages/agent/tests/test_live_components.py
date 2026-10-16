from __future__ import annotations

from argus.supervisor.live_components import LiveTester
from argus_contracts.artifacts import RunArtifactLayout
from argus_contracts.models import Step
from argus_contracts.models import TestCase as ContractTestCase


def test_live_tester_runs_case_and_publishes_cursor_action(tmp_path):
    layout = RunArtifactLayout(tmp_path / "runs", "live-tester").initialize()
    actions: list[dict] = []
    tester = LiveTester(layout, on_action=actions.append)
    case = ContractTestCase(
        id="live-click",
        journey="continue",
        description="Click the continue button",
        steps=[Step(action="click", locator="role=button[name=Continue]")],
    )

    results = tester.run(
        [case],
        "data:text/html,<button style='position:absolute;left:100px;top:60px;width:80px;height:40px'>Continue</button>",
    )

    assert len(results) == 1
    assert results[0].verdict == "PASS"
    assert len(actions) == 1
    assert actions[0]["x"] == 140
    assert actions[0]["y"] == 80
