"""Adapters that connect the merged A/B/C implementations to role Protocols."""

from __future__ import annotations

import asyncio
import logging
import re
from collections.abc import Callable, Sequence
from pathlib import Path
from uuid import uuid4

from argus_contracts.artifacts import RunArtifactLayout
from argus_contracts.models import (
    AppGraph,
    ElementCatalogEntry,
    Evidence,
    Finding,
    GateResult,
    RunEvent,
    Spec,
    TestCase,
    TestResult,
    VerificationResult,
)

from argus.browser.screencast import CdpScreencastCapture
from argus.models import ModelGateway

logger = logging.getLogger(__name__)


class LiveEventBus:
    def __init__(self, layout: RunArtifactLayout) -> None:
        self.layout = layout

    def emit(self, event: RunEvent) -> None:
        self.layout.append_event(event)


class LiveSpecParser:
    def __init__(
        self,
        run_id: str,
        gateway: ModelGateway,
        *,
        target_url: str | None = None,
        layout: RunArtifactLayout | None = None,
        on_screencast_frame: Callable[[dict], None] | None = None,
    ) -> None:
        self.run_id = run_id
        self.gateway = gateway
        self.target_url = target_url
        self.layout = layout
        self.on_screencast_frame = on_screencast_frame

    def parse(self, description: str) -> Spec:
        from argus.spec.parser import parse_spec

        async def parse_with_preview() -> Spec:
            spec_task = asyncio.create_task(
                parse_spec(description, gateway=self.gateway, run_id=self.run_id)
            )
            if not (self.target_url and self.layout and self.on_screencast_frame):
                return await spec_task

            from playwright.async_api import Error as PlaywrightError

            from argus.browser.session import BrowserSessionFactory

            factory = BrowserSessionFactory(artifact_dir=self.layout.artifacts_dir)
            session = None
            capture = None
            try:
                try:
                    await factory.start()
                    session, cdp_session = await factory.new_context("chromium")
                    if cdp_session is not None:
                        capture = CdpScreencastCapture(
                            cdp_session, self.on_screencast_frame, page=session.page,
                        )
                        await capture.start()
                    try:
                        await session.page.goto(
                            self.target_url,
                            wait_until="domcontentloaded",
                            timeout=15_000,
                        )
                    except PlaywrightError as exc:
                        logger.warning("Browser preview navigation failed: %s", exc)
                except Exception:
                    # A preview must not prevent a run from testing a target if
                    # its browser cannot launch. The live run will report its
                    # own browser/setup failure at the testing phase.
                    logger.exception("Unable to start browser preview for run %s", self.run_id)

                # Keep the preview context alive while the model call is in
                # flight so the UI receives frames instead of waiting through
                # the spec parser's potentially slow response.
                return await spec_task
            finally:
                try:
                    if capture is not None:
                        await capture.stop()
                finally:
                    try:
                        if session is not None:
                            await session.close()
                    finally:
                        await factory.close()

        return asyncio.run(parse_with_preview())


class LiveExplorer:
    def __init__(
        self,
        layout: RunArtifactLayout,
        *,
        on_screencast_frame: Callable[[dict], None] | None = None,
        on_action: Callable[[dict], None] | None = None,
    ) -> None:
        self.layout = layout
        self.on_screencast_frame = on_screencast_frame
        self.on_action = on_action

    def explore(self, target_url: str) -> tuple[AppGraph, list[ElementCatalogEntry]]:
        from argus.agents.explorer_crawler import crawl_app

        graph_path = self.layout.run_dir / "app_graph.json"
        graph = asyncio.run(
            crawl_app(
                target_url,
                graph_path,
                on_screencast_frame=self.on_screencast_frame,
                on_action=self.on_action,
            )
        )
        # The graph can contain multiple captures of a control in different
        # states. Keep a stable first-seen catalog for the binder.
        elements: dict[str, ElementCatalogEntry] = {}
        for node in graph.nodes:
            if not node.captured:
                continue
            for element in node.elements:
                elements.setdefault(element.id, element)
        return graph, list(elements.values())


class LivePlaybook:
    def __init__(self) -> None:
        self.elements: Sequence[ElementCatalogEntry] = ()

    def set_elements(self, elements: Sequence[ElementCatalogEntry]) -> None:
        self.elements = elements

    def generate(self, spec: Spec) -> list[TestCase]:
        from argus.spec.playbook import generate

        return generate(spec, self.elements)


class LiveBinder:
    def __init__(self, run_id: str, gateway: ModelGateway) -> None:
        self.run_id = run_id
        self.gateway = gateway
        self.last_results = []

    def bind(self, test_cases: Sequence[TestCase], elements: Sequence[ElementCatalogEntry]) -> list[TestCase]:
        from argus.spec.binder import bind_cases

        self.last_results = asyncio.run(bind_cases(
            test_cases, elements, gateway=self.gateway, run_id=self.run_id,
        ))
        return [item.test_case for item in self.last_results]


class LiveTester:
    def __init__(
        self,
        layout: RunArtifactLayout,
        *,
        on_screencast_frame: Callable[[dict], None] | None = None,
        on_action: Callable[[dict], None] | None = None,
    ) -> None:
        self.layout = layout
        self.on_screencast_frame = on_screencast_frame
        self.on_action = on_action
        self.binding_results: dict[str, dict] = {}

    def set_binding_results(self, results: list[dict]) -> None:
        self.binding_results = {item["test_case"]["id"]: item for item in results}

    def run(self, test_cases: Sequence[TestCase], target_url: str) -> list[TestResult]:
        from argus.agents.tester import execute_case
        from argus.browser.session import BrowserSessionFactory
        from argus.config import settings

        async def execute() -> list[TestResult]:
            output: list[TestResult] = []
            async with BrowserSessionFactory(artifact_dir=self.layout.artifacts_dir) as factory:
                for case in test_cases:
                    binding = self.binding_results.get(case.id)
                    if binding and binding["verdict"] == "INCONCLUSIVE":
                        evidence = Evidence(
                            steps=case.steps,
                            expected=case.expected,
                            actual=binding.get("reason") or "Semantic binding was inconclusive.",
                        )
                        result = TestResult(test_id=case.id, verdict="INCONCLUSIVE", evidence=evidence, duration_ms=0)
                    else:
                        session, cdp_session = await factory.new_context(
                            settings.browser,
                            record={"trace": True},
                            artifact_id="case-" + re.sub(r"[^A-Za-z0-9._-]", "_", case.id)[:100],
                        )
                        screencast = (
                            CdpScreencastCapture(
                                cdp_session, self.on_screencast_frame, page=session.page,
                            )
                            if cdp_session is not None and self.on_screencast_frame is not None
                            else None
                        )
                        try:
                            if screencast is not None:
                                await screencast.start()
                            verdict, evidence, duration = await execute_case(
                                session,
                                case,
                                self.layout,
                                initial_url=target_url,
                                on_action=self.on_action,
                            )
                        finally:
                            if screencast is not None:
                                await screencast.stop()
                            await session.close()
                        result = TestResult(
                            test_id=case.id, verdict=verdict, evidence=evidence, duration_ms=duration,
                        )
                    output.append(result)
                    self.layout.append_event(RunEvent(
                        id=uuid4().hex,
                        run_id=self.layout.run_dir.name,
                        agent="tester",
                        correlation_id=case.id,
                        type="test_result",
                        payload={"test_id": case.id, "verdict": result.verdict, "duration_ms": result.duration_ms},
                    ))
            return output

        return asyncio.run(execute())


class LiveOracle:
    def __init__(self, target_url: str) -> None:
        self.target_url = target_url
        self.test_cases: dict[str, TestCase] = {}

    def set_context(self, test_cases: Sequence[TestCase]) -> None:
        self.test_cases = {case.id: case for case in test_cases}

    def evaluate(self, result: TestResult) -> list[Finding]:
        from argus.oracles import Execution, evaluate

        case = self.test_cases.get(result.test_id)
        if case is None:
            return []
        return evaluate(Execution(test_case=case, result=result, target_url=self.target_url))


class LiveGate:
    def __init__(self, target_url: str, layout: RunArtifactLayout) -> None:
        self.target_url = target_url
        self.layout = layout
        self.reproduction = {}

    def evaluate(self, finding: Finding, evidence: Evidence) -> GateResult:
        from argus.agents.evidence_gate import gate_findings
        from argus.browser.session import BrowserSessionFactory

        async def gate() -> list:
            async with BrowserSessionFactory(artifact_dir=self.layout.artifacts_dir) as factory:
                return await gate_findings(
                    [finding], target_url=self.target_url, browser_factory=factory, layout=self.layout,
                )

        decisions = asyncio.run(gate())
        if not decisions:
            return GateResult(
                finding_id=finding.id, verdict="FLAKY", attempts=0, successes=0, threshold=2 / 3,
            )
        decision = decisions[0]
        self.reproduction[finding.id] = decision.reproduction
        return decision.result


class LiveRedTeam:
    """Placeholder adapter until C-07 supplies the adversarial reviewer."""
    def verify(self, finding: Finding, spec: Spec, evidence: Evidence) -> VerificationResult:
        del spec, evidence
        return VerificationResult(
            finding_id=finding.id,
            verdict="INCONCLUSIVE",
            reason="Red Team review is not configured; implement it in C-07.",
        )


class LiveReporter:
    def build(self, findings, gates, verifications, evidence_by_finding):
        from argus.agents.reporter import build_report

        gate_by_id = {item.finding_id: item for item in gates}
        verification_by_id = {item.finding_id: item for item in verifications}

        async def build_all():
            reports = []
            for finding in findings:
                gate = gate_by_id.get(finding.id)
                verification = verification_by_id.get(finding.id)
                evidence = evidence_by_finding.get(finding.id)
                if not gate or not verification or evidence is None:
                    continue
                if gate.verdict != "ACCEPTED" or verification.verdict != "CONFIRMED":
                    continue
                document = await build_report(
                    finding, evidence, gate=gate, verification=verification,
                )
                reports.append(document.report)
            return reports

        return asyncio.run(build_all())


class LiveExporter:
    def export(self, test_cases: Sequence[TestCase], destination: Path) -> list[Path]:
        from argus.reporter.playwright_export import export_test

        return [export_test(case, destination / f"{case.id}.spec.ts") for case in test_cases]


def build_live_components(
    *,
    run_id: str,
    target_url: str,
    layout: RunArtifactLayout,
    on_screencast_frame: Callable[[dict], None] | None = None,
    on_action: Callable[[dict], None] | None = None,
) -> dict:
    """Return runtime roles for the actual browser pipeline."""
    from argus.components import ROLE_PROTOCOLS
    from argus.stubs import GoldenFixtures, build_demo_registry

    gateway = ModelGateway()
    fixtures = GoldenFixtures()
    registry = build_demo_registry(fixtures, layout)
    factories = {
        "spec_parser": lambda: LiveSpecParser(
            run_id,
            gateway,
            target_url=target_url,
            layout=layout,
            on_screencast_frame=on_screencast_frame,
        ),
        "explorer": lambda: LiveExplorer(
            layout,
            on_screencast_frame=on_screencast_frame,
            on_action=on_action,
        ),
        "playbook": LivePlaybook,
        "binder": lambda: LiveBinder(run_id, gateway),
        "tester": lambda: LiveTester(
            layout,
            on_screencast_frame=on_screencast_frame,
            on_action=on_action,
        ),
        "oracle": lambda: LiveOracle(target_url),
        "gate": lambda: LiveGate(target_url, layout),
        "red_team": LiveRedTeam,
        "reporter": LiveReporter,
        "exporter": LiveExporter,
        "event_bus": lambda: LiveEventBus(layout),
    }
    for role, factory in factories.items():
        registry.register_real(role, factory)
    return registry.build(real_roles=ROLE_PROTOCOLS.keys())
