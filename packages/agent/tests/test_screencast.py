from __future__ import annotations

import base64
from types import SimpleNamespace
from uuid import uuid4

import pytest
from argus import api
from argus.agents.tester import execute_case
from argus.api import RunService
from argus.api.screencast import ScreencastHub
from argus.browser.screencast import CdpScreencastCapture
from argus.browser.session import BrowserSessionFactory
from argus_contracts.artifacts import RunArtifactLayout
from argus_contracts.models import Step as ContractStep
from argus_contracts.models import TestCase as ContractTestCase
from fastapi.testclient import TestClient
from starlette.websockets import WebSocketDisconnect


async def test_screencast_frames_do_not_mutate_page_dom_or_animations(tmp_path):
    async with BrowserSessionFactory(artifact_dir=tmp_path) as factory:
        session, cdp = await factory.new_context("chromium")
        assert cdp is not None
        await session.page.set_content(
            "<!doctype html><title>Static test page</title>"
            "<main><h1>Unmodified target</h1><button>Continue</button></main>"
        )
        before = await session.page.evaluate("""() => ({
          html: document.documentElement.outerHTML,
          nodeCount: document.querySelectorAll('*').length,
          animationCount: document.getAnimations().length
        })""")
        frames: list[dict] = []
        capture = CdpScreencastCapture(cdp, frames.append, page=session.page)
        try:
            await capture.start()
            # On Windows headless Chromium the CDP event can be silent, so
            # allow the screenshot fallback enough time to publish its first frame.
            await session.page.wait_for_timeout(1600)
        finally:
            await capture.stop()
        after = await session.page.evaluate("""() => ({
          html: document.documentElement.outerHTML,
          nodeCount: document.querySelectorAll('*').length,
          animationCount: document.getAnimations().length
        })""")
        await session.close()

    assert frames
    image = base64.b64decode(frames[0]["data"])
    assert image.startswith(b"\xff\xd8")
    assert after == before


async def test_screencast_falls_back_to_screenshots_when_cdp_is_silent(tmp_path):
    class SilentCdp:
        async def send(self, method, params=None):
            del method, params
            return {}

        def on(self, event, handler):
            del event, handler

        def remove_listener(self, event, handler):
            del event, handler

    async with BrowserSessionFactory(artifact_dir=tmp_path) as factory:
        session, _ = await factory.new_context("chromium")
        await session.page.set_content("<!doctype html><title>Fallback</title><h1>Visible</h1>")
        frames: list[dict] = []
        capture = CdpScreencastCapture(SilentCdp(), frames.append, page=session.page)
        try:
            await capture.start()
            await session.page.wait_for_timeout(1600)
        finally:
            await capture.stop()
        await session.close()

    assert frames
    assert frames[0]["metadata"]["source"] == "screenshot"
    assert base64.b64decode(frames[0]["data"]).startswith(b"\xff\xd8")


async def test_tester_emits_action_target_and_rationale_for_overlay(tmp_path):
    async with BrowserSessionFactory(artifact_dir=tmp_path / "browser") as factory:
        session, _ = await factory.new_context("chromium")
        await session.page.set_content(
            "<button style='position:absolute;left:100px;top:60px;width:80px;height:40px' "
            "onclick='document.body.dataset.clicked=" + '"yes"' + "'>Continue</button>"
        )
        layout = RunArtifactLayout(tmp_path / "runs", "run-action").initialize()
        actions: list[dict] = []
        case = ContractTestCase(
            id="journey-action",
            journey="publish article",
            description="Activate the continue button",
            steps=[ContractStep(action="click", locator="role=button[name=Continue]")],
        )
        await execute_case(session, case, layout, on_action=actions.append)
        await session.close()

    assert len(actions) == 1
    assert actions[0]["x"] == 140
    assert actions[0]["y"] == 80
    assert "publish article" in actions[0]["rationale"]


def test_screencast_hub_keeps_action_cues_and_latest_frame():
    hub = ScreencastHub("run-test", queue_size=2)
    queue = hub.subscribe()
    hub.publish_frame({"data": "frame-one", "metadata": {}})
    first_frame = queue.get_nowait()
    hub.publish_action({"action": "click", "x": 18, "y": 25, "rationale": "Open the menu."})
    hub.publish_frame({"data": "frame-two", "metadata": {}})
    action, second_frame = queue.get_nowait(), queue.get_nowait()
    hub.close()
    closed = queue.get_nowait()
    assert [first_frame["kind"], action["kind"], second_frame["kind"], closed["kind"]] == [
        "frame", "action", "frame", "closed",
    ]
    assert action["sequence"] + 1 == second_frame["sequence"]
    assert action["rationale"] == "Open the menu."

    late_hub = ScreencastHub("late-run")
    late_hub.publish_action({"action": "click", "rationale": "Open the menu."})
    late_hub.publish_frame({"data": "after-action", "metadata": {}})
    late_queue = late_hub.subscribe()
    assert [late_queue.get_nowait()["kind"], late_queue.get_nowait()["kind"]] == [
        "action", "frame",
    ]


def test_screencast_websocket_authenticates_and_delivers_frames(tmp_path, monkeypatch):
    token = "screencast-test-token"
    monkeypatch.setenv("ARGUS_API_TOKEN", token)
    monkeypatch.setattr(
        api,
        "settings",
        SimpleNamespace(artifact_dir=tmp_path, allowed_domains=(), api_token=token),
    )
    monkeypatch.setattr(api, "run_service", RunService(tmp_path))
    run_id = uuid4().hex
    api.run_service.store.create(run_id, "http://localhost", "screencast test")
    hub = ScreencastHub(run_id)
    api.run_service.screencasts[run_id] = hub
    hub.publish_frame({"data": base64.b64encode(b"fake-jpeg").decode(), "metadata": {"deviceWidth": 1280}})
    hub.close()

    with TestClient(api.app) as client:
        with (
            pytest.raises(WebSocketDisconnect),
            client.websocket_connect(f"/runs/{run_id}/screencast") as websocket,
        ):
            websocket.send_json({"type": "auth", "token": "wrong-token"})
            websocket.receive_json()

        with client.websocket_connect(f"/runs/{run_id}/screencast") as websocket:
            websocket.send_json({"type": "auth", "token": token})
            assert websocket.receive_json() == {"kind": "ready", "run_id": run_id}
            frame = websocket.receive_json()
            assert frame["kind"] == "frame"
            assert frame["run_id"] == run_id
            assert base64.b64decode(frame["data"]) == b"fake-jpeg"
            assert websocket.receive_json()["kind"] == "closed"
