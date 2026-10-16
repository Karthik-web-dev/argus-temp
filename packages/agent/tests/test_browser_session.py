from __future__ import annotations

import asyncio
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from unittest.mock import AsyncMock, Mock

import pytest
from argus.browser.session import (
    BrowserSession,
    BrowserSessionFactory,
    RecordingOptions,
)
from playwright.async_api import Error as PlaywrightError


class _PageHandler(BaseHTTPRequestHandler):
    def do_GET(self):
        body = b"<!doctype html><html><head><title>fixture</title></head><body>Browser fixture</body></html>"
        self.send_response(200)
        self.send_header("Content-Type", "text/html")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def log_message(self, *_args):
        pass


@pytest.fixture(scope="module")
def local_origin():
    server = ThreadingHTTPServer(("127.0.0.1", 0), _PageHandler)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    try:
        yield f"http://127.0.0.1:{server.server_port}/"
    finally:
        server.shutdown()
        server.server_close()
        thread.join(timeout=2)


@pytest.mark.parametrize("engine", ["chromium", "firefox", "webkit"])
async def test_factory_launches_supported_engines(engine, tmp_path: Path):
    async with BrowserSessionFactory(artifact_dir=tmp_path) as factory:
        session, cdp = await factory.new_context(engine)
        assert session.browser_type == engine
        assert (cdp is not None) is (engine == "chromium")
        await session.close()


async def test_close_tolerates_playwright_errors_and_releases_session(tmp_path: Path):
    context = Mock()
    context.tracing.stop = AsyncMock(side_effect=PlaywrightError("target already closed"))
    context.close = AsyncMock(side_effect=PlaywrightError("target already closed"))
    cdp_session = Mock()
    cdp_session.detach = AsyncMock(side_effect=PlaywrightError("target already closed"))
    factory = Mock()
    session = BrowserSession(
        browser_type="chromium",
        context=context,
        page=Mock(),
        cdp_session=cdp_session,
        session_id="cleanup-regression",
        artifact_dir=tmp_path,
        recording=RecordingOptions(trace=True),
        _factory=factory,
    )

    await session.close()

    factory._release.assert_called_once_with(session)


async def test_ten_concurrent_contexts_isolate_cookies_localstorage_and_indexeddb(tmp_path, local_origin):
    factory = BrowserSessionFactory(artifact_dir=tmp_path, concurrency_cap=10)
    sessions = []
    barrier = asyncio.Barrier(10)
    try:
        async def exercise(index):
            session, _ = await factory.new_context("chromium")
            sessions.append(session)
            await session.page.goto(local_origin)
            state = await session.page.evaluate("""async owner => {
              const cookieWasEmpty = document.cookie === '';
              const localWasEmpty = localStorage.getItem('owner') === null;
              const previous = await new Promise((resolve, reject) => {
                const request = indexedDB.open('argus-isolation', 1);
                request.onupgradeneeded = () => request.result.createObjectStore('kv');
                request.onerror = () => reject(request.error);
                request.onsuccess = () => {
                  const db = request.result;
                  const tx = db.transaction('kv', 'readwrite');
                  const store = tx.objectStore('kv');
                  const get = store.get('owner');
                  get.onsuccess = () => store.put(owner, 'owner');
                  tx.oncomplete = () => resolve(get.result ?? null);
                  tx.onerror = () => reject(tx.error);
                };
              });
              localStorage.setItem('owner', owner);
              document.cookie = `owner=${owner}; path=/`;
              return {cookieWasEmpty, localWasEmpty, indexedDbWasEmpty: previous === null};
            }""", str(index))
            await barrier.wait()
            after = await session.page.evaluate("""owner => ({
              cookie: document.cookie,
              local: localStorage.getItem('owner'),
              sameContext: localStorage.getItem('owner') === owner
            })""", str(index))
            return state, after

        results = await asyncio.gather(*(exercise(index) for index in range(10)))
        assert all(all(flags.values()) for flags, _ in results)
        assert all(after["sameContext"] and after["local"] == str(i) and after["cookie"] == f"owner={i}"
                   for i, (_, after) in enumerate(results))
    finally:
        await factory.close()


async def test_context_limit_blocks_until_a_session_closes(tmp_path):
    factory = BrowserSessionFactory(artifact_dir=tmp_path, concurrency_cap=2)
    first, _ = await factory.new_context()
    _second, _ = await factory.new_context()
    pending = asyncio.create_task(factory.new_context())
    try:
        await asyncio.sleep(0.1)
        assert not pending.done()
        await first.close()
        third, _ = await asyncio.wait_for(pending, timeout=5)
        assert third is not first
    finally:
        await factory.close()


async def test_trace_is_written_when_requested(tmp_path, local_origin):
    async with BrowserSessionFactory(artifact_dir=tmp_path) as factory:
        session, _ = await factory.new_context(record={"trace": True})
        await session.page.goto(local_origin)
        await session.close()
        assert session.trace_path.is_file()
        assert session.trace_path.stat().st_size > 0


async def test_init_script_and_telemetry_are_attached(tmp_path, local_origin):
    async with BrowserSessionFactory(artifact_dir=tmp_path) as factory:
        session, _ = await factory.new_context(
            init_scripts=["globalThis.__argusInit = 'ready'; console.info('init-ready');"]
        )
        await session.page.goto(local_origin)
        assert await session.page.evaluate("globalThis.__argusInit") == "ready"
        assert any(item["text"] == "init-ready" for item in session.console)
        assert any(item["url"].startswith(local_origin) for item in session.requests)
        assert any(item["url"].startswith(local_origin) for item in session.responses)
