"""Isolated Playwright browser contexts for all Argus agents."""
from __future__ import annotations

import asyncio
import re
import uuid
from dataclasses import dataclass, field
from pathlib import Path
from types import TracebackType
from typing import Any, Literal, Self

from playwright.async_api import (
    Browser,
    BrowserContext,
    CDPSession,
    Page,
    Playwright,
    async_playwright,
)
from playwright.async_api import Error as PlaywrightError

BrowserTypeName = Literal["chromium", "firefox", "webkit"]
DEFAULT_VIEWPORT = {"width": 1280, "height": 720}


@dataclass(frozen=True)
class RecordingOptions:
    """Requested evidence outputs for a browser session."""

    trace: bool = False
    video: bool = False
    har: bool = False

    @classmethod
    def from_value(cls, value: RecordingOptions | dict[str, bool] | None) -> RecordingOptions:
        if value is None:
            return cls()
        if isinstance(value, cls):
            return value
        unknown = set(value) - {"trace", "video", "har"}
        if unknown:
            raise ValueError(f"Unknown recording options: {', '.join(sorted(unknown))}")
        if any(not isinstance(flag, bool) for flag in value.values()):
            raise TypeError("record options must be booleans")
        return cls(**value)


@dataclass
class BrowserSession:
    """Context wrapper and collected browser telemetry."""

    browser_type: BrowserTypeName
    context: BrowserContext
    page: Page
    cdp_session: CDPSession | None
    session_id: str
    artifact_dir: Path
    recording: RecordingOptions
    requests: list[dict[str, Any]] = field(default_factory=list)
    responses: list[dict[str, Any]] = field(default_factory=list)
    console: list[dict[str, Any]] = field(default_factory=list)
    _factory: BrowserSessionFactory | None = field(default=None, repr=False)
    _closed: bool = field(default=False, init=False, repr=False)

    @property
    def trace_path(self) -> Path:
        return self.artifact_dir / "trace.zip"

    @property
    def har_path(self) -> Path:
        return self.artifact_dir / "network.har"

    @property
    def video_dir(self) -> Path:
        return self.artifact_dir / "videos"

    async def close(self) -> None:
        """Flush requested evidence and release this context's concurrency slot."""
        if self._closed:
            return
        self._closed = True
        try:
            if self.recording.trace:
                try:
                    await self.context.tracing.stop(path=str(self.trace_path))
                except PlaywrightError:
                    pass
            if self.cdp_session is not None:
                try:
                    await self.cdp_session.detach()
                except PlaywrightError:
                    pass
        finally:
            try:
                await self.context.close()
            except PlaywrightError:
                pass
            finally:
                if self._factory is not None:
                    self._factory._release(self)

    async def __aenter__(self) -> Self:
        return self

    async def __aexit__(
        self,
        exc_type: type[BaseException] | None,
        exc: BaseException | None,
        tb: TracebackType | None,
    ) -> None:
        await self.close()


class BrowserSessionFactory:
    """Lazily launches browsers and bounds the number of open contexts.

    Contexts do not share cookies, localStorage or IndexedDB. Pass a
    ``storage_state`` only when an intentionally pre-authenticated context is
    required. Each context receives a fresh page, telemetry listeners and, for
    Chromium only, a CDP session attached to that page.
    """

    def __init__(self, *, artifact_dir: str | Path = "artifacts/browser",
                 concurrency_cap: int = 4, headless: bool = True) -> None:
        if concurrency_cap < 1:
            raise ValueError("concurrency_cap must be at least 1")
        self.artifact_dir = Path(artifact_dir)
        self.concurrency_cap = concurrency_cap
        self.headless = headless
        self._slots = asyncio.Semaphore(concurrency_cap)
        self._playwright: Playwright | None = None
        self._browsers: dict[str, Browser] = {}
        self._launch_lock = asyncio.Lock()
        self._sessions: list[BrowserSession] = []
        self._closed = False

    async def start(self) -> BrowserSessionFactory:
        if self._closed:
            raise RuntimeError("This BrowserSessionFactory is closed")
        if self._playwright is None:
            async with self._launch_lock:
                if self._playwright is None:
                    self._playwright = await async_playwright().start()
        return self

    async def new_context(
        self,
        browser_type: BrowserTypeName = "chromium",
        viewport: dict[str, int] | None = None,
        storage_state: str | Path | dict[str, Any] | None = None,
        record: RecordingOptions | dict[str, bool] | None = None,
        *,
        init_scripts: list[str] | None = None,
        artifact_id: str | None = None,
    ) -> tuple[BrowserSession, CDPSession | None]:
        """Create a fresh isolated context and return ``(wrapper, cdp_handle)``.

        ``init_scripts`` accepts JavaScript source strings that run before page
        scripts. Keep these script-only: DOM and style injection are not exposed
        through the factory API.
        """
        if browser_type not in ("chromium", "firefox", "webkit"):
            raise ValueError(f"Unsupported browser type: {browser_type}")
        options = RecordingOptions.from_value(record)
        scripts = init_scripts or []
        if any(not isinstance(script, str) or not script.strip() for script in scripts):
            raise ValueError("init_scripts must contain non-empty JavaScript source strings")
        if artifact_id is not None and not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9._-]{0,127}", artifact_id):
            raise ValueError("artifact_id must be a safe path segment")

        await self.start()
        await self._slots.acquire()
        context: BrowserContext | None = None
        try:
            browser = await self._browser(browser_type)
            session_id = uuid.uuid4().hex
            session_dir = self.artifact_dir / (artifact_id or session_id)
            context_options: dict[str, Any] = {
                "viewport": viewport if viewport is not None else DEFAULT_VIEWPORT,
                "storage_state": str(storage_state) if isinstance(storage_state, Path) else storage_state,
            }
            if options.video:
                (session_dir / "videos").mkdir(parents=True, exist_ok=True)
                context_options["record_video_dir"] = str(session_dir / "videos")
            if options.har:
                session_dir.mkdir(parents=True, exist_ok=True)
                context_options["record_har_path"] = str(session_dir / "network.har")
                context_options["record_har_mode"] = "full"
            context = await browser.new_context(**context_options)
            for script in scripts:
                await context.add_init_script(script=script)
            if options.trace:
                session_dir.mkdir(parents=True, exist_ok=True)
                await context.tracing.start(screenshots=True, snapshots=True, sources=True)
            page = await context.new_page()
            session = BrowserSession(
                browser_type=browser_type,
                context=context,
                page=page,
                cdp_session=None,
                session_id=session_id,
                artifact_dir=session_dir,
                recording=options,
                _factory=self,
            )
            self._attach_listeners(session, page)
            context.on("page", lambda new_page: self._attach_listeners(session, new_page))
            if browser_type == "chromium":
                session.cdp_session = await context.new_cdp_session(page)
            self._sessions.append(session)
            return session, session.cdp_session
        except BaseException:
            try:
                if context is not None:
                    await context.close()
            finally:
                self._slots.release()
            raise

    async def _browser(self, browser_type: BrowserTypeName) -> Browser:
        browser = self._browsers.get(browser_type)
        if browser is not None and browser.is_connected():
            return browser
        async with self._launch_lock:
            browser = self._browsers.get(browser_type)
            if browser is None or not browser.is_connected():
                assert self._playwright is not None
                engine = getattr(self._playwright, browser_type)
                browser = await engine.launch(headless=self.headless)
                self._browsers[browser_type] = browser
        return browser

    @staticmethod
    def _attach_listeners(session: BrowserSession, page: Page) -> None:
        page.on("request", lambda request: session.requests.append({
            "url": request.url, "method": request.method, "resource_type": request.resource_type,
        }))
        page.on("response", lambda response: session.responses.append({
            "url": response.url, "status": response.status, "ok": response.ok,
        }))
        page.on("console", lambda message: session.console.append({
            "type": message.type, "text": message.text, "location": message.location,
        }))
        page.on("pageerror", lambda error: session.console.append({
            "type": "error", "text": str(error), "source": "pageerror",
        }))

        def dismiss_dialog(dialog) -> None:
            session.console.append({
                "type": "error", "text": f"JavaScript dialog: {dialog.type}", "source": "dialog",
            })
            asyncio.create_task(dialog.dismiss())

        page.on("dialog", dismiss_dialog)

    def _release(self, session: BrowserSession) -> None:
        if session in self._sessions:
            self._sessions.remove(session)
        self._slots.release()

    async def close(self) -> None:
        if self._closed:
            return
        self._closed = True
        for session in list(self._sessions):
            await session.close()
        for browser in self._browsers.values():
            if browser.is_connected():
                await browser.close()
        self._browsers.clear()
        if self._playwright is not None:
            await self._playwright.stop()
            self._playwright = None

    async def __aenter__(self) -> Self:
        return await self.start()

    async def __aexit__(
        self,
        exc_type: type[BaseException] | None,
        exc: BaseException | None,
        tb: TracebackType | None,
    ) -> None:
        await self.close()
