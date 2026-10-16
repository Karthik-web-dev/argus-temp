"""Chromium CDP screencast capture, kept outside the target page DOM."""

from __future__ import annotations

import asyncio
import base64
import logging
from collections.abc import Callable
from typing import Any

from playwright.async_api import CDPSession, Page
from playwright.async_api import Error as PlaywrightError

logger = logging.getLogger(__name__)


class CdpScreencastCapture:
    """Capture bounded 720p JPEG frames from one Chromium CDP session.

    Windows headless Chromium often never emits ``Page.screencastFrame``. When
    no CDP frame arrives, fall back to page screenshots so the UI still has a
    live view without injecting anything into the target DOM.
    """

    def __init__(
        self,
        session: CDPSession,
        publish: Callable[[dict[str, Any]], None],
        *,
        page: Page | None = None,
        max_width: int = 1280,
        max_height: int = 720,
        quality: int = 65,
    ) -> None:
        self.session = session
        self.publish = publish
        self.page = page
        self.max_width = max_width
        self.max_height = max_height
        self.quality = quality
        self._listener: Callable[[dict[str, Any]], None] | None = None
        self._started = False
        self._frame_count = 0
        self._last_cdp_frame_time = 0.0
        self._loop: asyncio.AbstractEventLoop | None = None
        self._fallback_task: asyncio.Task[None] | None = None

    async def start(self) -> None:
        if self._started:
            return
        self._loop = asyncio.get_running_loop()

        def on_frame(frame: dict[str, Any]) -> None:
            self._frame_count += 1
            self._last_cdp_frame_time = asyncio.get_running_loop().time()
            self.publish({
                "data": frame.get("data", ""),
                "metadata": frame.get("metadata", {}),
            })
            session_id = frame.get("sessionId")
            if session_id is not None:
                self._schedule_ack(session_id)

        self._listener = on_frame
        try:
            await self.session.send("Page.enable")
            self.session.on("Page.screencastFrame", on_frame)
            await self.session.send(
                "Page.startScreencast",
                {
                    "format": "jpeg",
                    "quality": self.quality,
                    "maxWidth": self.max_width,
                    "maxHeight": self.max_height,
                    "everyNthFrame": 1,
                },
            )
        except Exception as exc:  # noqa: BLE001 - CDP failure must leave screenshot fallback available.
            logger.warning("CDP screencast start failed: %s; relying on screenshot fallback", exc)
        self._started = True
        if self.page is not None:
            self._fallback_task = asyncio.create_task(self._screenshot_fallback())

    async def stop(self) -> None:
        if not self._started:
            return
        self._started = False
        if self._fallback_task is not None:
            self._fallback_task.cancel()
            try:
                await self._fallback_task
            except asyncio.CancelledError:
                pass
            self._fallback_task = None
        if self._listener is not None:
            self.session.remove_listener("Page.screencastFrame", self._listener)
            self._listener = None
        try:
            await self.session.send("Page.stopScreencast")
        except PlaywrightError:
            # The browser may already have closed the target while shutting down.
            pass

    def _schedule_ack(self, session_id: int) -> None:
        loop = self._loop
        if loop is None or loop.is_closed():
            return

        def enqueue() -> None:
            loop.create_task(self._ack(session_id))

        try:
            running = asyncio.get_running_loop()
        except RuntimeError:
            running = None
        if running is loop:
            enqueue()
        else:
            loop.call_soon_threadsafe(enqueue)

    async def _ack(self, session_id: int) -> None:
        try:
            await self.session.send("Page.screencastFrameAck", {"sessionId": session_id})
        except PlaywrightError:
            pass

    async def _screenshot_fallback(self) -> None:
        try:
            # Brief pause to let initial page setup start
            await asyncio.sleep(0.15)
            while self._started and self.page is not None:
                loop = asyncio.get_running_loop()
                # If CDP delivered a frame in the last 400ms, back off to avoid duplicate frames
                if self._last_cdp_frame_time > 0 and (loop.time() - self._last_cdp_frame_time) < 0.4:
                    await asyncio.sleep(0.15)
                    continue
                try:
                    image = await self.page.screenshot(
                        type="jpeg",
                        quality=self.quality,
                        timeout=5_000,
                    )
                except PlaywrightError:
                    await asyncio.sleep(0.5)
                    continue
                except Exception:  # noqa: BLE001 - transient screenshot failures should not stop the stream.
                    await asyncio.sleep(0.5)
                    continue
                if not self._started:
                    return
                self.publish({
                    "data": base64.b64encode(image).decode("ascii"),
                    "metadata": {"source": "screenshot"},
                })
                await asyncio.sleep(0.35)
        except asyncio.CancelledError:
            return
