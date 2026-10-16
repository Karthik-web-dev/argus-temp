from __future__ import annotations

from pathlib import Path
from urllib.parse import urlsplit

from argus.agents.explorer_crawler import DeterministicCrawler
from argus.browser.session import BrowserSessionFactory
from argus_contracts import AppGraph


class RoutedSessionFactory(BrowserSessionFactory):
    """Serve a deterministic mini SPA without starting an external web server."""

    async def new_context(self, *args, **kwargs):
        session, cdp = await super().new_context(*args, **kwargs)

        async def serve(route):
            request = route.request
            path = urlsplit(request.url).path
            is_authenticated = "argus_auth=1" in request.headers.get("cookie", "")
            pages = {
                "/": """<h1>Home</h1>
                    <a href='/a'>A route</a><a href='/b'>B route</a>
                    <a href='/loop'>Loop</a>
                    <a href='https://outside.example/private'>External</a>
                    <button onclick=\"document.body.insertAdjacentHTML('beforeend',
                      '<section role=dialog><h2>Sign in panel</h2></section>')\">Open menu</button>
                    """,
                "/a": "<h1>Route A</h1><a href='/loop'>Loop</a>",
                "/b": "<h1>Route B</h1>",
                "/loop": "<h1>Loop</h1><a href='/loop'>Loop again</a>",
                "/private": "<h1>Private route</h1>",
            }
            body = pages.get(path)
            if path == "/" and is_authenticated:
                body += "<a href='/private'>Private route</a>"
            if body is None:
                await route.fulfill(status=404, body="missing")
            else:
                await route.fulfill(
                    status=200,
                    content_type="text/html",
                    body=f"<!doctype html><html><head><title>{path}</title></head><body>{body}</body></html>",
                )

        await session.page.route("http://crawler.test/**", serve)
        return session, cdp


async def test_crawler_discovers_routes_buttons_and_stops_at_caps(tmp_path: Path):
    actions: list[dict] = []
    async with RoutedSessionFactory(artifact_dir=tmp_path) as factory:
        crawler = DeterministicCrawler(
            session_factory=factory, max_pages=20, max_steps=100, timeout_ms=2000,
            on_action=actions.append,
        )
        output = tmp_path / "app_graph.json"
        graph = await crawler.crawl("http://crawler.test/", output)

    assert isinstance(graph, AppGraph)
    assert {urlsplit(node.url).path for node in graph.nodes} >= {"/", "/a", "/b", "/loop"}
    assert any(edge.action == "button" for edge in graph.edges)
    assert all(
        edge.action != "link" or edge.locator.startswith(("role=link", "css="))
        for edge in graph.edges
    )
    assert all("outside.example" not in edge.locator for edge in graph.edges)
    assert all("outside.example" not in node.url for node in graph.nodes)
    node_ids = {node.id for node in graph.nodes}
    assert all(edge.source in node_ids and edge.target in node_ids for edge in graph.edges)
    assert AppGraph.model_validate_json(output.read_text(encoding="utf-8")) == graph
    assert graph.adjacency
    assert any(
        action.get("action") == "click"
        and isinstance(action.get("x"), (int, float))
        and isinstance(action.get("y"), (int, float))
        for action in actions
    )


async def test_crawler_bootstraps_auth_and_reuses_storage_state(tmp_path: Path):
    async def authenticate(page):
        await page.context.add_cookies(
            [{"name": "argus_auth", "value": "1", "url": "http://crawler.test/"}]
        )

    async with RoutedSessionFactory(artifact_dir=tmp_path) as factory:
        crawler = DeterministicCrawler(
            session_factory=factory,
            auth_bootstrap=authenticate,
            max_pages=20,
            max_steps=100,
            timeout_ms=2000,
        )
        graph = await crawler.crawl("http://crawler.test/", tmp_path / "app_graph.json")

    assert any(node.authenticated for node in graph.nodes)
    assert any(urlsplit(node.url).path == "/private" for node in graph.nodes)
    assert (tmp_path / "storage_state.json").exists()


async def test_crawler_enforces_page_and_step_caps(tmp_path: Path):
    async with RoutedSessionFactory(artifact_dir=tmp_path) as factory:
        crawler = DeterministicCrawler(
            session_factory=factory, max_pages=2, max_steps=3, timeout_ms=2000
        )
        graph = await crawler.crawl("http://crawler.test/", tmp_path / "app_graph.json")
        assert crawler._step_count <= 3

    captured_nodes = [node for node in graph.nodes if node.captured]
    assert len({node.url for node in captured_nodes}) <= 2
    assert len(graph.edges) <= 3
    assert graph.adjacency
