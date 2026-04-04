from __future__ import annotations

import asyncio
import hashlib
import json
import logging
import os
import re
import threading
from collections.abc import Callable
from dataclasses import dataclass
from html.parser import HTMLParser
from pathlib import Path
from urllib.parse import urljoin, urlparse
from urllib.request import Request, urlopen
from uuid import uuid4
import xml.etree.ElementTree as ET

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Optional Playwright import – gracefully degrade to urllib BFS if absent.
# Install with: pip install playwright && playwright install chromium
# ---------------------------------------------------------------------------
try:
    from playwright.async_api import async_playwright as _async_playwright  # noqa: F401
    _PLAYWRIGHT_AVAILABLE = True
except ImportError:
    _PLAYWRIGHT_AVAILABLE = False
    logger.warning(
        "playwright is not installed. JS-rendered (Puppeteer-style) crawling is "
        "unavailable; the crawler will fall back to plain urllib link-discovery. "
        "To enable it: pip install playwright && playwright install chromium"
    )

try:
    from playwright_stealth import stealth_async as _stealth_async
    _STEALTH_AVAILABLE = True
except ImportError:
    _STEALTH_AVAILABLE = False
    logger.warning(
        "playwright-stealth is not installed. Crawling may fail on Cloudflare-protected "
        "sites. To enable it: pip install playwright-stealth"
    )

USER_AGENT = (
    "Mozilla/5.0 (compatible; ChatbotKnowledgeCrawler/1.0; +https://localhost)"
)

# Resource types to block inside the Playwright browser context.
# NOTE: Do NOT block "stylesheet" — many SPAs and JS-heavy sites rely on
# CSS classes to render visible content; blocking CSS breaks their rendering.
_BLOCK_RESOURCE_TYPES: frozenset[str] = frozenset({
    "image", "media", "font",
})


# ---------------------------------------------------------------------------
# Data-transfer objects
# ---------------------------------------------------------------------------

@dataclass(slots=True)
class WebsiteDocument:
    source_file: str
    text: str
    # doc_id = root source URL shared by all pages in this crawl.
    # Enables surgical deletion of all page chunks via a single Qdrant filter.
    doc_id: str = ""


@dataclass(slots=True)
class CrawledPage:
    url: str
    title: str
    text: str


@dataclass(slots=True)
class CrawledWebsite:
    source_url: str
    display_name: str
    pages: list[CrawledPage]


@dataclass(slots=True)
class WebsiteSourceSummary:
    source_url: str
    display_name: str
    page_count: int
    page_urls: list[str]


@dataclass(slots=True)
class WebsitePageRecord:
    index: int
    url: str
    title: str
    text: str


# ---------------------------------------------------------------------------
# Service
# ---------------------------------------------------------------------------

class WebsiteService:
    def __init__(self, website_directory: str | Path) -> None:
        self.website_directory = Path(website_directory)
        self.max_pages = int(os.getenv("MAX_CRAWL_PAGES", "200"))
        self.request_timeout = float(os.getenv("CRAWL_TIMEOUT_SECONDS", "20"))
        # Per-page browser navigation timeout used by the Playwright crawler.
        self.puppeteer_page_timeout = float(
            os.getenv("CRAWL_PUPPETEER_TIMEOUT_SECONDS", "30")
        )

    # ------------------------------------------------------------------
    # Public document / storage helpers
    # ------------------------------------------------------------------

    def load_documents(self) -> list[WebsiteDocument]:
        if not self.website_directory.exists():
            return []

        documents: list[WebsiteDocument] = []
        for path in sorted(self.website_directory.glob("*.json")):
            try:
                payload = json.loads(path.read_text(encoding="utf-8"))
            except Exception:
                continue

            root_source_url = str(payload.get("source_url", "")).strip()

            for page in payload.get("pages", []):
                text = str(page.get("text", "")).strip()
                if not text:
                    continue
                title = str(
                    page.get("title")
                    or page.get("url")
                    or payload.get("display_name")
                    or "Website page"
                )
                url = str(page.get("url", "")).strip()
                source = f"{title} ({url})" if url else title
                documents.append(
                    WebsiteDocument(
                        source_file=source,
                        text=text,
                        doc_id=root_source_url,
                    )
                )

        return documents

    def list_sources(self) -> dict[str, WebsiteSourceSummary]:
        if not self.website_directory.exists():
            return {}

        sources: dict[str, WebsiteSourceSummary] = {}
        for path in sorted(self.website_directory.glob("*.json")):
            try:
                payload = json.loads(path.read_text(encoding="utf-8"))
            except Exception:
                continue

            source_url = str(payload.get("source_url", "")).strip()
            if not source_url:
                continue

            page_urls = [
                str(page.get("url", "")).strip()
                for page in payload.get("pages", [])
                if str(page.get("url", "")).strip()
            ]
            sources[source_url] = WebsiteSourceSummary(
                source_url=source_url,
                display_name=str(payload.get("display_name") or source_url),
                page_count=len(page_urls),
                page_urls=page_urls,
            )

        return sources

    # ------------------------------------------------------------------
    # Primary crawl entry-point
    # ------------------------------------------------------------------

    def crawl_in_batches(
        self,
        source_url: str,
        batch_size: int = 50,
        progress_callback: Callable[[dict[str, int | str | None]], None] | None = None,
    ):
        """
        Generator version of crawl() — yields pages in batches of `batch_size`
        so callers can ingest each batch immediately without waiting for the
        entire crawl to finish.

        Strategy (same waterfall as crawl()):
          1. Sitemap found  → fetch all URLs, yield in slices of batch_size
          2. No sitemap + Playwright available → BFS with per-batch yielding
          3. No sitemap + no Playwright → urllib BFS with per-batch yielding

        Yields:
            list[CrawledPage] — one batch at a time (may be smaller than
            batch_size for the final batch or when pages are exhausted).
        """
        import queue as _queue

        normalized_url = self._normalize_url(source_url)

        # ── Step 1: URL itself might be a sitemap ─────────────────────────
        try:
            content, content_type = self._fetch(normalized_url)
        except Exception as exc:
            raise ValueError(f"Unable to reach {normalized_url}: {exc}") from exc

        if self._looks_like_sitemap(normalized_url, content_type, content):
            urls = self._extract_sitemap_urls(content, normalized_url)
            if urls:
                yield from self._yield_sitemap_batches(urls, batch_size, normalized_url, progress_callback)
                return

        # ── Step 2: Discover and use the site's sitemap ───────────────────
        self._notify_progress(
            progress_callback, stage="discovery", discovered_pages=0,
            indexed_pages=0, current_url=normalized_url,
            message="Searching for sitemap.xml…",
        )
        sitemap_url = self._discover_sitemap_url(normalized_url)
        if sitemap_url:
            try:
                raw_xml, _ = self._fetch(sitemap_url)
                if self._looks_like_sitemap(sitemap_url, "", raw_xml):
                    urls = self._extract_sitemap_urls(raw_xml, sitemap_url)
                    if urls:
                        yield from self._yield_sitemap_batches(urls, batch_size, sitemap_url, progress_callback)
                        return
            except Exception as exc:
                logger.debug("Sitemap crawl failed (%s): %s", sitemap_url, exc)

        # ── Step 3: Dynamic fallback with streaming batches ───────────────
        page_queue: _queue.Queue = _queue.Queue()

        def _streaming_progress(update: dict) -> None:
            if progress_callback:
                progress_callback(update)

        # ── True streaming: accumulate pages in the worker thread and flush
        # a batch into page_queue every batch_size pages.  This means
        # ingestion starts as soon as the first batch_size pages are crawled
        # rather than waiting for the entire crawl to finish first.
        import threading as _threading
        errors: list[BaseException] = []
        all_pages: list[CrawledPage] = []

        def _worker() -> None:
            current_batch: list[CrawledPage] = []

            def _on_page(page: CrawledPage) -> None:
                """Called by the BFS crawler for every successfully crawled page."""
                current_batch.append(page)
                if len(current_batch) >= batch_size:
                    page_queue.put(current_batch[:])
                    current_batch.clear()

            try:
                if _PLAYWRIGHT_AVAILABLE:
                    self._notify_progress(
                        progress_callback, stage="crawling", discovered_pages=0,
                        indexed_pages=0, current_url=normalized_url,
                        message="Sitemap unavailable. Switching to browser-rendered crawl…",
                    )
                    self._crawl_site_puppeteer(
                        normalized_url, _streaming_progress, page_callback=_on_page
                    )
                else:
                    self._notify_progress(
                        progress_callback, stage="crawling", discovered_pages=0,
                        indexed_pages=0, current_url=normalized_url,
                        message="Sitemap unavailable. Starting link-discovery crawl…",
                    )
                    self._crawl_site(
                        normalized_url, _streaming_progress, page_callback=_on_page
                    )
                # Flush any remaining pages that didn't fill a full batch
                if current_batch:
                    page_queue.put(current_batch[:])
            except Exception as exc:
                errors.append(exc)
            finally:
                page_queue.put(None)  # sentinel — signals end of crawl

        thread = _threading.Thread(target=_worker, daemon=True)
        thread.start()

        total_indexed = 0
        while True:
            batch = page_queue.get()
            if batch is None:
                break
            if batch:
                all_pages.extend(batch)
                total_indexed += len(batch)
                yield batch

        thread.join(timeout=5)
        if errors:
            raise errors[0]

        if not all_pages:
            raise ValueError("No readable website content was found at the provided URL.")

    def _yield_sitemap_batches(
        self,
        urls: list[str],
        batch_size: int,
        sitemap_url: str,
        progress_callback: Callable[[dict[str, int | str | None]], None] | None,
    ):
        """Fetch sitemap URLs in batches, yielding each batch after fetching."""
        total = min(len(urls), self.max_pages)
        self._notify_progress(
            progress_callback, stage="crawling",
            discovered_pages=total, indexed_pages=0,
            current_url=sitemap_url,
            message=f"Sitemap found — crawling {total} URLs in batches.",
        )
        indexed = 0
        batch: list[CrawledPage] = []
        for url in urls[: self.max_pages]:
            try:
                html, _ = self._fetch(url)
            except Exception as exc:
                logger.debug("Skipping %s: %s", url, exc)
                continue
            page = self._parse_html_page(url, html)
            if page is not None:
                batch.append(page)
                indexed += 1
                self._notify_progress(
                    progress_callback, stage="crawling",
                    discovered_pages=total, indexed_pages=indexed,
                    current_url=url,
                    message=f"Crawled {indexed} of {total} pages.",
                )
                if len(batch) >= batch_size:
                    yield batch
                    batch = []
        if batch:
            yield batch

    def crawl(
        self,
        source_url: str,
        progress_callback: Callable[[dict[str, int | str | None]], None] | None = None,
    ) -> CrawledWebsite:
        """
        Crawl strategy (in priority order):

        1. If the supplied URL *is* a sitemap → parse it directly and crawl
           each listed URL with a plain HTTP fetch.

        2. Discover the site's sitemap via robots.txt or well-known paths
           (/sitemap.xml, /sitemap_index.xml, /sitemap-index.xml).
           If found and non-empty → crawl every URL listed there directly.

        3. Fallback A – Playwright (Puppeteer-style): if the sitemap is
           unavailable or empty and playwright is installed, launch a headless
           Chromium browser to fully render JS-heavy pages and discover links
           dynamically through BFS.

        4. Fallback B – urllib BFS: if playwright is not installed, fall back
           to the plain link-discovery crawler using urllib (no JS rendering).
        """
        normalized_url = self._normalize_url(source_url)

        # ── Step 1: URL itself might be a sitemap ─────────────────────────────
        try:
            content, content_type = self._fetch(normalized_url)
        except Exception as exc:
            raise ValueError(
                f"Unable to reach {normalized_url}: {exc}"
            ) from exc

        if self._looks_like_sitemap(normalized_url, content_type, content):
            pages = self._crawl_sitemap(
                normalized_url, content, progress_callback=progress_callback
            )
            if pages:
                return CrawledWebsite(
                    source_url=normalized_url,
                    display_name=pages[0].title or normalized_url,
                    pages=pages,
                )

        # ── Step 2: Discover and use the site's sitemap ───────────────────────
        self._notify_progress(
            progress_callback,
            stage="discovery",
            discovered_pages=0,
            indexed_pages=0,
            current_url=normalized_url,
            message="Searching for sitemap.xml…",
        )

        sitemap_url = self._discover_sitemap_url(normalized_url)
        if sitemap_url:
            try:
                raw_xml, _ = self._fetch(sitemap_url)
                if self._looks_like_sitemap(sitemap_url, "", raw_xml):
                    pages = self._crawl_sitemap(
                        sitemap_url, raw_xml, progress_callback=progress_callback
                    )
                    if pages:
                        return CrawledWebsite(
                            source_url=normalized_url,
                            display_name=pages[0].title or normalized_url,
                            pages=pages,
                        )
            except Exception as exc:
                logger.debug("Sitemap crawl failed (%s): %s", sitemap_url, exc)

        # ── Step 3: Dynamic fallback ──────────────────────────────────────────
        if _PLAYWRIGHT_AVAILABLE:
            self._notify_progress(
                progress_callback,
                stage="crawling",
                discovered_pages=0,
                indexed_pages=0,
                current_url=normalized_url,
                message=(
                    "Sitemap unavailable or empty. "
                    "Switching to browser-rendered (Puppeteer) crawl…"
                ),
            )
            pages = self._crawl_site_puppeteer(
                normalized_url, progress_callback=progress_callback
            )
        else:
            self._notify_progress(
                progress_callback,
                stage="crawling",
                discovered_pages=0,
                indexed_pages=0,
                current_url=normalized_url,
                message=(
                    "Sitemap unavailable or empty. "
                    "Starting link-discovery crawl…"
                ),
            )
            pages = self._crawl_site(
                normalized_url, progress_callback=progress_callback
            )

        if not pages:
            raise ValueError(
                "No readable website content was found at the provided URL."
            )

        return CrawledWebsite(
            source_url=normalized_url,
            display_name=pages[0].title or normalized_url,
            pages=pages,
        )

    # ------------------------------------------------------------------
    # Persistence helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _page_hash(text: str) -> str:
        """MD5 hash of page text — used to detect content changes on re-crawl."""
        return hashlib.md5(text.encode("utf-8", errors="ignore")).hexdigest()

    def _merge_and_save_pages(
        self,
        source_url: str,
        pages: list[CrawledPage],
    ) -> dict[str, int]:
        """
        Incrementally merge a batch of CrawledPage objects into the on-disk
        JSON file for source_url.  Creates the file if it does not exist yet.
        Safe to call multiple times — new pages are upserted, existing ones
        are updated in place.

        Each page record stores a ``text_hash`` field (MD5 of page text) so
        that scheduled re-crawls can skip re-indexing pages whose content has
        not changed.

        Returns a dict with counts:
            added   — pages that did not exist before
            changed — pages that existed but whose text changed
            unchanged — pages whose text is identical to the stored version
        """
        self.website_directory.mkdir(parents=True, exist_ok=True)
        normalized_source_url = self._normalize_url(source_url)
        path = self._find_source_path(normalized_source_url)

        if path is not None:
            try:
                payload = json.loads(path.read_text(encoding="utf-8"))
            except Exception:
                payload = {
                    "source_url": normalized_source_url,
                    "display_name": normalized_source_url,
                    "pages": [],
                }
        else:
            file_name = f"{self._slugify(normalized_source_url)}-{uuid4().hex[:8]}.json"
            path = self.website_directory / file_name
            payload = {
                "source_url": normalized_source_url,
                "display_name": normalized_source_url,
                "pages": [],
            }

        # Build a lookup of existing pages by URL for O(1) change detection
        existing: dict[str, dict] = {
            self._normalize_url(str(p.get("url", "")).strip()): p
            for p in payload.get("pages", [])
            if str(p.get("url", "")).strip()
        }

        added = changed = unchanged = 0

        for page in pages:
            norm_url = self._normalize_url(page.url)
            new_hash = self._page_hash(page.text)
            record = {
                "url": norm_url,
                "title": page.title.strip() or norm_url,
                "text": page.text.strip(),
                "text_hash": new_hash,
            }
            if norm_url not in existing:
                added += 1
            elif existing[norm_url].get("text_hash", "") != new_hash:
                changed += 1
            else:
                unchanged += 1
            existing[norm_url] = record

        payload["source_url"] = normalized_source_url
        payload["pages"] = list(existing.values())

        # Update display name from first page with a real title
        if payload.get("display_name") in ("", normalized_source_url):
            for p in pages:
                if p.title and p.title.strip() and p.title.strip() != p.url:
                    payload["display_name"] = p.title.strip()
                    break

        path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
        return {"added": added, "changed": changed, "unchanged": unchanged}

    def get_changed_pages(self, source_url: str) -> tuple[list[CrawledPage], list[CrawledPage]]:
        """
        After a re-crawl, compare the newly-crawled pages against the
        stored hashes to return only changed/new pages for re-indexing.

        Returns:
            (changed_pages, all_stored_pages)
            changed_pages — pages that are new or whose text changed
            all_stored_pages — all pages currently stored on disk
        """
        normalized_source_url = self._normalize_url(source_url)
        path = self._find_source_path(normalized_source_url)
        if path is None:
            return [], []

        try:
            payload = json.loads(path.read_text(encoding="utf-8"))
        except Exception:
            return [], []

        all_stored: list[CrawledPage] = []
        changed: list[CrawledPage] = []

        # Pages without a text_hash were stored before change detection was
        # added — treat them as changed so they get re-indexed once.
        for p in payload.get("pages", []):
            page = CrawledPage(
                url=str(p.get("url", "")).strip(),
                title=str(p.get("title", "")).strip(),
                text=str(p.get("text", "")).strip(),
            )
            all_stored.append(page)
            if not p.get("text_hash"):
                changed.append(page)

        return changed, all_stored

    def save_crawl(self, crawl: CrawledWebsite) -> Path:
        self.website_directory.mkdir(parents=True, exist_ok=True)
        normalized_source_url = self._normalize_url(crawl.source_url)
        path = self._find_source_path(normalized_source_url)
        if path is not None:
            try:
                payload = json.loads(path.read_text(encoding="utf-8"))
            except Exception:
                payload = {
                    "source_url": normalized_source_url,
                    "display_name": crawl.display_name,
                    "pages": [],
                }
        else:
            file_name = (
                f"{self._slugify(normalized_source_url)}-{uuid4().hex[:8]}.json"
            )
            path = self.website_directory / file_name
            payload = {
                "source_url": normalized_source_url,
                "display_name": crawl.display_name,
                "pages": [],
            }

        payload["source_url"] = normalized_source_url
        payload["display_name"] = crawl.display_name
        payload["pages"] = self._merge_pages(
            existing_pages=list(payload.get("pages", [])),
            crawled_pages=crawl.pages,
        )
        path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
        return path

    def delete_source(self, source_url: str) -> None:
        self._delete_existing_source(source_url)

    def list_source_pages(self, source_url: str) -> list[WebsitePageRecord]:
        _, payload = self._load_source_payload(source_url)
        pages = payload.get("pages", [])
        return [
            WebsitePageRecord(
                index=index,
                url=str(page.get("url", "")).strip(),
                title=str(page.get("title", "")).strip(),
                text=str(page.get("text", "")).strip(),
            )
            for index, page in enumerate(pages)
        ]

    def create_source_page(
        self, source_url: str, *, url: str, title: str, text: str
    ) -> WebsitePageRecord:
        path, payload = self._load_source_payload(source_url)
        pages = list(payload.get("pages", []))
        normalized_url = self._normalize_url(url)
        if self._find_page_index_by_url(pages, normalized_url) is not None:
            raise ValueError("This page URL has already been crawled.")
        record = {
            "url": normalized_url,
            "title": title.strip() or url.strip(),
            "text": text.strip(),
        }
        pages.append(record)
        payload["pages"] = pages
        self._write_source_payload(path, payload)
        return WebsitePageRecord(index=len(pages) - 1, **record)

    def crawl_single_page(self, url: str) -> CrawledPage:
        """
        Fetch and parse exactly ONE page.

        Strategy:
          1. Try plain urllib first (fast, no browser needed).
          2. If the page is empty/JS-rendered and Playwright is available,
             fall back to a headless browser render of just that one URL.
        """
        normalized_url = self._normalize_url(url)

        # ── Try plain HTTP first ──────────────────────────────────────────
        try:
            html, content_type = self._fetch(normalized_url)
            if "html" in content_type.lower() or "<html" in html.lower():
                page = self._parse_html_page(normalized_url, html)
                if page is not None and len(page.text.strip()) > 200:
                    return page
        except Exception:
            html = ""

        # ── Fallback to Playwright for JS-heavy pages ─────────────────────
        if _PLAYWRIGHT_AVAILABLE:
            logger.debug(
                "Plain fetch returned thin content for %s — "
                "trying Playwright single-page render.",
                normalized_url,
            )
            pages = self._crawl_site_puppeteer(normalized_url)
            if pages:
                return pages[0]

        # ── Final fallback: return whatever urllib got ─────────────────────
        if html:
            page = self._parse_html_page(normalized_url, html)
            if page is not None:
                return page

        raise ValueError(
            "No readable page content was found at the provided URL."
        )

    def update_source_page(
        self,
        source_url: str,
        page_index: int,
        *,
        url: str,
        title: str,
        text: str,
    ) -> WebsitePageRecord:
        path, payload = self._load_source_payload(source_url)
        pages = list(payload.get("pages", []))
        if page_index < 0 or page_index >= len(pages):
            raise IndexError("Website page not found.")
        normalized_url = self._normalize_url(url)
        existing_index = self._find_page_index_by_url(pages, normalized_url)
        if existing_index is not None and existing_index != page_index:
            raise ValueError("This page URL has already been crawled.")

        updated = {
            "url": normalized_url,
            "title": title.strip() or url.strip(),
            "text": text.strip(),
        }
        pages[page_index] = updated
        payload["pages"] = pages
        self._write_source_payload(path, payload)
        return WebsitePageRecord(index=page_index, **updated)

    def delete_source_page(self, source_url: str, page_index: int) -> None:
        path, payload = self._load_source_payload(source_url)
        pages = list(payload.get("pages", []))
        if page_index < 0 or page_index >= len(pages):
            raise IndexError("Website page not found.")
        pages.pop(page_index)
        payload["pages"] = pages
        self._write_source_payload(path, payload)

    # ------------------------------------------------------------------
    # Sitemap crawl
    # ------------------------------------------------------------------

    def _crawl_sitemap(
        self,
        sitemap_url: str,
        raw_xml: str,
        progress_callback: Callable[[dict[str, int | str | None]], None] | None = None,
    ) -> list[CrawledPage]:
        """
        Parse the sitemap XML, extract all page URLs, then fetch and parse
        each URL directly with a plain HTTP request.

        Supports both <urlset> (standard sitemap) and <sitemapindex> (sitemap
        of sitemaps) formats, including recursive nesting.
        """
        urls = self._extract_sitemap_urls(raw_xml, sitemap_url)
        if not urls:
            return []

        pages: list[CrawledPage] = []
        total_pages = min(len(urls), self.max_pages)
        self._notify_progress(
            progress_callback,
            stage="crawling",
            discovered_pages=total_pages,
            indexed_pages=0,
            current_url=sitemap_url,
            message=f"Sitemap found — crawling {total_pages} URLs directly.",
        )

        for url in urls[: self.max_pages]:
            try:
                html, _ = self._fetch(url)
            except Exception as exc:
                logger.debug("Skipping %s: %s", url, exc)
                continue
            page = self._parse_html_page(url, html)
            if page is not None:
                pages.append(page)
                self._notify_progress(
                    progress_callback,
                    stage="crawling",
                    discovered_pages=total_pages,
                    indexed_pages=len(pages),
                    current_url=url,
                    message=f"Crawled {len(pages)} of {total_pages} pages.",
                )

        return pages

    def _discover_sitemap_url(self, start_url: str) -> str | None:
        """
        Locate the site's sitemap by checking:
          1. The Sitemap directive in robots.txt
          2. Well-known sitemap paths: /sitemap.xml, /sitemap_index.xml,
             /sitemap-index.xml
        Returns the first valid sitemap URL found, or None.
        """
        parsed = urlparse(start_url)
        base = f"{parsed.scheme}://{parsed.netloc}"

        # 1. robots.txt Sitemap directive
        robots_url = urljoin(base, "/robots.txt")
        try:
            robots_text, _ = self._fetch(robots_url)
        except Exception:
            robots_text = ""

        for line in robots_text.splitlines():
            if line.lower().startswith("sitemap:"):
                candidate = line.split(":", 1)[1].strip()
                if candidate:
                    return self._normalize_url(candidate)

        # 2. Well-known paths
        default_candidates = [
            urljoin(base, "/sitemap.xml"),
            urljoin(base, "/sitemap_index.xml"),
            urljoin(base, "/sitemap-index.xml"),
        ]
        for candidate in default_candidates:
            try:
                raw_xml, content_type = self._fetch(candidate)
            except Exception:
                continue
            if self._looks_like_sitemap(candidate, content_type, raw_xml):
                return candidate

        return None

    def _extract_sitemap_urls(self, raw_xml: str, source_url: str) -> list[str]:
        """
        Recursively extract all page URLs from a sitemap or sitemap index.
        Handles the standard XML namespace used by most sitemap generators.
        """
        try:
            root = ET.fromstring(raw_xml)
        except ET.ParseError:
            return []

        ns = ""
        if root.tag.startswith("{"):
            ns = root.tag.split("}")[0] + "}"

        # Sitemap index → recurse into nested sitemaps
        if root.tag.endswith("sitemapindex"):
            urls: list[str] = []
            for sitemap_loc in root.findall(f".//{ns}sitemap/{ns}loc"):
                nested_url = (sitemap_loc.text or "").strip()
                if not nested_url:
                    continue
                try:
                    nested_xml, _ = self._fetch(nested_url)
                except Exception:
                    continue
                urls.extend(self._extract_sitemap_urls(nested_xml, nested_url))
                if len(urls) >= self.max_pages:
                    break
            return urls[: self.max_pages]

        # Standard urlset
        if root.tag.endswith("urlset"):
            return [
                loc_text
                for loc in root.findall(f".//{ns}url/{ns}loc")
                if (loc_text := (loc.text or "").strip())
            ][: self.max_pages]

        # Unknown XML root – treat the source URL itself as the only entry
        return [source_url]

    # ------------------------------------------------------------------
    # Playwright (Puppeteer-style) dynamic crawler
    # ------------------------------------------------------------------

    def _crawl_site_puppeteer(
        self,
        start_url: str,
        progress_callback: Callable[[dict[str, int | str | None]], None] | None = None,
        page_callback: Callable[[CrawledPage], None] | None = None,
    ) -> list[CrawledPage]:
        """
        Synchronous wrapper for the async Playwright crawler.

        Runs the async crawl in a dedicated OS thread so that asyncio.run()
        can create its own event loop without conflicting with FastAPI /
        uvicorn's already-running loop in the main thread.

        page_callback — called immediately for every crawled page so that
        crawl_in_batches() can stream pages into ingestion without waiting
        for the full crawl to finish.
        """
        result: list[CrawledPage] = []
        errors: list[BaseException] = []

        def _run() -> None:
            try:
                pages = asyncio.run(
                    self._async_puppeteer_crawl(start_url, progress_callback, page_callback)
                )
                result.extend(pages)
            except Exception as exc:  # noqa: BLE001
                errors.append(exc)

        thread = threading.Thread(target=_run, daemon=True)
        thread.start()
        # Generous wall-clock budget: per-page timeout × (capped pages) + headroom
        wall_timeout = self.puppeteer_page_timeout * min(self.max_pages, 50) + 120
        thread.join(timeout=wall_timeout)

        if thread.is_alive():
            logger.warning(
                "Playwright crawl thread exceeded %.0f s — returning %d partial pages.",
                wall_timeout,
                len(result),
            )

        if errors:
            raise errors[0]

        return result

    async def _async_puppeteer_crawl(
        self,
        start_url: str,
        progress_callback: Callable[[dict[str, int | str | None]], None] | None = None,
        page_callback: Callable[[CrawledPage], None] | None = None,
    ) -> list[CrawledPage]:
        """
        Async BFS crawler powered by Playwright / headless Chromium.

        For every URL in the queue:
          1. Navigate the browser to the page (JS is fully executed).
          2. Collect all <a href> links from the *rendered* DOM before any
             mutation, so dynamic menus and JS-injected links are captured.
          3. Strip noisy layout elements (nav, header, footer, script…) from
             the DOM and capture body.innerText as clean page text.
          4. Enqueue discovered same-domain links that haven't been visited.

        Heavy static assets (images, fonts, media) are intercepted and aborted
        at the browser context level to keep navigation fast.
        """
        from playwright.async_api import (  # type: ignore[import]
            async_playwright,
            TimeoutError as PwTimeoutError,
        )

        # Strip leading "www." so that links on mosil.com are accepted when
        # the crawl started at www.mosil.com (and vice-versa).
        raw_origin = urlparse(start_url).netloc
        origin = self._canonical_netloc(raw_origin)

        queue: list[str] = [start_url]
        seen: set[str] = set()
        pages: list[CrawledPage] = []
        nav_timeout_ms = int(self.puppeteer_page_timeout * 1000)

        self._notify_progress(
            progress_callback,
            stage="crawling",
            discovered_pages=1,
            indexed_pages=0,
            current_url=start_url,
            message="Launching headless browser for dynamic crawl…",
        )

        async with async_playwright() as pw:
            # --no-sandbox is required when running as root inside Docker.
            # --disable-blink-features=AutomationControlled removes the
            # navigator.webdriver=true flag that Cloudflare and other WAFs
            # check to detect headless browser automation.
            browser = await pw.chromium.launch(
                headless=True,
                args=[
                    "--no-sandbox",
                    "--disable-setuid-sandbox",
                    "--disable-blink-features=AutomationControlled",
                    "--disable-dev-shm-usage",
                ],
            )

            # Mimic a real desktop browser as closely as possible.
            # Many bot-detection systems (Cloudflare, Akamai, etc.) check
            # these headers and viewport values in addition to navigator.webdriver.
            ctx = await browser.new_context(
                user_agent=(
                    "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
                    "AppleWebKit/537.36 (KHTML, like Gecko) "
                    "Chrome/122.0.0.0 Safari/537.36"
                ),
                java_script_enabled=True,
                accept_downloads=False,
                viewport={"width": 1280, "height": 800},
                locale="en-US",
                timezone_id="America/New_York",
                extra_http_headers={
                    "Accept-Language": "en-US,en;q=0.9",
                    "Accept": (
                        "text/html,application/xhtml+xml,"
                        "application/xml;q=0.9,image/avif,"
                        "image/webp,*/*;q=0.8"
                    ),
                },
            )

            # Patch navigator.webdriver to undefined so Cloudflare/WAF
            # JS checks cannot detect the Playwright automation context.
            await ctx.add_init_script("""
                Object.defineProperty(navigator, 'webdriver', {
                    get: () => undefined,
                });
            """)

            # Abort heavy static resources to keep navigation snappy.
            # IMPORTANT: route.abort() is a coroutine — it MUST be awaited
            # inside an async handler. A sync lambda that calls route.abort()
            # without await creates a coroutine object that is immediately
            # garbage-collected, so nothing is ever blocked and every page
            # hangs waiting for images/fonts/media to load.
            async def _block_heavy(route, request) -> None:
                if request.resource_type in _BLOCK_RESOURCE_TYPES:
                    await route.abort()
                else:
                    await route.continue_()

            await ctx.route("**/*", _block_heavy)

            try:
                while queue and len(pages) < self.max_pages:
                    current_url = queue.pop(0)
                    if current_url in seen:
                        continue
                    seen.add(current_url)

                    pw_page = await ctx.new_page()

                    # Apply stealth patches on every page to remove all
                    # automation signals that Cloudflare / WAFs check.
                    # This patches navigator.webdriver, plugins, languages,
                    # WebGL fingerprint, and ~20 other detection vectors.
                    if _STEALTH_AVAILABLE:
                        await _stealth_async(pw_page)

                    try:
                        # Use "load" (fires after all sync scripts run) rather
                        # than "networkidle" — sites that load CDN assets like
                        # Tailwind Browser CDN keep the network busy indefinitely,
                        # so networkidle never fires and wastes the full timeout
                        # on every single page.
                        # Fall back to domcontentloaded if load also times out.
                        response = None
                        try:
                            response = await pw_page.goto(
                                current_url,
                                timeout=nav_timeout_ms,
                                wait_until="load",
                            )
                        except PwTimeoutError:
                            response = await pw_page.goto(
                                current_url,
                                timeout=nav_timeout_ms,
                                wait_until="domcontentloaded",
                            )
                    except Exception as exc:
                        logger.debug(
                            "Playwright navigation failed for %s: %s",
                            current_url,
                            exc,
                        )
                        await pw_page.close()
                        continue

                    try:
                        final_url = self._normalize_url(pw_page.url or current_url)
                    except Exception:
                        final_url = current_url

                    if response is not None and int(response.status) >= 400:
                        logger.debug(
                            "Skipping %s due to HTTP status %s after render.",
                            final_url,
                            response.status,
                        )
                        await pw_page.close()
                        continue

                    # Single JS evaluate: collect links then clean + extract text.
                    # One round-trip is faster than two separate evaluate() calls.
                    try:
                        extracted: dict = await pw_page.evaluate(
                            """
                            () => {
                                // 1. Collect all hrefs BEFORE any DOM mutation so
                                //    JS-injected navigation links are included.
                                const links = Array.from(
                                    document.querySelectorAll('a[href]')
                                ).map(el => el.href).filter(Boolean);

                                // 2. Remove layout chrome and non-content elements
                                //    to produce clean, signal-rich body text.
                                const noisyTags = [
                                    'script', 'style', 'noscript',
                                    'nav', 'header', 'footer', 'aside',
                                    'form', 'button', 'select', 'option',
                                    'iframe', 'menu',
                                ];
                                noisyTags.forEach(tag =>
                                    document.querySelectorAll(tag)
                                            .forEach(el => el.remove())
                                );

                                // 3. innerText returns only visible, rendered text —
                                //    no raw HTML or hidden elements.
                                const text = document.body
                                    ? document.body.innerText
                                    : '';

                                return { text, links };
                            }
                            """
                        )
                    except Exception as exc:
                        logger.debug(
                            "JS evaluate failed for %s: %s", current_url, exc
                        )
                        await pw_page.close()
                        continue

                    try:
                        title = await pw_page.title()
                    except Exception:
                        title = final_url

                    await pw_page.close()

                    # Store page only if it yielded useful text content
                    # and is not a bot-detection / challenge page.
                    text = self._clean_text(extracted.get("text", ""))
                    if self._is_bot_challenge(title, text):
                        logger.warning(
                            "Bot-detection page detected at %s (title=%r) — "
                            "skipping. The site may require stealth-playwright "
                            "or manual cookie injection to bypass Cloudflare.",
                            final_url,
                            title,
                        )
                        continue

                    if self._is_soft_404(final_url, title, text):
                        logger.debug(
                            "Soft-404 page detected at %s (title=%r) - skipping.",
                            final_url,
                            title,
                        )
                        continue

                    if text:
                        logger.debug(
                            "Crawled page: %s | title=%r | text_len=%d",
                            final_url, title, len(text),
                        )
                        page = CrawledPage(
                            url=final_url,
                            title=title or final_url,
                            text=text,
                        )
                        pages.append(page)
                        # Notify the batch streamer immediately so ingestion
                        # can start without waiting for the full crawl.
                        if page_callback is not None:
                            page_callback(page)
                        discovered = max(len(seen) + len(queue), len(pages))
                        self._notify_progress(
                            progress_callback,
                            stage="crawling",
                            discovered_pages=min(discovered, self.max_pages),
                            indexed_pages=len(pages),
                            current_url=final_url,
                            message=f"Crawled {len(pages)} pages so far.",
                        )
                    else:
                        logger.debug(
                            "Empty text for %s (title=%r) — skipping.",
                            final_url, title,
                        )

                    # Enqueue newly discovered same-domain links.
                    # Strip www. from discovered link domains too so that
                    # www.mosil.com and mosil.com are treated as the same site.
                    for href in extracted.get("links", []):
                        absolute = self._normalize_link(final_url, href)
                        if not absolute:
                            continue
                        link_origin = self._canonical_netloc(urlparse(absolute).netloc)
                        if link_origin != origin:
                            continue
                        if absolute not in seen and absolute not in queue:
                            queue.append(absolute)

            finally:
                await browser.close()

        return pages

    # ------------------------------------------------------------------
    # urllib BFS fallback (used when Playwright is not installed)
    # ------------------------------------------------------------------

    def _crawl_site(
        self,
        start_url: str,
        progress_callback: Callable[[dict[str, int | str | None]], None] | None = None,
        page_callback: Callable[[CrawledPage], None] | None = None,
    ) -> list[CrawledPage]:
        """
        Plain BFS crawler using urllib.  No JS execution – pages that depend
        on client-side rendering will return incomplete text.  Used only when
        playwright is not installed.

        page_callback — called immediately for every crawled page so that
        crawl_in_batches() can stream pages into ingestion without waiting
        for the full crawl to finish.
        """
        # Strip www. so mosil.com and www.mosil.com are treated as same site
        origin = self._canonical_netloc(urlparse(start_url).netloc)
        queue = [start_url]
        seen: set[str] = set()
        pages: list[CrawledPage] = []

        self._notify_progress(
            progress_callback,
            stage="crawling",
            discovered_pages=1,
            indexed_pages=0,
            current_url=start_url,
            message="Starting link-discovery crawl (no browser rendering).",
        )

        while queue and len(pages) < self.max_pages:
            current = queue.pop(0)
            if current in seen:
                continue
            seen.add(current)

            try:
                html, content_type = self._fetch(current)
            except Exception:
                continue

            if "html" not in content_type.lower() and "<html" not in html.lower():
                continue

            extractor = _HTMLContentParser()
            extractor.feed(html)
            text = self._clean_text(extractor.text())
            if text:
                title = extractor.title or current
                page = CrawledPage(url=current, title=title, text=text)
                pages.append(page)
                if page_callback is not None:
                    page_callback(page)
                discovered_pages = max(len(seen) + len(queue), len(pages))
                self._notify_progress(
                    progress_callback,
                    stage="crawling",
                    discovered_pages=min(discovered_pages, self.max_pages),
                    indexed_pages=len(pages),
                    current_url=current,
                    message=f"Crawled {len(pages)} pages so far.",
                )

            for link in extractor.links:
                absolute = self._normalize_link(current, link)
                if not absolute:
                    continue
                if self._canonical_netloc(urlparse(absolute).netloc) != origin:
                    continue
                if absolute not in seen and absolute not in queue:
                    queue.append(absolute)

        return pages

    # ------------------------------------------------------------------
    # Single-page HTML parsing helper
    # ------------------------------------------------------------------

    def _parse_html_page(self, url: str, html: str) -> CrawledPage | None:
        extractor = _HTMLContentParser()
        extractor.feed(html)
        text = self._clean_text(extractor.text())
        if not text:
            return None
        return CrawledPage(url=url, title=extractor.title or url, text=text)

    # ------------------------------------------------------------------
    # HTTP fetch
    # ------------------------------------------------------------------

    def _fetch(self, url: str) -> tuple[str, str]:
        request = Request(url, headers={"User-Agent": USER_AGENT})
        with urlopen(request, timeout=self.request_timeout) as response:
            raw = response.read()
            content_type = response.headers.get("Content-Type", "")
            charset = response.headers.get_content_charset() or "utf-8"
            return raw.decode(charset, errors="ignore"), content_type

    # ------------------------------------------------------------------
    # Storage internals
    # ------------------------------------------------------------------

    def _delete_existing_source(self, source_url: str) -> None:
        if not self.website_directory.exists():
            return
        for path in self.website_directory.glob("*.json"):
            try:
                payload = json.loads(path.read_text(encoding="utf-8"))
            except Exception:
                continue
            if str(payload.get("source_url", "")).strip() == source_url:
                path.unlink(missing_ok=True)

    def _load_source_payload(self, source_url: str) -> tuple[Path, dict]:
        normalized_source_url = self._normalize_url(source_url)
        if not self.website_directory.exists():
            raise FileNotFoundError(
                f"Website source '{normalized_source_url}' not found."
            )
        path = self._find_source_path(normalized_source_url)
        if path is not None:
            try:
                payload = json.loads(path.read_text(encoding="utf-8"))
            except Exception as exc:
                raise FileNotFoundError(
                    f"Website source '{normalized_source_url}' not found."
                ) from exc
            return path, payload
        raise FileNotFoundError(
            f"Website source '{normalized_source_url}' not found."
        )

    @staticmethod
    def _find_page_index_by_url(
        pages: list[dict], normalized_url: str
    ) -> int | None:
        for index, page in enumerate(pages):
            if str(page.get("url", "")).strip() == normalized_url:
                return index
        return None

    @staticmethod
    def _write_source_payload(path: Path, payload: dict) -> None:
        path.write_text(json.dumps(payload, indent=2), encoding="utf-8")

    def _find_source_path(self, source_url: str) -> Path | None:
        if not self.website_directory.exists():
            return None
        normalized_source_url = self._normalize_url(source_url)
        for path in self.website_directory.glob("*.json"):
            try:
                payload = json.loads(path.read_text(encoding="utf-8"))
            except Exception:
                continue
            if (
                self._normalize_url(str(payload.get("source_url", "")).strip())
                == normalized_source_url
            ):
                return path
        return None

    def _merge_pages(
        self,
        existing_pages: list[dict],
        crawled_pages: list[CrawledPage],
    ) -> list[dict]:
        merged_pages: list[dict] = []
        seen: dict[str, int] = {}

        for page in existing_pages:
            normalized_url = self._normalize_url(str(page.get("url", "")).strip())
            if normalized_url in seen:
                continue
            seen[normalized_url] = len(merged_pages)
            merged_pages.append(
                {
                    "url": normalized_url,
                    "title": str(page.get("title", "")).strip() or normalized_url,
                    "text": str(page.get("text", "")).strip(),
                }
            )

        for page in crawled_pages:
            normalized_url = self._normalize_url(page.url)
            record = {
                "url": normalized_url,
                "title": page.title.strip() or normalized_url,
                "text": page.text.strip(),
            }
            existing_index = seen.get(normalized_url)
            if existing_index is not None:
                merged_pages[existing_index] = record
            else:
                seen[normalized_url] = len(merged_pages)
                merged_pages.append(record)

        return merged_pages

    # ------------------------------------------------------------------
    # Static utilities
    # ------------------------------------------------------------------

    @staticmethod
    def _is_bot_challenge(title: str, text: str) -> bool:
        """
        Returns True if the page looks like a Cloudflare / WAF challenge page
        rather than real site content.  These pages should be skipped rather
        than stored as knowledge-base documents.
        """
        title_lower = (title or "").lower()
        text_lower = (text or "").lower()[:500]

        challenge_titles = {
            "just a moment",        # Cloudflare 5-second challenge
            "attention required",   # Cloudflare block page
            "access denied",
            "403 forbidden",
            "checking your browser",
            "please wait",
            "ddos protection",
            "security check",
            "bot check",
        }
        if any(phrase in title_lower for phrase in challenge_titles):
            return True

        challenge_body_phrases = (
            "checking your browser",
            "enable javascript and cookies",
            "cloudflare ray id",
            "please enable cookies",
            "ddos protection by cloudflare",
            "your ip has been blocked",
        )
        if any(phrase in text_lower for phrase in challenge_body_phrases):
            return True

        return False

    @staticmethod
    def _is_soft_404(url: str, title: str, text: str) -> bool:
        """
        Detect branded or SPA-rendered not-found pages that still contain text.
        """
        title_lower = (title or "").lower()
        text_lower = (text or "").lower()[:1200]
        url_lower = (url or "").lower()

        title_signals = (
            "404",
            "page not found",
            "not found",
        )
        body_signals = (
            "404",
            "page not found",
            "the page you are looking for",
            "this page could not be found",
            "sorry, the page you requested could not be found",
            "we can't find the page you're looking for",
            "the requested url was not found",
        )

        if any(signal in title_lower for signal in title_signals):
            return True
        if any(signal in text_lower for signal in body_signals):
            return True
        if any(token in url_lower for token in ("/404", "/not-found")):
            if "not found" in text_lower or "404" in text_lower:
                return True

        return False

    @staticmethod
    def _looks_like_sitemap(url: str, content_type: str, content: str) -> bool:
        lowered = url.lower()
        if "sitemap" in lowered or content_type.lower().startswith(
            ("application/xml", "text/xml")
        ):
            return True
        snippet = content.lstrip()[:200].lower()
        return (
            snippet.startswith("<?xml")
            or "<urlset" in snippet
            or "<sitemapindex" in snippet
        )

    @staticmethod
    def _normalize_url(url: str) -> str:
        url = url.strip()
        if not url:
            raise ValueError("Website URL is required.")
        parsed = urlparse(url)
        if not parsed.scheme:
            url = f"https://{url}"
            parsed = urlparse(url)

        normalized_path = parsed.path or "/"
        if normalized_path != "/" and normalized_path.endswith("/"):
            normalized_path = normalized_path.rstrip("/")

        normalized = parsed._replace(
            scheme=parsed.scheme.lower(),
            netloc=WebsiteService._canonical_netloc(parsed.netloc),
            path=normalized_path,
            fragment="",
        )
        return normalized.geturl()

    @staticmethod
    def _normalize_link(base_url: str, link: str) -> str | None:
        link = (link or "").strip()
        if not link or link.startswith(
            ("#", "mailto:", "tel:", "javascript:", "data:")
        ):
            return None
        absolute = urljoin(base_url, link)
        parsed = urlparse(absolute)
        if parsed.scheme not in {"http", "https"}:
            return None
        return WebsiteService._normalize_url(parsed._replace(fragment="").geturl())

    @staticmethod
    def _canonical_netloc(netloc: str) -> str:
        """
        Canonicalize hosts for storage and same-site checks.

        The crawler already treats apex and www as the same site while
        traversing links, so we canonicalize them here as well to prevent
        duplicate pages like example.com/ and www.example.com/.
        """
        normalized = (netloc or "").strip().lower()
        return normalized.removeprefix("www.")

    @staticmethod
    def _clean_text(text: str) -> str:
        text = text.replace("\r", "\n")
        text = re.sub(r"[ \t]+", " ", text)
        text = re.sub(r"\n{3,}", "\n\n", text)
        return text.strip()

    @staticmethod
    def _slugify(value: str) -> str:
        slug = re.sub(r"[^a-zA-Z0-9]+", "-", value).strip("-").lower()
        return slug[:80] or "website"

    @staticmethod
    def _notify_progress(
        callback: Callable[[dict[str, int | str | None]], None] | None,
        *,
        stage: str,
        discovered_pages: int,
        indexed_pages: int,
        current_url: str | None,
        message: str,
    ) -> None:
        if callback is None:
            return
        callback(
            {
                "stage": stage,
                "discovered_pages": discovered_pages,
                "indexed_pages": indexed_pages,
                "current_url": current_url,
                "message": message,
            }
        )


# ---------------------------------------------------------------------------
# HTML parser – used for sitemap-fetched pages and the urllib BFS fallback
# ---------------------------------------------------------------------------

class _HTMLContentParser(HTMLParser):
    # Layout / chrome tags whose content pollutes every page with repeated noise.
    _SKIP_TAGS: frozenset[str] = frozenset(
        {
            "script", "style", "noscript",
            "nav", "header", "footer", "aside",
            "form", "button", "select", "option",
            "iframe", "menu",
        }
    )

    _BLOCK_TAGS: frozenset[str] = frozenset(
        {
            "p", "div", "section", "article", "li",
            "br", "h1", "h2", "h3", "h4", "h5", "h6",
            "blockquote", "pre", "table", "tr", "td", "th",
        }
    )

    def __init__(self) -> None:
        super().__init__()
        self._text_parts: list[str] = []
        self.links: list[str] = []
        self.title = ""
        self._in_title = False
        self._skip_depth = 0

    def handle_starttag(
        self, tag: str, attrs: list[tuple[str, str | None]]
    ) -> None:
        # Fix 5: Collect <a href> links BEFORE the skip-depth check.
        # Previously, links inside <nav>, <header>, <footer> were silently
        # dropped because the skip-depth gate returned early before reaching
        # the anchor-extraction code.  On most websites, nav links are the
        # only way to discover product/service pages during urllib BFS crawl.
        if tag == "a":
            href = dict(attrs).get("href")
            if href:
                self.links.append(href)

        if tag in self._SKIP_TAGS:
            self._skip_depth += 1
            return
        if tag == "title":
            self._in_title = True
        if tag in self._BLOCK_TAGS:
            self._text_parts.append("\n")

    def handle_endtag(self, tag: str) -> None:
        if tag in self._SKIP_TAGS and self._skip_depth:
            self._skip_depth -= 1
            return
        if tag == "title":
            self._in_title = False
        if tag in self._BLOCK_TAGS:
            self._text_parts.append("\n")

    def handle_data(self, data: str) -> None:
        if self._skip_depth:
            return
        cleaned = data.strip()
        if not cleaned:
            return
        if self._in_title and not self.title:
            self.title = cleaned
        self._text_parts.append(cleaned)

    def text(self) -> str:
        return "".join(self._text_parts)
