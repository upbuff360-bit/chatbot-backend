from __future__ import annotations

import json
import os
import re
from collections.abc import Callable
from dataclasses import dataclass, field
from html.parser import HTMLParser
from pathlib import Path
from urllib.parse import urljoin, urlparse
from urllib.request import Request, urlopen
from uuid import uuid4
import xml.etree.ElementTree as ET


USER_AGENT = (
    "Mozilla/5.0 (compatible; ChatbotKnowledgeCrawler/1.0; +https://localhost)"
)


@dataclass(slots=True)
class WebsiteDocument:
    source_file: str
    text: str
    # FIX: doc_id = root source URL, shared by all pages in this crawl.
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


class WebsiteService:
    def __init__(self, website_directory: str | Path) -> None:
        self.website_directory = Path(website_directory)
        self.max_pages = int(os.getenv("MAX_CRAWL_PAGES", "200"))
        self.request_timeout = float(os.getenv("CRAWL_TIMEOUT_SECONDS", "20"))

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

            pages = payload.get("pages", [])
            for page in pages:
                text = str(page.get("text", "")).strip()
                if not text:
                    continue
                title = str(page.get("title") or page.get("url") or payload.get("display_name") or "Website page")
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

    def crawl(
        self,
        source_url: str,
        progress_callback: Callable[[dict[str, int | str | None]], None] | None = None,
    ) -> CrawledWebsite:
        normalized_url = self._normalize_url(source_url)
        content, content_type = self._fetch(normalized_url)

        if self._looks_like_sitemap(normalized_url, content_type, content):
            pages = self._crawl_sitemap(normalized_url, content, progress_callback=progress_callback)
        else:
            pages = self._crawl_site_or_sitemap(normalized_url, progress_callback=progress_callback)

        if not pages:
            raise ValueError("No readable website content was found at the provided URL.")

        display_name = pages[0].title or normalized_url
        return CrawledWebsite(source_url=normalized_url, display_name=display_name, pages=pages)

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
            file_name = f"{self._slugify(normalized_source_url)}-{uuid4().hex[:8]}.json"
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

    def create_source_page(self, source_url: str, *, url: str, title: str, text: str) -> WebsitePageRecord:
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
        normalized_url = self._normalize_url(url)
        html, content_type = self._fetch(normalized_url)
        if "html" not in content_type.lower() and "<html" not in html.lower():
            raise ValueError("The provided URL does not appear to contain readable HTML content.")

        page = self._parse_html_page(normalized_url, html)
        if page is None:
            raise ValueError("No readable page content was found at the provided URL.")
        return page

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

    def _crawl_site_or_sitemap(
        self,
        start_url: str,
        progress_callback: Callable[[dict[str, int | str | None]], None] | None = None,
    ) -> list[CrawledPage]:
        sitemap_url = self._discover_sitemap_url(start_url)
        if sitemap_url:
            try:
                raw_xml, content_type = self._fetch(sitemap_url)
                if self._looks_like_sitemap(sitemap_url, content_type, raw_xml):
                    pages = self._crawl_sitemap(sitemap_url, raw_xml, progress_callback=progress_callback)
                    if pages:
                        return pages
            except Exception:
                pass

        return self._crawl_site(start_url, progress_callback=progress_callback)

    def _crawl_sitemap(
        self,
        sitemap_url: str,
        raw_xml: str,
        progress_callback: Callable[[dict[str, int | str | None]], None] | None = None,
    ) -> list[CrawledPage]:
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
            message=f"Discovered {total_pages} pages from sitemap.",
        )

        for url in urls[: self.max_pages]:
            try:
                html, _ = self._fetch(url)
            except Exception:
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

    def _crawl_site(
        self,
        start_url: str,
        progress_callback: Callable[[dict[str, int | str | None]], None] | None = None,
    ) -> list[CrawledPage]:
        origin = urlparse(start_url).netloc
        queue = [start_url]
        seen: set[str] = set()
        pages: list[CrawledPage] = []

        self._notify_progress(
            progress_callback,
            stage="crawling",
            discovered_pages=1,
            indexed_pages=0,
            current_url=start_url,
            message="Starting website crawl.",
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
                pages.append(CrawledPage(url=current, title=title, text=text))
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
                parsed = urlparse(absolute)
                if parsed.netloc != origin:
                    continue
                if absolute not in seen and absolute not in queue:
                    queue.append(absolute)

        return pages

    def _parse_html_page(self, url: str, html: str) -> CrawledPage | None:
        extractor = _HTMLContentParser()
        extractor.feed(html)
        text = self._clean_text(extractor.text())
        if not text:
            return None
        return CrawledPage(url=url, title=extractor.title or url, text=text)

    def _fetch(self, url: str) -> tuple[str, str]:
        request = Request(url, headers={"User-Agent": USER_AGENT})
        with urlopen(request, timeout=self.request_timeout) as response:
            raw = response.read()
            content_type = response.headers.get("Content-Type", "")
            charset = response.headers.get_content_charset() or "utf-8"
            return raw.decode(charset, errors="ignore"), content_type

    def _extract_sitemap_urls(self, raw_xml: str, source_url: str) -> list[str]:
        try:
            root = ET.fromstring(raw_xml)
        except ET.ParseError:
            return []

        namespace = ""
        if root.tag.startswith("{"):
            namespace = root.tag.split("}")[0] + "}"

        if root.tag.endswith("sitemapindex"):
            urls: list[str] = []
            for sitemap in root.findall(f".//{namespace}sitemap/{namespace}loc"):
                nested_url = (sitemap.text or "").strip()
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

        if root.tag.endswith("urlset"):
            urls = [
                (loc.text or "").strip()
                for loc in root.findall(f".//{namespace}url/{namespace}loc")
                if (loc.text or "").strip()
            ]
            return urls[: self.max_pages]

        return [source_url]

    def _discover_sitemap_url(self, start_url: str) -> str | None:
        parsed = urlparse(start_url)
        base = f"{parsed.scheme}://{parsed.netloc}"

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
            raise FileNotFoundError(f"Website source '{normalized_source_url}' not found.")

        path = self._find_source_path(normalized_source_url)
        if path is not None:
            try:
                payload = json.loads(path.read_text(encoding="utf-8"))
            except Exception as exc:
                raise FileNotFoundError(f"Website source '{normalized_source_url}' not found.") from exc
            return path, payload

        raise FileNotFoundError(f"Website source '{normalized_source_url}' not found.")

    @staticmethod
    def _find_page_index_by_url(pages: list[dict], normalized_url: str) -> int | None:
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
            if self._normalize_url(str(payload.get("source_url", "")).strip()) == normalized_source_url:
                return path
        return None

    def _merge_pages(self, existing_pages: list[dict], crawled_pages: list[CrawledPage]) -> list[dict]:
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

    @staticmethod
    def _looks_like_sitemap(url: str, content_type: str, content: str) -> bool:
        lowered = url.lower()
        if "sitemap" in lowered or content_type.lower().startswith(("application/xml", "text/xml")):
            return True
        snippet = content.lstrip()[:200].lower()
        return snippet.startswith("<?xml") or "<urlset" in snippet or "<sitemapindex" in snippet

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
            netloc=parsed.netloc.lower(),
            path=normalized_path,
            fragment="",
        )
        return normalized.geturl()

    @staticmethod
    def _normalize_link(base_url: str, link: str) -> str | None:
        link = (link or "").strip()
        if not link or link.startswith(("#", "mailto:", "tel:", "javascript:")):
            return None
        absolute = urljoin(base_url, link)
        parsed = urlparse(absolute)
        if parsed.scheme not in {"http", "https"}:
            return None
        return parsed._replace(fragment="").geturl()

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


class _HTMLContentParser(HTMLParser):
    # FIX (Bug 1): Extended to skip nav/header/footer/aside/form — layout
    # chrome that pollutes every page with repeated navigation noise.
    _SKIP_TAGS: frozenset[str] = frozenset({
        "script", "style", "noscript",
        "nav", "header", "footer", "aside",
        "form", "button", "select", "option",
        "iframe", "menu",
    })

    _BLOCK_TAGS: frozenset[str] = frozenset({
        "p", "div", "section", "article", "li",
        "br", "h1", "h2", "h3", "h4", "h5", "h6",
        "blockquote", "pre", "table", "tr", "td", "th",
    })

    def __init__(self) -> None:
        super().__init__()
        self._text_parts: list[str] = []
        self.links: list[str] = []
        self.title = ""
        self._in_title = False
        self._skip_depth = 0

    def handle_starttag(self, tag: str, attrs: list[tuple[str, str | None]]) -> None:
        if tag in self._SKIP_TAGS:
            self._skip_depth += 1
            return
        if tag == "title":
            self._in_title = True
        if tag in self._BLOCK_TAGS:
            self._text_parts.append("\n")
        if tag == "a":
            href = dict(attrs).get("href")
            if href:
                self.links.append(href)

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
        # FIX (Bug 2): "".join preserves \n block boundaries.
        return "".join(self._text_parts)
