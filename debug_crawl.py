"""
Run this from your backend folder to debug what Playwright sees on any site.

    python debug_crawl.py https://www.mosil.com/

It prints:
  - HTTP status of the page
  - Final URL after redirects
  - Page title
  - First 1000 chars of extracted text
  - First 20 links found
  - Raw HTML snippet (first 2000 chars) so you can see what Cloudflare returns
"""
import asyncio
import sys
import urllib.request

URL = sys.argv[1] if len(sys.argv) > 1 else "https://www.mosil.com/"

# ── 1. Plain urllib fetch ─────────────────────────────────────────────────────
print("\n" + "="*60)
print("STEP 1 — Plain urllib (no browser)")
print("="*60)
try:
    req = urllib.request.Request(
        URL,
        headers={
            "User-Agent": (
                "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
                "AppleWebKit/537.36 (KHTML, like Gecko) "
                "Chrome/122.0.0.0 Safari/537.36"
            )
        }
    )
    with urllib.request.urlopen(req, timeout=15) as r:
        body = r.read(2000).decode("utf-8", errors="ignore")
        print(f"Status      : {r.status}")
        print(f"Final URL   : {r.url}")
        print(f"Content-Type: {r.headers.get('Content-Type', '')}")
        print(f"Body[:500]  :\n{body[:500]}")
except Exception as e:
    print(f"FAILED: {type(e).__name__}: {e}")


# ── 2. Playwright fetch ───────────────────────────────────────────────────────
print("\n" + "="*60)
print("STEP 2 — Playwright headless Chromium")
print("="*60)


async def run():
    from playwright.async_api import async_playwright, TimeoutError as PwTimeout

    try:
        from playwright_stealth import stealth_async
        stealth_available = True
        print("playwright-stealth : AVAILABLE ✓")
    except ImportError:
        stealth_available = False
        print("playwright-stealth : NOT INSTALLED ✗")

    async with async_playwright() as pw:
        browser = await pw.chromium.launch(
            headless=True,
            args=[
                "--no-sandbox",
                "--disable-setuid-sandbox",
                "--disable-blink-features=AutomationControlled",
                "--disable-dev-shm-usage",
            ],
        )
        ctx = await browser.new_context(
            user_agent=(
                "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
                "AppleWebKit/537.36 (KHTML, like Gecko) "
                "Chrome/122.0.0.0 Safari/537.36"
            ),
            viewport={"width": 1280, "height": 800},
            locale="en-US",
            timezone_id="America/New_York",
            extra_http_headers={
                "Accept-Language": "en-US,en;q=0.9",
                "Accept": (
                    "text/html,application/xhtml+xml,application/xml;"
                    "q=0.9,image/avif,image/webp,*/*;q=0.8"
                ),
            },
        )

        await ctx.add_init_script("""
            Object.defineProperty(navigator, 'webdriver', { get: () => undefined });
        """)

        page = await ctx.new_page()

        if stealth_available:
            await stealth_async(page)
            print("Stealth applied    : YES ✓")
        else:
            print("Stealth applied    : NO ✗")

        # Capture HTTP response status
        response_info = {}
        page.on("response", lambda r: response_info.update(
            {"status": r.status, "url": r.url}
        ) if r.url == page.url or not response_info else None)

        # Navigate
        print(f"\nNavigating to: {URL}")
        try:
            resp = await page.goto(URL, timeout=30000, wait_until="networkidle")
            print(f"Response status : {resp.status if resp else 'N/A'}")
            print(f"Final URL       : {page.url}")
        except PwTimeout:
            print("networkidle timed out — trying domcontentloaded...")
            resp = await page.goto(URL, timeout=30000, wait_until="domcontentloaded")
            print(f"Response status : {resp.status if resp else 'N/A'}")
            print(f"Final URL       : {page.url}")

        title = await page.title()
        print(f"Page title      : {title!r}")

        # Check navigator.webdriver (should be undefined after stealth)
        webdriver_val = await page.evaluate("() => navigator.webdriver")
        print(f"navigator.webdriver: {webdriver_val!r}  (should be None/undefined)")

        # Raw HTML (first 2000 chars)
        html = await page.content()
        print(f"\nRaw HTML [:2000]:\n{html[:2000]}")

        # Extract text + links
        extracted = await page.evaluate("""
            () => {
                const links = Array.from(
                    document.querySelectorAll('a[href]')
                ).map(el => el.href).filter(Boolean);

                const noisyTags = [
                    'script','style','noscript',
                    'nav','header','footer','aside',
                    'form','button','select','option',
                    'iframe','menu',
                ];
                noisyTags.forEach(tag =>
                    document.querySelectorAll(tag).forEach(el => el.remove())
                );
                const text = document.body ? document.body.innerText : '';
                return { text, links };
            }
        """)

        text = extracted.get("text", "").strip()
        links = extracted.get("links", [])

        print(f"\nExtracted text length : {len(text)} chars")
        print(f"Extracted text [:1000]:\n{text[:1000]}")
        print(f"\nLinks found: {len(links)}")
        for link in links[:20]:
            print(f"  {link}")

        await browser.close()


asyncio.run(run())