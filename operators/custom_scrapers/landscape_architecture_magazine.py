# operators/custom_scrapers/landscape_architecture_magazine.py
"""
Landscape Architecture Magazine (LAM) Custom Scraper - HTTP Pattern Approach
Scrapes landscape architecture news from LAM (ASLA publication)

Site: https://landscapearchitecturemagazine.org/
Strategy: Extract links matching /YYYY/* pattern from HTML

Pattern Analysis:
- Article URLs: /YYYY/article-slug (e.g., /2025/ping-design-brings-extraordinary-moments-to-the-house-next-door)
- Article URLs: /YYYY/MM/article-slug (e.g., /2017/06/the-los-angeles-river-cut-loose)
- Non-article URLs: 
  - /about-lam, /all-articles, /project-categories
  - /search, /contact-us
  - /joan-nassauer-interview-profile (no year prefix)

Architecture (Simplified):
- Custom scraper discovers article URLs from homepage (no article page visits)
- Article tracker handles new/seen filtering (with TEST_MODE support)
- Main pipeline handles: content scraping, hero image extraction (og:image), AI filtering

On first run: All homepage articles marked as seen
On subsequent runs: Only new articles returned for processing

Usage:
    scraper = LandscapeArchitectureMagazineScraper()
    articles = await scraper.fetch_articles()
    await scraper.close()
"""

import asyncio
import re
from typing import Optional, List, Tuple
from urllib.parse import urljoin, urlparse

from bs4 import BeautifulSoup

from operators.custom_scraper_base import BaseCustomScraper, custom_scraper_registry
from storage.article_tracker import ArticleTracker


class LandscapeArchitectureMagazineScraper(BaseCustomScraper):
    """
    HTTP pattern-based custom scraper for Landscape Architecture Magazine.
    Extracts article URLs matching /YYYY/* or /YYYY/MM/* patterns from homepage.
    """

    source_id = "landscape_architecture_magazine"
    source_name = "Landscape Architecture Magazine"
    base_url = "https://landscapearchitecturemagazine.org/"

    # Configuration
    MAX_NEW_ARTICLES = 10

    # URL patterns for articles:
    # - /2025/article-slug
    # - /2017/06/article-slug
    # Matches years 2000-2099
    ARTICLE_PATTERNS = [
        re.compile(r'^/20\d{2}/\d{2}/[a-z0-9-]+/?$', re.IGNORECASE),  # /YYYY/MM/slug
        re.compile(r'^/20\d{2}/[a-z0-9-]+/?$', re.IGNORECASE),         # /YYYY/slug
    ]

    # URL patterns to exclude (not articles)
    EXCLUDED_PATTERNS = [
        r'^/about',           # About pages
        r'^/all-articles',    # Article listing
        r'^/project-',        # Project pages
        r'^/search',          # Search
        r'^/contact',         # Contact
        r'^/cdn-cgi/',        # CDN paths
        r'^/getContentAsset', # Asset paths
        r'^\?',               # Query strings
        r'^#',                # Anchors
    ]

    def __init__(self):
        """Initialize scraper with article tracker."""
        super().__init__()
        self.tracker: Optional[ArticleTracker] = None

    async def _ensure_tracker(self):
        """Ensure article tracker is connected."""
        if not self.tracker:
            self.tracker = ArticleTracker()
            await self.tracker.connect()

    def _is_excluded_path(self, path: str) -> bool:
        """Check if URL path matches an excluded pattern."""
        for pattern in self.EXCLUDED_PATTERNS:
            if re.match(pattern, path, re.IGNORECASE):
                return True
        return False

    def _is_valid_article_url(self, path: str) -> bool:
        """
        Check if URL path is a valid article URL.
        Must match /YYYY/slug or /YYYY/MM/slug pattern.
        """
        # Must start with /
        if not path.startswith('/'):
            return False
            
        # Check exclusions first
        if self._is_excluded_path(path):
            return False

        # Check against article patterns
        for pattern in self.ARTICLE_PATTERNS:
            if pattern.match(path):
                return True

        return False

    def _extract_article_links(self, html: str) -> List[Tuple[str, str]]:
        """
        Extract potential article links with titles from HTML.
        Returns list of (url, title) tuples.
        """
        soup = BeautifulSoup(html, 'html.parser')
        articles = []
        seen_urls = set()

        # Find all links
        for link in soup.find_all('a', href=True):
            href = link.get('href', '')
            
            # Skip empty or special links
            if not href or href.startswith(('#', 'javascript:', 'mailto:')):
                continue
                
            # Parse URL
            parsed = urlparse(href)
            
            # Only process internal links
            if parsed.netloc and 'landscapearchitecturemagazine.org' not in parsed.netloc:
                continue
            
            # Get the path
            path = parsed.path
            if not path:
                continue
            
            # Normalize path
            if not path.startswith('/'):
                path = '/' + path
            
            # Check if it's a valid article URL
            if self._is_valid_article_url(path):
                full_url = urljoin(self.base_url, path)
                
                # Normalize URL (remove trailing slash for consistency)
                full_url = full_url.rstrip('/')
                
                if full_url not in seen_urls:
                    seen_urls.add(full_url)
                    
                    # Try to get title from link text or nearby elements
                    title = link.get_text(strip=True)
                    
                    # If link text is too short, look for nearby heading
                    if not title or len(title) < 10:
                        parent = link.find_parent(['article', 'div', 'section'])
                        if parent:
                            title_el = parent.find(['h1', 'h2', 'h3', 'h4'])
                            if title_el:
                                title = title_el.get_text(strip=True)
                    
                    # Fall back to slug if no good title found
                    if not title or len(title) < 10:
                        slug = path.strip('/').split('/')[-1]
                        title = slug.replace('-', ' ').title()
                    
                    articles.append((full_url, title))

        return articles

    async def fetch_articles(self, hours: int = 24) -> list[dict]:
        """
        Fetch new articles from Landscape Architecture Magazine homepage.
        
        Returns minimal article dicts for main pipeline processing.
        """
        await self._ensure_tracker()
        await self._init_stats()

        print(f"\n[{self.source_id}] Starting article fetch...")

        try:
            # Fetch homepage HTML
            html = await self._fetch_html(self.base_url)
            if not html:
                print(f"[{self.source_id}] Failed to fetch homepage")
                return []

            # Extract article links
            extracted = self._extract_article_links(html)
            print(f"[{self.source_id}] Extracted {len(extracted)} potential article URLs")

            if not extracted:
                print(f"[{self.source_id}] No article links found")
                return []

            # Create URL to title mapping
            url_to_title = {url: title for url, title in extracted}
            all_urls = list(url_to_title.keys())

            # Filter for new articles using tracker
            new_urls = await self.tracker.filter_new_articles(
                source_id=self.source_id,
                urls=all_urls
            )

            print(f"[{self.source_id}] New articles: {len(new_urls)} of {len(all_urls)}")

            if not new_urls:
                print(f"[{self.source_id}] No new articles to process")
                if self.stats:
                    self.stats.log_final_count(0)
                    self.stats.print_summary()
                    await self._upload_stats_to_r2()
                return []

            # Mark new URLs as seen
            await self.tracker.mark_as_seen(self.source_id, new_urls)

            # Build article list
            new_articles = []
            for url in new_urls[:self.MAX_NEW_ARTICLES]:
                title = url_to_title.get(url, url.strip('/').split('/')[-1].replace('-', ' ').title())

                # Create minimal article dict
                article = self._create_minimal_article_dict(
                    title=title,
                    link=url,
                    published=None  # Will be extracted by main pipeline
                )

                if self._validate_article(article):
                    new_articles.append(article)
                    print(f"[{self.source_id}]    Added: {title[:60]}...")

            # Final Summary
            print(f"\n[{self.source_id}] Processing Summary:")
            print(f"   Articles found: {len(extracted)}")
            print(f"   New articles: {len(new_urls)}")
            print(f"   Returning to pipeline: {len(new_articles)}")

            # Log final count and upload stats
            if self.stats:
                self.stats.log_final_count(len(new_articles))
                self.stats.print_summary()
                await self._upload_stats_to_r2()

            return new_articles

        except Exception as e:
            print(f"[{self.source_id}] Error in scraping: {e}")
            if self.stats:
                self.stats.log_error(f"Critical error: {str(e)}")
                self.stats.print_summary()
                await self._upload_stats_to_r2()
            import traceback
            traceback.print_exc()
            return []

    async def close(self):
        """Close browser and tracker connections."""
        await super().close()

        if self.tracker:
            await self.tracker.close()
            self.tracker = None


# Register this scraper
custom_scraper_registry.register(LandscapeArchitectureMagazineScraper)


# =============================================================================
# Standalone Test
# =============================================================================

async def test_landscape_architecture_magazine_scraper():
    """Test the HTTP pattern scraper."""
    print("=" * 60)
    print("Testing Landscape Architecture Magazine HTTP Pattern Scraper")
    print("=" * 60)

    # Show TEST_MODE status
    from storage.article_tracker import ArticleTracker
    print(f"\nTEST_MODE: {ArticleTracker.TEST_MODE}")
    if ArticleTracker.TEST_MODE:
        print("   All articles will appear as 'new' (ignoring database)")
    else:
        print("   Normal mode - filtering seen articles")

    scraper = LandscapeArchitectureMagazineScraper()

    try:
        # Test connection
        print("\n1. Testing connection...")
        connected = await scraper.test_connection()

        if not connected:
            print("   Connection failed")
            return

        # Show tracker stats
        print("\n2. Checking article tracker stats...")
        await scraper._ensure_tracker()
        stats = await scraper.tracker.get_source_stats(scraper.source_id)
        print(f"   Previously seen: {stats['total_seen']} articles")
        print(f"   Seen today: {stats['seen_today']}")

        # Fetch articles
        print("\n3. Fetching articles...")
        articles = await scraper.fetch_articles()

        print(f"\n4. Results: {len(articles)} articles")
        for i, art in enumerate(articles, 1):
            print(f"\n   [{i}] {art['title'][:70]}...")
            print(f"       URL: {art['link']}")

    finally:
        await scraper.close()


if __name__ == "__main__":
    asyncio.run(test_landscape_architecture_magazine_scraper())
