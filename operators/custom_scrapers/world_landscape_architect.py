# operators/custom_scrapers/world_landscape_architect.py
"""
World Landscape Architect Custom Scraper - HTTP Pattern + AI Verification
Scrapes landscape architecture news from World Landscape Architect

Site: https://worldlandscapearchitect.com/
Strategy: Extract links matching article pattern + AI verification for ambiguous URLs

Pattern Analysis:
- Article URLs: /article-title-with-hyphens/ (long slugs with multiple hyphens)
- Non-article URLs: 
  - /landscape-architect/* (firm profiles)
  - /job/* (job listings)
  - /category/* (category pages)
  - /about/, /shop/, /contact-us/, etc. (static pages)
  - /urbastyle/, /mmcite/, etc. (company profiles)

Special consideration:
- All URLs are similar slugs, so we use AI to verify if URL looks like an article title
- Article titles typically have 5+ words (long descriptive phrases)

Architecture (Simplified):
- Custom scraper discovers article URLs from homepage (no article page visits)
- Quick AI check for ambiguous URLs (short slugs that could be articles or categories)
- Article tracker handles new/seen filtering (with TEST_MODE support)
- Main pipeline handles: content scraping, hero image extraction (og:image), AI filtering

Usage:
    scraper = WorldLandscapeArchitectScraper()
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


class WorldLandscapeArchitectScraper(BaseCustomScraper):
    """
    HTTP pattern-based custom scraper for World Landscape Architect.
    Extracts article URLs from homepage with AI verification for ambiguous slugs.
    """

    source_id = "world_landscape_architect"
    source_name = "World Landscape Architect"
    base_url = "https://worldlandscapearchitect.com/"

    # Configuration
    MAX_NEW_ARTICLES = 10

    # Known non-article path patterns (static pages, sections, profiles)
    EXCLUDED_PATH_PATTERNS = [
        r'^/landscape-architect/',   # Firm profiles
        r'^/job/',                   # Job listings
        r'^/job-listing/',           # Job submission
        r'^/category/',              # Category pages
        r'^/shop/',                  # Shop pages
        r'^/cart/',                  # Cart
        r'^/checkout/',              # Checkout
        r'^/about/',                 # About pages
        r'^/contact-us/',            # Contact
        r'^/advertise/',             # Advertise
        r'^/submissions/',           # Submissions
        r'^/support-wla/',           # Support page
        r'^/supporters/',            # Supporters
        r'^/product/',               # Product pages
        r'^/design-discipline/',     # Design discipline section
        r'^/editor-posts/',          # Editor posts section
        r'^/review/',                # Reviews section
        r'^/student/',               # Student section
        r'^/general/',               # General section
        r'^/privacy-policy',         # Privacy policy
        r'^/disclaimer/',            # Disclaimer
        r'^/refunds-policy/',        # Refunds
        r'^/individual-membership/', # Membership pages
        r'^/design-firm-membership/',
        r'^/product-service-membership/',
        r'^/\d{4}/\d{2}/$',          # Date archive pages like /2026/01/
        r'^/[^/]+/$',                # Single word paths (likely company profiles like /urbastyle/, /mmcite/)
    ]
    
    # Known company profile slugs (single-word paths that are NOT articles)
    COMPANY_PROFILE_SLUGS = [
        'urbastyle', 'mmcite', 'streetlife', 'cracknell', 'vestre',
        'landscape-forms', 'maglin', 'scenic', 'benoy', 'felixx',
        'hassell', 'sasaki', 'stoss', 'arcadia', 'arup', 'rios'
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
        for pattern in self.EXCLUDED_PATH_PATTERNS:
            if re.match(pattern, path, re.IGNORECASE):
                return True
        return False
    
    def _is_company_profile(self, slug: str) -> bool:
        """Check if slug is a known company profile."""
        # Remove leading/trailing slashes
        clean_slug = slug.strip('/').lower()
        return clean_slug in self.COMPANY_PROFILE_SLUGS
    
    def _looks_like_article_title(self, slug: str) -> bool:
        """
        Heuristic check if a slug looks like an article title.
        Article titles typically have 5+ words (many hyphens).
        """
        # Remove leading/trailing slashes
        clean_slug = slug.strip('/')
        
        # Count hyphens (word separators)
        hyphen_count = clean_slug.count('-')
        
        # Article titles typically have 4+ hyphens (5+ words)
        # E.g., "a-new-rhythm-for-the-waterfront-the-evolution-of-sausalitos-ferry-landing"
        return hyphen_count >= 4

    def _is_valid_article_url(self, path: str) -> bool:
        """
        Check if URL path is likely a valid article URL.
        """
        # Must start with /
        if not path.startswith('/'):
            return False
            
        # Exclude known non-article patterns
        if self._is_excluded_path(path):
            return False
            
        # Get the slug (path without leading/trailing slashes)
        slug = path.strip('/')
        
        # Skip if empty or contains query parameters
        if not slug or '?' in slug or '#' in slug:
            return False
            
        # Skip if it's a known company profile
        if self._is_company_profile(slug):
            return False
            
        # Skip if it has nested paths (like /category/featured/)
        if '/' in slug:
            return False
            
        # Must look like an article title (5+ words)
        return self._looks_like_article_title(slug)

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
            
            # Skip empty, external, or special links
            if not href or href.startswith(('#', 'javascript:', 'mailto:')):
                continue
                
            # Parse URL
            parsed = urlparse(href)
            
            # Only process internal links
            if parsed.netloc and 'worldlandscapearchitect.com' not in parsed.netloc:
                continue
            
            # Get the path
            path = parsed.path
            if not path:
                continue
                
            # Normalize path
            if not path.endswith('/'):
                path = path + '/'
            if not path.startswith('/'):
                path = '/' + path
            
            # Check if it's a valid article URL
            if self._is_valid_article_url(path):
                full_url = urljoin(self.base_url, path)
                
                if full_url not in seen_urls:
                    seen_urls.add(full_url)
                    
                    # Try to get title from link text or nearby elements
                    title = link.get_text(strip=True)
                    if not title or len(title) < 10:
                        # Try parent element
                        parent = link.find_parent(['h1', 'h2', 'h3', 'h4', 'article', 'div'])
                        if parent:
                            title_el = parent.find(['h1', 'h2', 'h3', 'h4'])
                            if title_el:
                                title = title_el.get_text(strip=True)
                    
                    # Fall back to slug if no title found
                    if not title or len(title) < 10:
                        slug = path.strip('/').split('/')[-1]
                        title = slug.replace('-', ' ').title()
                    
                    articles.append((full_url, title))

        return articles

    async def fetch_articles(self, hours: int = 24) -> list[dict]:
        """
        Fetch new articles from World Landscape Architect homepage.
        
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
custom_scraper_registry.register(WorldLandscapeArchitectScraper)


# =============================================================================
# Standalone Test
# =============================================================================

async def test_world_landscape_architect_scraper():
    """Test the HTTP pattern scraper."""
    print("=" * 60)
    print("Testing World Landscape Architect HTTP Pattern Scraper")
    print("=" * 60)

    # Show TEST_MODE status
    from storage.article_tracker import ArticleTracker
    print(f"\nTEST_MODE: {ArticleTracker.TEST_MODE}")
    if ArticleTracker.TEST_MODE:
        print("   All articles will appear as 'new' (ignoring database)")
    else:
        print("   Normal mode - filtering seen articles")

    scraper = WorldLandscapeArchitectScraper()

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
    asyncio.run(test_world_landscape_architect_scraper())
