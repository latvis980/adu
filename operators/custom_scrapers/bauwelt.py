# operators/custom_scrapers/bauwelt.py
"""
Bauwelt Custom Scraper - HTML Pattern + AI Filter Approach
Scrapes architecture news from Bauwelt (German architecture magazine)

Site: https://www.bauwelt.de/rubriken/bauten/standard_index_2073531.html
Strategy: Extract links matching /rubriken/bauten/ pattern, use AI to filter real articles

Pattern Analysis:
- Article links: /rubriken/bauten/Article-Name-Here-1234567.html
- Index pages: /rubriken/bauten/standard_index_2073531.html (to exclude)

Workflow:
1. Fetch page HTML
2. Extract all links matching /rubriken/bauten/ pattern
3. Use AI to filter: keep article links, exclude index/category pages
4. Check database for new URLs
5. For new articles: visit page to get publication date
6. Return minimal article dicts for main pipeline

Usage:
    scraper = BauweltScraper()
    articles = await scraper.fetch_articles()
    await scraper.close()
"""

import asyncio
import re
import os
from typing import Optional, List
from datetime import datetime, timezone
from urllib.parse import urljoin

from langchain_openai import ChatOpenAI
from langchain_core.messages import HumanMessage
from pydantic import SecretStr

from operators.custom_scraper_base import BaseCustomScraper, custom_scraper_registry
from storage.article_tracker import ArticleTracker


class BauweltScraper(BaseCustomScraper):
    """
    HTML pattern-based custom scraper for Bauwelt.
    Extracts article links from HTML and uses AI to filter real articles.
    """

    source_id = "bauwelt"
    source_name = "Bauwelt"
    base_url = "https://www.bauwelt.de/rubriken/bauten/standard_index_2073531.html"

    # Configuration
    MAX_ARTICLE_AGE_DAYS = 14
    MAX_NEW_ARTICLES = 10

    # URL pattern for buildings section
    ARTICLE_PATTERN = re.compile(r'/rubriken/bauten/[^"\'>\s]+\.html')

    def __init__(self):
        """Initialize scraper with article tracker and LLM."""
        super().__init__()
        self.tracker: Optional[ArticleTracker] = None
        self.llm: Optional[ChatOpenAI] = None

    async def _ensure_tracker(self):
        """Ensure article tracker is connected."""
        if not self.tracker:
            self.tracker = ArticleTracker()
            await self.tracker.connect()

    def _ensure_llm(self):
        """Ensure LLM is initialized."""
        if not self.llm:
            api_key = os.getenv("OPENAI_API_KEY")
            if not api_key:
                raise ValueError("OPENAI_API_KEY not set")

            self.llm = ChatOpenAI(
                model="gpt-4o-mini",
                api_key=SecretStr(api_key),
                temperature=0.1
            )
            print(f"[{self.source_id}] LLM initialized")

    def _extract_article_links(self, html: str) -> List[str]:
        """
        Extract all potential article links from HTML.
        
        Finds links matching /rubriken/bauten/*.html pattern.
        
        Args:
            html: Page HTML content
            
        Returns:
            List of unique URLs (absolute)
        """
        # Find all matching hrefs
        matches = self.ARTICLE_PATTERN.findall(html)
        
        # Convert to absolute URLs and deduplicate
        urls: set[str] = set()
        for path in matches:
            full_url = urljoin("https://www.bauwelt.de", path)
            urls.add(full_url)
        
        return list(urls)

    async def _filter_article_urls_with_ai(self, urls: List[str]) -> List[str]:
        """
        Use AI to filter real article URLs from index/category pages.
        
        Args:
            urls: List of URLs to filter
            
        Returns:
            List of valid article URLs
        """
        self._ensure_llm()
        
        if not urls or not self.llm:
            return []
        
        # Create prompt for AI filtering
        urls_text = "\n".join([f"{i+1}. {url}" for i, url in enumerate(urls)])
        
        prompt = f"""Analyze these URLs from Bauwelt (German architecture magazine) and identify which are REAL ARTICLE pages.

URLs:
{urls_text}

REAL ARTICLES have:
- Descriptive names with project/location/architect: /rubriken/bauten/Jenaplansschule-am-Hartwege-Weimar-4330561.html
- End with a numeric ID before .html

EXCLUDE (not articles):
- Index pages with "standard_index" in URL
- Category/navigation pages

Respond with ONLY the numbers of real articles, comma-separated.
Example: 1, 3, 5, 7

If no real articles found, respond: NONE"""

        try:
            response = self.llm.invoke([HumanMessage(content=prompt)])
            response_text = str(response.content).strip()
            
            if response_text.upper() == "NONE":
                return []
            
            # Parse response - extract numbers
            numbers = re.findall(r'\d+', response_text)
            
            # Convert to URLs (1-indexed in prompt)
            valid_urls: List[str] = []
            for num_str in numbers:
                idx = int(num_str) - 1
                if 0 <= idx < len(urls):
                    valid_urls.append(urls[idx])
            
            return valid_urls
            
        except Exception as e:
            print(f"[{self.source_id}] AI filtering error: {e}")
            return []

    async def _get_article_date(self, page, url: str) -> Optional[str]:
        """
        Visit article page and extract publication date using AI.
        
        Args:
            page: Playwright page object
            url: Article URL
            
        Returns:
            ISO format date string or None
        """
        try:
            await page.goto(url, timeout=self.timeout, wait_until="domcontentloaded")
            await page.wait_for_timeout(1000)
            
            # Extract text for date extraction
            article_text = await page.evaluate("""
                () => {
                    const article = document.querySelector('article, main, .content, .post, .article');
                    if (article) {
                        return article.textContent.substring(0, 2000);
                    }
                    return document.body.textContent.substring(0, 2000);
                }
            """)
            
            # Use base class AI date extraction
            return self._parse_date_with_ai(article_text)
            
        except Exception as e:
            print(f"[{self.source_id}] Date extraction error for {url}: {e}")
            return None

    async def fetch_articles(self, hours: int = 24) -> list[dict]:
        """
        Fetch new articles from Bauwelt buildings section.
        
        Workflow:
        1. Load page and extract all /rubriken/bauten/ links
        2. Use AI to filter real articles from index pages
        3. Check database for new URLs
        4. For new articles: get publication date
        5. Filter by date (within MAX_ARTICLE_AGE_DAYS)
        6. Return minimal article dicts
        
        Args:
            hours: Ignored (we use database tracking instead)
            
        Returns:
            List of article dicts for main pipeline
        """
        # Initialize statistics tracking
        self._init_stats()
        
        print(f"[{self.source_id}] Starting HTML pattern scraping...")
        
        await self._ensure_tracker()
        self._ensure_llm()
        
        try:
            page = await self._create_page()
            
            try:
                # ============================================================
                # Step 1: Load Page and Extract Links
                # ============================================================
                print(f"[{self.source_id}] Loading buildings section...")
                await page.goto(self.base_url, timeout=self.timeout, wait_until="networkidle")
                await page.wait_for_timeout(2000)
                
                # Get page HTML
                html = await page.content()
                
                # Extract all potential article links
                all_links = self._extract_article_links(html)
                print(f"[{self.source_id}] Found {len(all_links)} links matching /rubriken/bauten/ pattern")
                
                if not all_links:
                    print(f"[{self.source_id}] No links found")
                    if self.stats:
                        self.stats.log_final_count(0)
                        self.stats.print_summary()
                        await self._upload_stats_to_r2()
                    return []
                
                # ============================================================
                # Step 2: AI Filter - Real Articles vs Index Pages
                # ============================================================
                print(f"[{self.source_id}] Filtering with AI...")
                article_urls = await self._filter_article_urls_with_ai(all_links)
                
                print(f"[{self.source_id}] AI identified {len(article_urls)} real articles")
                
                if not article_urls:
                    print(f"[{self.source_id}] No real articles found after AI filtering")
                    if self.stats:
                        self.stats.log_final_count(0)
                        self.stats.print_summary()
                        await self._upload_stats_to_r2()
                    return []
                
                # Log URLs as "headlines" for stats
                if self.stats:
                    self.stats.log_headlines_extracted(article_urls)
                
                # ============================================================
                # Step 3: Check Database for New Articles
                # ============================================================
                if not self.tracker:
                    raise RuntimeError("Article tracker not initialized")
                
                # Get stored URLs
                seen_urls = await self.tracker.get_stored_headlines(self.source_id)
                seen_set = set(seen_urls)
                
                # Filter to new articles (not seen before)
                new_urls = [url for url in article_urls if url not in seen_set]
                
                print(f"[{self.source_id}] {len(new_urls)} new articles (not in database)")
                
                # Log new URLs
                if self.stats:
                    self.stats.log_new_headlines(new_urls, len(article_urls))
                
                if not new_urls:
                    print(f"[{self.source_id}] No new articles to process")
                    # Store all current URLs for future reference
                    await self.tracker.store_headlines(self.source_id, article_urls)
                    
                    if self.stats:
                        self.stats.log_final_count(0)
                        self.stats.print_summary()
                        await self._upload_stats_to_r2()
                    return []
                
                # Limit to max new articles
                urls_to_process = new_urls[:self.MAX_NEW_ARTICLES]
                
                # ============================================================
                # Step 4: Process Each New Article
                # ============================================================
                new_articles: list[dict] = []
                skipped_old = 0
                
                for i, url in enumerate(urls_to_process, 1):
                    # Extract title from URL for logging
                    url_title = url.split("/")[-1].replace("-", " ").replace(".html", "")
                    print(f"\n[{self.source_id}] Processing {i}/{len(urls_to_process)}: {url_title[:50]}...")
                    
                    try:
                        # Get publication date
                        published = await self._get_article_date(page, url)
                        
                        if self.stats:
                            if published:
                                self.stats.log_date_fetched(url_title, url, published)
                            else:
                                self.stats.log_date_fetch_failed(url_title)
                        
                        # Date filtering
                        if published:
                            try:
                                article_date = datetime.fromisoformat(published.replace('Z', '+00:00'))
                                current_date = datetime.now(timezone.utc)
                                days_old = (current_date - article_date).days
                                
                                if days_old > self.MAX_ARTICLE_AGE_DAYS:
                                    print(f"   Skipping old article ({days_old} days old)")
                                    skipped_old += 1
                                    continue
                                
                                print(f"   Fresh article ({days_old} day(s) old)")
                            except Exception as e:
                                print(f"   Date parsing error: {e} - including anyway")
                        else:
                            print(f"   No date found - including anyway")
                        
                        # Log successful processing
                        if self.stats:
                            self.stats.log_headline_matched(url_title, url)
                        
                        # Create minimal article dict
                        # Title will be properly extracted by main pipeline
                        article = self._create_minimal_article_dict(
                            title=url_title,  # Temporary, will be replaced by scraper
                            link=url,
                            published=published
                        )
                        
                        if self._validate_article(article):
                            new_articles.append(article)
                            
                            # Store URL in database
                            await self.tracker.update_headline_url(
                                self.source_id,
                                url_title,
                                url
                            )
                        
                        # Small delay between requests
                        await asyncio.sleep(0.5)
                        
                    except Exception as e:
                        print(f"   Error processing article: {e}")
                        if self.stats:
                            self.stats.log_error(f"Error processing '{url[-50:]}': {str(e)}")
                        continue
                
                # ============================================================
                # Step 5: Store All URLs and Finalize
                # ============================================================
                # Store all current URLs for future reference
                await self.tracker.store_headlines(self.source_id, article_urls)
                
                # Final Summary
                print(f"\n[{self.source_id}] Processing Summary:")
                print(f"   Links found: {len(all_links)}")
                print(f"   Real articles (AI filter): {len(article_urls)}")
                print(f"   New articles: {len(new_urls)}")
                print(f"   Skipped (too old): {skipped_old}")
                print(f"   Successfully scraped: {len(new_articles)}")
                
                # Log final count and upload stats
                if self.stats:
                    self.stats.log_final_count(len(new_articles))
                    self.stats.print_summary()
                    await self._upload_stats_to_r2()
                
                return new_articles
                
            finally:
                await page.close()
                
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
custom_scraper_registry.register(BauweltScraper)


# =============================================================================
# Standalone Test
# =============================================================================

async def test_bauwelt_scraper():
    """Test the HTML pattern scraper."""
    print("=" * 60)
    print("Testing Bauwelt HTML Pattern Scraper")
    print("=" * 60)
    
    scraper = BauweltScraper()
    
    try:
        # Test connection
        print("\n1. Testing connection...")
        connected = await scraper.test_connection()
        
        if not connected:
            print("   Connection failed")
            return
        
        # Show tracker stats
        print("\n2. Checking tracker stats...")
        await scraper._ensure_tracker()
        
        if scraper.tracker:
            stats = await scraper.tracker.get_stats(source_id="bauwelt")
            print(f"   Total articles in database: {stats['total_articles']}")
            if stats['oldest_seen']:
                print(f"   Oldest: {stats['oldest_seen']}")
            if stats['newest_seen']:
                print(f"   Newest: {stats['newest_seen']}")
        
        # Fetch new articles
        print("\n3. Running HTML pattern scraping...")
        articles = await scraper.fetch_articles(hours=24)
        
        print(f"\n   Found {len(articles)} NEW articles")
        
        # Display articles
        if articles:
            print("\n4. New articles:")
            for i, article in enumerate(articles, 1):
                print(f"\n   --- Article {i} ---")
                print(f"   Title: {article['title'][:60]}...")
                print(f"   Link: {article['link']}")
                print(f"   Published: {article.get('published', 'No date')}")
        else:
            print("\n4. No new articles (all previously seen)")
        
        print("\n" + "=" * 60)
        print("Test complete!")
        print("=" * 60)
        
    finally:
        await scraper.close()


if __name__ == "__main__":
    asyncio.run(test_bauwelt_scraper())
