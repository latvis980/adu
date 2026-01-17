# operators/custom_scrapers/domus.py
"""
Domus Custom Scraper - Visual AI Approach
Scrapes architecture news from Domus (Italian architecture magazine)

Site: https://www.domusweb.it/
Challenge: Use User-Agent as precaution

Visual Scraping Strategy:
1. Take screenshot of homepage
2. Use GPT-4o vision to extract article headlines
3. On first run: Store all headlines in database as "seen"
4. On subsequent runs: Only process NEW headlines (not in database)
5. Find headline text in HTML coupled with link using AI
6. Click link to get publication date and metadata
7. Continue with standard scraping logic

Usage:
    scraper = DomusScraper()
    articles = await scraper.fetch_articles()
    await scraper.close()
"""

import asyncio
import base64
from typing import Optional, List, cast
from datetime import datetime, timezone

from langchain_openai import ChatOpenAI
from langchain_core.messages import HumanMessage

from operators.custom_scraper_base import BaseCustomScraper, custom_scraper_registry
from storage.article_tracker import ArticleTracker
from prompts.homepage_analyzer import HOMEPAGE_ANALYZER_PROMPT_TEMPLATE, parse_headlines


class DomusScraper(BaseCustomScraper):
    """
    Visual AI-powered custom scraper for Domus
    Uses GPT-4o vision to identify articles on homepage.
    """

    source_id = "domus"
    source_name = "Domus"
    base_url = "https://www.domusweb.it/"

    MAX_ARTICLE_AGE_DAYS = 2

    def __init__(self):
        super().__init__()
        self.tracker: Optional[ArticleTracker] = None
        self.vision_model: Optional[ChatOpenAI] = None

    async def _ensure_tracker(self):
        if not self.tracker:
            self.tracker = ArticleTracker()
            await self.tracker.connect()

    def _ensure_vision_model(self):
        if not self.vision_model:
            import os
            api_key: Optional[str] = os.getenv("OPENAI_API_KEY")
            if not api_key:
                raise ValueError("OPENAI_API_KEY not set")
            api_key_str = cast(str, api_key)
            self.vision_model = ChatOpenAI(
                model="gpt-4o-mini",
                api_key=api_key_str,
                temperature=0.1
            )
            print(f"[{self.source_id}] Vision model initialized")

    async def _analyze_homepage_screenshot(self, screenshot_path: str) -> List[str]:
        self._ensure_vision_model()
        print(f"[{self.source_id}] 📸 Analyzing screenshot with AI vision...")
        
        with open(screenshot_path, 'rb') as f:
            image_data = base64.b64encode(f.read()).decode('utf-8')
        
        message = HumanMessage(
            content=[
                {"type": "text", "text": HOMEPAGE_ANALYZER_PROMPT_TEMPLATE.format()},
                {"type": "image_url", "image_url": {"url": f"data:image/png;base64,{image_data}"}}
            ]
        )
        
        if not self.vision_model:
            raise RuntimeError("Vision model not initialized")
        
        response = await asyncio.to_thread(self.vision_model.invoke, [message])
        response_text = response.content if hasattr(response, 'content') else str(response)
        if not isinstance(response_text, str):
            response_text = str(response_text)
        
        headlines = parse_headlines(response_text)
        print(f"[{self.source_id}] ✅ Extracted {len(headlines)} headlines from screenshot")
        return headlines

    async def _find_headline_in_html_with_ai(self, page, headline: str) -> Optional[dict]:
        self._ensure_vision_model()
        
        html_context = await page.evaluate("""
            (headline) => {
                const containers = document.querySelectorAll(
                    'article, .post, [class*="post"], [class*="item"], [class*="card"], [class*="entry"]'
                );
                const articleData = [];
                containers.forEach((container, index) => {
                    const links = container.querySelectorAll('a[href]');
                    if (links.length === 0) return;
                    let mainLink = null;
                    let mainLinkText = '';
                    links.forEach(link => {
                        const text = link.textContent.trim();
                        if (text.length > mainLinkText.length) {
                            mainLink = link;
                            mainLinkText = text;
                        }
                    });
                    if (!mainLink) return;
                    const href = mainLink.href;
                    const linkText = mainLinkText;
                    const descEl = container.querySelector('p, .excerpt, [class*="excerpt"], [class*="desc"]');
                    const description = descEl ? descEl.textContent.trim().substring(0, 150) : '';
                    const imgEl = container.querySelector('img');
                    const imageUrl = imgEl ? imgEl.src : null;
                    if (linkText.length > 5 && href.includes('/')) {
                        articleData.push({index, link_text: linkText, href, description, image_url: imageUrl});
                    }
                });
                return articleData;
            }
        """, headline)
        
        if not html_context:
            return None
        
        context_text = f"Looking for headline: '{headline}'\n\nArticle containers found on page:\n"
        for item in html_context[:15]:
            context_text += f"\n--- Container {item['index']} ---\n"
            context_text += f"Link text: {item['link_text']}\nURL: {item['href']}\n"
            if item['description']:
                context_text += f"Description: {item['description']}\n"
        
        prompt = f"""Given this headline: "{headline}"

Which of these article containers is the best match? Consider:
1. Semantic similarity (meaning, not just exact words)
2. Context clues (description, URL patterns)
3. Partial matches are OK if context is clear

{context_text}

Respond with ONLY the container index number (e.g., "3") or "NONE" if no good match.
Do not include any explanation."""
        
        if not self.vision_model:
            raise RuntimeError("Vision model not initialized")
        
        ai_response = await asyncio.to_thread(self.vision_model.invoke, [HumanMessage(content=prompt)])
        response_text = ai_response.content if hasattr(ai_response, 'content') else str(ai_response)
        if not isinstance(response_text, str):
            response_text = str(response_text)
        response_clean = response_text.strip().upper()
        
        if response_clean == "NONE":
            return None
        
        import re
        match = re.search(r'\d+', response_clean)
        if not match:
            return None
        
        selected_index = int(match.group(0))
        for item in html_context:
            if item['index'] == selected_index:
                return {'title': item['link_text'], 'link': item['href'], 
                        'description': item['description'], 'image_url': item['image_url']}
        return None

    async def fetch_articles(self, hours: int = 24) -> list[dict]:
        await self._ensure_tracker()
        try:
            page = await self._create_page()
            try:
                print(f"[{self.source_id}] Loading homepage...")
                await page.goto(self.base_url, timeout=self.timeout, wait_until="networkidle")
                await page.wait_for_timeout(2000)
                
                screenshot_path = f"/tmp/{self.source_id}_homepage.png"
                await page.screenshot(path=screenshot_path, full_page=False)
                print(f"[{self.source_id}] Screenshot saved")
                
                current_headlines = await self._analyze_homepage_screenshot(screenshot_path)
                if not current_headlines:
                    print(f"[{self.source_id}] No headlines found")
                    return []
                
                if not self.tracker:
                    raise RuntimeError("Article tracker not initialized")
                
                seen_headlines = await self.tracker.get_stored_headlines(self.source_id)
                new_headlines = [h for h in current_headlines if h not in seen_headlines]
                
                print(f"[{self.source_id}] Headlines: {len(current_headlines)} total, {len(new_headlines)} new")
                
                if not new_headlines:
                    return []
                
                MAX_NEW = 10
                if len(new_headlines) > MAX_NEW:
                    new_headlines = new_headlines[:MAX_NEW]
                
                new_articles = []
                skipped_old = skipped_no_link = 0
                
                for i, headline in enumerate(new_headlines, 1):
                    print(f"\n[{i}/{len(new_headlines)}] {headline[:60]}...")
                    try:
                        homepage_data = await self._find_headline_in_html_with_ai(page, headline)
                        if not homepage_data or not homepage_data.get('link'):
                            skipped_no_link += 1
                            continue
                        
                        url = homepage_data['link']
                        await page.goto(url, timeout=self.timeout)
                        await page.wait_for_timeout(1000)
                        
                        article_metadata = await page.evaluate("""
                            () => {
                                const dateEl = document.querySelector('time[datetime], .date, [class*="date"]');
                                const dateText = dateEl ? (dateEl.getAttribute('datetime') || dateEl.textContent.trim()) : '';
                                const ogImage = document.querySelector('meta[property="og:image"]');
                                return {date_text: dateText, hero_image_url: ogImage ? ogImage.content : null};
                            }
                        """)
                        
                        published = self._parse_date(article_metadata['date_text'])
                        if published:
                            days_old = (datetime.now(timezone.utc) - datetime.fromisoformat(published.replace('Z', '+00:00'))).days
                            if days_old > self.MAX_ARTICLE_AGE_DAYS:
                                skipped_old += 1
                                continue
                        
                        hero_image = None
                        if article_metadata.get('hero_image_url'):
                            hero_image = {"url": article_metadata['hero_image_url'], "width": None, "height": None, "source": "scraper"}
                        elif homepage_data.get('image_url'):
                            hero_image = {"url": homepage_data['image_url'], "width": None, "height": None, "source": "scraper"}
                        
                        article = self._create_article_dict(
                            title=homepage_data['title'], link=url,
                            description=homepage_data.get('description', ''),
                            published=published, hero_image=hero_image
                        )
                        
                        if self._validate_article(article):
                            new_articles.append(article)
                            await self.tracker.update_headline_url(self.source_id, headline, url)
                        
                        await asyncio.sleep(0.5)
                        await page.goto(self.base_url, timeout=self.timeout)
                        await page.wait_for_timeout(1000)
                    except Exception as e:
                        print(f"      ⚠️ Error: {e}")
                
                await self.tracker.store_headlines(self.source_id, current_headlines)
                print(f"\n[{self.source_id}] ✅ {len(new_articles)} articles (skipped: {skipped_old} old, {skipped_no_link} no link)")
                return new_articles
            finally:
                await page.close()
        except Exception as e:
            print(f"[{self.source_id}] Error: {e}")
            import traceback
            traceback.print_exc()
            return []

    async def close(self):
        await super().close()
        if self.tracker:
            await self.tracker.close()
            self.tracker = None


custom_scraper_registry.register(DomusScraper)
