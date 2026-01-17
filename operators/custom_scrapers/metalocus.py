# operators/custom_scrapers/metalocus.py
"""
Metalocus Custom Scraper - Visual AI Approach
Scrapes architecture news from Metalocus (Spanish architecture magazine)

Site: https://www.metalocus.es/en
Challenge: Use User-Agent as precaution

Visual Scraping Strategy:
1. Take screenshot of English homepage
2. Use GPT-4o vision to extract article headlines
3. On first run: Store all headlines in database as "seen"
4. On subsequent runs: Only process NEW headlines (not in database)
5. Find headline text in HTML coupled with link using AI
6. Click link to get publication date and metadata

Usage:
    scraper = MetalocusScraper()
    articles = await scraper.fetch_articles()
    await scraper.close()
"""

import asyncio
import base64
from typing import Optional, List
from datetime import datetime, timezone

from langchain_openai import ChatOpenAI
from langchain_core.messages import HumanMessage

from operators.custom_scraper_base import BaseCustomScraper, custom_scraper_registry
from storage.article_tracker import ArticleTracker
from prompts.homepage_analyzer import HOMEPAGE_ANALYZER_PROMPT_TEMPLATE, parse_headlines


class MetalocusScraper(BaseCustomScraper):
    """Visual AI-powered custom scraper for Metalocus"""

    source_id = "metalocus"
    source_name = "Metalocus"
    base_url = "https://www.metalocus.es/en"
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
            api_key = os.getenv("OPENAI_API_KEY")
            if not api_key:
                raise ValueError("OPENAI_API_KEY not set")
            self.vision_model = ChatOpenAI(model="gpt-4o-mini", api_key=api_key, temperature=0.1)

    async def _analyze_homepage_screenshot(self, screenshot_path: str) -> List[str]:
        self._ensure_vision_model()
        with open(screenshot_path, 'rb') as f:
            image_data = base64.b64encode(f.read()).decode('utf-8')
        message = HumanMessage(content=[
            {"type": "text", "text": HOMEPAGE_ANALYZER_PROMPT_TEMPLATE.format()},
            {"type": "image_url", "image_url": {"url": f"data:image/png;base64,{image_data}"}}
        ])
        response = await asyncio.to_thread(self.vision_model.invoke, [message])
        return parse_headlines(str(response.content if hasattr(response, 'content') else response))

    async def _find_headline_in_html_with_ai(self, page, headline: str) -> Optional[dict]:
        self._ensure_vision_model()
        html_context = await page.evaluate("""
            () => {
                const containers = document.querySelectorAll('article, .post, [class*="item"], [class*="card"]');
                const data = [];
                containers.forEach((c, i) => {
                    const links = c.querySelectorAll('a[href]');
                    let main = null, text = '';
                    links.forEach(l => {
                        const t = l.textContent.trim();
                        if (t.length > text.length) { main = l; text = t; }
                    });
                    if (main && text.length > 5) {
                        data.push({index: i, link_text: text, href: main.href,
                            description: (c.querySelector('p')?.textContent || '').substring(0, 150),
                            image_url: c.querySelector('img')?.src || null});
                    }
                });
                return data;
            }
        """)
        if not html_context: return None
        
        context = f"Looking for: '{headline}'\n\n" + "\n".join([f"[{i['index']}] {i['link_text']}\n    {i['href']}" for i in html_context[:15]])
        prompt = f"{context}\n\nWhich index matches '{headline}'? Number or 'NONE':"
        resp = await asyncio.to_thread(self.vision_model.invoke, [HumanMessage(content=prompt)])
        resp_text = str(resp.content if hasattr(resp, 'content') else resp).strip().upper()
        
        if resp_text == "NONE": return None
        import re
        match = re.search(r'\d+', resp_text)
        if match:
            idx = int(match.group(0))
            for item in html_context:
                if item['index'] == idx:
                    return {'title': item['link_text'], 'link': item['href'],
                            'description': item['description'], 'image_url': item['image_url']}
        return None

    async def fetch_articles(self, hours: int = 24) -> list[dict]:
        await self._ensure_tracker()
        try:
            page = await self._create_page()
            try:
                await page.goto(self.base_url, timeout=self.timeout, wait_until="networkidle")
                await page.wait_for_timeout(2000)
                
                screenshot_path = f"/tmp/{self.source_id}_homepage.png"
                await page.screenshot(path=screenshot_path, full_page=False)
                
                headlines = await self._analyze_homepage_screenshot(screenshot_path)
                if not headlines: return []
                
                seen = await self.tracker.get_stored_headlines(self.source_id)
                new = [h for h in headlines if h not in seen][:10]
                if not new: return []
                
                articles = []
                for headline in new:
                    try:
                        data = await self._find_headline_in_html_with_ai(page, headline)
                        if not data or not data.get('link'): continue
                        
                        await page.goto(data['link'], timeout=self.timeout)
                        await page.wait_for_timeout(1000)
                        
                        meta = await page.evaluate("""
                            () => ({
                                date_text: document.querySelector('time[datetime], .date')?.getAttribute('datetime') || '',
                                hero_image_url: document.querySelector('meta[property="og:image"]')?.content || null
                            })
                        """)
                        
                        published = self._parse_date(meta['date_text'])
                        if published:
                            days = (datetime.now(timezone.utc) - datetime.fromisoformat(published.replace('Z', '+00:00'))).days
                            if days > self.MAX_ARTICLE_AGE_DAYS: continue
                        
                        article = self._create_article_dict(
                            title=data['title'], link=data['link'],
                            description=data.get('description', ''), published=published,
                            hero_image={"url": meta['hero_image_url'], "width": None, "height": None, "source": "scraper"} if meta.get('hero_image_url') else None
                        )
                        
                        if self._validate_article(article):
                            articles.append(article)
                            await self.tracker.update_headline_url(self.source_id, headline, data['link'])
                        
                        await asyncio.sleep(0.5)
                        await page.goto(self.base_url, timeout=self.timeout)
                        await page.wait_for_timeout(1000)
                    except Exception as e:
                        print(f"[{self.source_id}] Error: {e}")
                
                await self.tracker.store_headlines(self.source_id, headlines)
                return articles
            finally:
                await page.close()
        except Exception as e:
            print(f"[{self.source_id}] Error: {e}")
            return []

    async def close(self):
        await super().close()
        if self.tracker:
            await self.tracker.close()
            self.tracker = None


custom_scraper_registry.register(MetalocusScraper)
