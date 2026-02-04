# prompts/translate.py
"""
Translation Prompts and Utilities
Translates article summaries and headlines to multiple languages.

Supported languages: ES (Spanish), FR (French), PT-BR (Brazilian Portuguese), RU (Russian)

Usage:
    from prompts.translate import translate_article

    llm = create_llm()
    article = translate_article(article, llm)
    # article["ai_summary_translations"] = {"es": "...", "fr": "...", "pt-br": "...", "ru": "..."}
    # article["headline_translations"] = {"es": "...", "fr": "...", "pt-br": "...", "ru": "..."}
"""

import json
import re
from langchain_core.prompts import ChatPromptTemplate, SystemMessagePromptTemplate, HumanMessagePromptTemplate


# Target languages
TARGET_LANGUAGES = {
    "es": "Spanish",
    "fr": "French",
    "pt-br": "Brazilian Portuguese",
    "ru": "Russian",
}

# System prompt for translation
TRANSLATE_SYSTEM_PROMPT = """You are a professional translator specializing in architecture, design, and urbanism.
Your task is to translate article headlines and summaries accurately into multiple languages.

Guidelines:
- Preserve architectural terminology and proper nouns (architect names, studio names, place names)
- Keep the same tone: informative, concise, professional
- Do not add any extra information or commentary
- Do not use emojis anywhere in your translations
- Maintain the same sentence structure and length as the original
- For project names: keep them in their original language unless there is a widely known translation
- For architect/bureau names after the slash: never translate these, keep them exactly as in the original
- PT-BR means Brazilian Portuguese, not European Portuguese

CRITICAL: You must respond with ONLY valid JSON. No markdown, no backticks, no preamble, no explanation.
The JSON must be properly escaped - use \\n for newlines, \\" for quotes, \\\\ for backslashes."""

# User message template
TRANSLATE_USER_TEMPLATE = """Translate the following headline and summary into {languages_list}.

HEADLINE:
{headline}

SUMMARY:
{summary}

Respond with ONLY this JSON structure (no markdown, no backticks, no extra text):
{{"headline": {{"es": "...", "fr": "...", "pt-br": "...", "ru": "..."}}, "summary": {{"es": "...", "fr": "...", "pt-br": "...", "ru": "..."}}}}

Remember: All quotes and special characters must be properly escaped in the JSON."""

# Combined ChatPromptTemplate for LangChain
TRANSLATE_PROMPT_TEMPLATE = ChatPromptTemplate.from_messages([
    SystemMessagePromptTemplate.from_template(TRANSLATE_SYSTEM_PROMPT),
    HumanMessagePromptTemplate.from_template(TRANSLATE_USER_TEMPLATE),
])


def clean_json_response(text: str) -> str:
    """
    Clean up AI response to extract pure JSON.

    Handles common issues:
    - Markdown code blocks
    - Text before/after JSON
    - Multiple JSON objects (takes first)
    """
    text = text.strip()

    # Remove markdown code blocks
    text = re.sub(r'^```(?:json)?\s*', '', text)
    text = re.sub(r'\s*```$', '', text)
    text = text.strip()

    # Try to find JSON object bounds
    # Look for the first { and last }
    start = text.find('{')
    end = text.rfind('}')

    if start != -1 and end != -1 and end > start:
        text = text[start:end+1]

    return text


def parse_translation_response(response_text: str) -> dict:
    """
    Parse AI translation response into structured dict.

    Args:
        response_text: Raw AI response (should be JSON)

    Returns:
        Dict with 'headline' and 'summary' keys, each containing
        language code -> translated text mappings.
        Returns empty dicts on parse failure.
    """
    # Clean up response
    text = clean_json_response(response_text)

    try:
        parsed = json.loads(text)

        headline_translations = parsed.get("headline", {})
        summary_translations = parsed.get("summary", {})

        # Validate we got the expected languages
        expected_keys = set(TARGET_LANGUAGES.keys())

        if not isinstance(headline_translations, dict):
            headline_translations = {}
        if not isinstance(summary_translations, dict):
            summary_translations = {}

        # Filter to only expected language keys
        headline_translations = {
            k: v for k, v in headline_translations.items()
            if k in expected_keys and isinstance(v, str) and v.strip()
        }
        summary_translations = {
            k: v for k, v in summary_translations.items()
            if k in expected_keys and isinstance(v, str) and v.strip()
        }

        return {
            "headline": headline_translations,
            "summary": summary_translations,
        }

    except json.JSONDecodeError as e:
        # More detailed error logging
        print(f"      [WARN] Failed to parse translation JSON: {e}")
        print(f"      [DEBUG] First 200 chars: {text[:200]}")
        print(f"      [DEBUG] Last 200 chars: {text[-200:]}")
        return {
            "headline": {},
            "summary": {},
        }
    except (KeyError, TypeError) as e:
        print(f"      [WARN] Invalid translation structure: {e}")
        return {
            "headline": {},
            "summary": {},
        }


def translate_article(article: dict, llm, max_tokens: int = 2000) -> dict:
    """
    Translate a single article's headline and summary to all target languages.

    Adds two new keys to the article dict:
    - ai_summary_translations: {"es": "...", "fr": "...", "pt-br": "...", "ru": "..."}
    - headline_translations: {"es": "...", "fr": "...", "pt-br": "...", "ru": "..."}

    Args:
        article: Article dict with 'headline' and 'ai_summary' keys
        llm: LangChain LLM instance
        max_tokens: Maximum tokens for response (default 2000 for 4 languages)

    Returns:
        Article dict with translation fields added
    """
    headline = article.get("headline", "")
    summary = article.get("ai_summary", "")

    if not headline and not summary:
        article["headline_translations"] = {}
        article["ai_summary_translations"] = {}
        return article

    # Build languages list string
    languages_list = ", ".join(
        f"{name} ({code})" for code, name in TARGET_LANGUAGES.items()
    )

    try:
        # Create and invoke the chain with higher token limit
        chain = TRANSLATE_PROMPT_TEMPLATE | llm

        # Bind max_tokens if the LLM supports it
        try:
            llm_with_tokens = llm.bind(max_tokens=max_tokens)
            chain = TRANSLATE_PROMPT_TEMPLATE | llm_with_tokens
        except:
            # If binding fails, use the chain as-is
            pass

        response = chain.invoke({
            "headline": headline,
            "summary": summary,
            "languages_list": languages_list,
        })

        result = parse_translation_response(response.content)

        article["headline_translations"] = result.get("headline", {})
        article["ai_summary_translations"] = result.get("summary", {})

        # Log success
        lang_count = len(article["ai_summary_translations"])
        if lang_count > 0:
            print(f"      ✅ Translated to {lang_count} languages")
        else:
            print(f"      ⚠️  No translations produced")

    except Exception as e:
        print(f"      ❌ Translation failed: {e}")
        article["headline_translations"] = {}
        article["ai_summary_translations"] = {}

    return article


def translate_articles(articles: list, llm, max_tokens: int = 2000) -> list:
    """
    Translate all articles in a list.

    Args:
        articles: List of article dicts with headline and ai_summary
        llm: LangChain LLM instance
        max_tokens: Maximum tokens for each translation (default 2000)

    Returns:
        Articles with translation fields added
    """
    print(f"\n[TRANSLATE] Translating {len(articles)} articles to {len(TARGET_LANGUAGES)} languages...")

    for i, article in enumerate(articles, 1):
        title = article.get("headline", article.get("title", "No title"))
        source_name = article.get("source_name", article.get("source_id", "Unknown"))
        print(f"   [{i}/{len(articles)}] [{source_name}] {title[:40]}...")

        translate_article(article, llm, max_tokens=max_tokens)

    # Stats
    translated_count = sum(
        1 for a in articles if a.get("ai_summary_translations")
    )
    print(f"\n   [STATS] Translated: {translated_count}/{len(articles)} articles")

    return articles