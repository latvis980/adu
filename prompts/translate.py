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
from langchain_core.prompts import ChatPromptTemplate, SystemMessagePromptTemplate, HumanMessagePromptTemplate


# Target languages
TARGET_LANGUAGES = {
    "es": "Spanish",
    "fr": "French",
    "pt-br": "Brazilian Portuguese",
    "ru": "Russian",
}

# System prompt for translation
TRANSLATE_SYSTEM_PROMPT = """You are a professional translator specialising in architecture, design, and urbanism.
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

You must respond with valid JSON only. No markdown, no backticks, no explanation."""

# User message template
TRANSLATE_USER_TEMPLATE = """Translate the following headline and summary into {languages_list}.

HEADLINE:
{headline}

SUMMARY:
{summary}

Respond with ONLY a JSON object in this exact format (no markdown, no backticks):
{{"headline": {{"es": "...", "fr": "...", "pt-br": "...", "ru": "..."}}, "summary": {{"es": "...", "fr": "...", "pt-br": "...", "ru": "..."}}}}"""

# Combined ChatPromptTemplate for LangChain
TRANSLATE_PROMPT_TEMPLATE = ChatPromptTemplate.from_messages([
    SystemMessagePromptTemplate.from_template(TRANSLATE_SYSTEM_PROMPT),
    HumanMessagePromptTemplate.from_template(TRANSLATE_USER_TEMPLATE),
])


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
    # Clean up common AI response issues
    text = response_text.strip()

    # Remove markdown code block wrappers if present
    if text.startswith("```json"):
        text = text[7:]
    elif text.startswith("```"):
        text = text[3:]
    if text.endswith("```"):
        text = text[:-3]
    text = text.strip()

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

    except (json.JSONDecodeError, KeyError, TypeError) as e:
        print(f"      [WARN] Failed to parse translation response: {e}")
        return {
            "headline": {},
            "summary": {},
        }


def translate_article(article: dict, llm) -> dict:
    """
    Translate a single article's headline and summary to all target languages.

    Adds two new keys to the article dict:
    - ai_summary_translations: {"es": "...", "fr": "...", "pt-br": "...", "ru": "..."}
    - headline_translations: {"es": "...", "fr": "...", "pt-br": "...", "ru": "..."}

    Args:
        article: Article dict with 'headline' and 'ai_summary' keys
        llm: LangChain LLM instance

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
        # Create and invoke the chain
        chain = TRANSLATE_PROMPT_TEMPLATE | llm
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
            print(f"      Translated to {lang_count} languages")
        else:
            print(f"      [WARN] No translations produced")

    except Exception as e:
        print(f"      [WARN] Translation failed: {e}")
        article["headline_translations"] = {}
        article["ai_summary_translations"] = {}

    return article


def translate_articles(articles: list, llm) -> list:
    """
    Translate all articles in a list.

    Args:
        articles: List of article dicts with headline and ai_summary
        llm: LangChain LLM instance

    Returns:
        Articles with translation fields added
    """
    print(f"\n[TRANSLATE] Translating {len(articles)} articles to {len(TARGET_LANGUAGES)} languages...")

    for i, article in enumerate(articles, 1):
        title = article.get("headline", article.get("title", "No title"))
        source_name = article.get("source_name", article.get("source_id", "Unknown"))
        print(f"   [{i}/{len(articles)}] [{source_name}] {title[:40]}...")

        translate_article(article, llm)

    # Stats
    translated_count = sum(
        1 for a in articles if a.get("ai_summary_translations")
    )
    print(f"\n   [STATS] Translated: {translated_count}/{len(articles)} articles")

    return articles
