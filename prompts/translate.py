# prompts/translate.py
"""
Translation Utilities (DeepL API)
Translates article summaries and headlines to multiple languages using DeepL.

Supported languages: ES (Spanish), FR (French), PT-BR (Brazilian Portuguese), RU (Russian)

Usage:
    from prompts.translate import translate_article, translate_articles

    article = translate_article(article)
    # article["ai_summary_translations"] = {"es": "...", "fr": "...", "pt-br": "...", "ru": "..."}
    # article["headline_translations"] = {"es": "...", "fr": "...", "pt-br": "...", "ru": "..."}
"""

import os
import deepl


# Target languages: internal code -> DeepL code
TARGET_LANGUAGES = {
    "es": "ES",
    "fr": "FR",
    "pt-br": "PT-BR",
    "ru": "RU",
}


def _get_translator() -> deepl.Translator:
    """
    Create a DeepL Translator client.

    Reads DEEPL_API_KEY from environment variables.

    Returns:
        deepl.Translator instance

    Raises:
        ValueError: If DEEPL_API_KEY is not set
    """
    api_key = os.getenv("DEEPL_API_KEY")
    if not api_key:
        raise ValueError("DEEPL_API_KEY not set in environment")
    return deepl.Translator(api_key)


def _translate_text(translator: deepl.Translator, text: str, target_lang: str) -> str:
    """
    Translate a single text string to the target language.

    Args:
        translator: DeepL Translator instance
        text: Text to translate
        target_lang: DeepL language code (e.g. "ES", "PT-BR")

    Returns:
        Translated text string, or empty string on failure
    """
    if not text or not text.strip():
        return ""

    try:
        result = translator.translate_text(text, target_lang=target_lang)
        return result.text
    except Exception as e:
        print(f"      [WARN] DeepL translation failed for {target_lang}: {e}")
        return ""


def translate_article(article: dict) -> dict:
    """
    Translate a single article's headline and summary to all target languages.

    Adds two new keys to the article dict:
    - headline_translations: {"es": "...", "fr": "...", "pt-br": "...", "ru": "..."}
    - ai_summary_translations: {"es": "...", "fr": "...", "pt-br": "...", "ru": "..."}

    Args:
        article: Article dict with 'headline' and 'ai_summary' keys

    Returns:
        Article dict with translation fields added
    """
    headline = article.get("headline", "")
    summary = article.get("ai_summary", "")

    if not headline and not summary:
        article["headline_translations"] = {}
        article["ai_summary_translations"] = {}
        return article

    try:
        translator = _get_translator()

        headline_translations = {}
        summary_translations = {}

        for internal_code, deepl_code in TARGET_LANGUAGES.items():
            # Translate headline
            if headline:
                translated_headline = _translate_text(translator, headline, deepl_code)
                if translated_headline:
                    headline_translations[internal_code] = translated_headline

            # Translate summary
            if summary:
                translated_summary = _translate_text(translator, summary, deepl_code)
                if translated_summary:
                    summary_translations[internal_code] = translated_summary

        article["headline_translations"] = headline_translations
        article["ai_summary_translations"] = summary_translations

        # Log success
        lang_count = len(summary_translations)
        if lang_count > 0:
            print(f"      ✅ Translated to {lang_count} languages")
        else:
            print(f"      ⚠️  No translations produced")

    except Exception as e:
        print(f"      ❌ Translation failed: {e}")
        article["headline_translations"] = {}
        article["ai_summary_translations"] = {}

    return article


def translate_articles(articles: list) -> list:
    """
    Translate all articles in a list.

    Args:
        articles: List of article dicts with headline and ai_summary

    Returns:
        Articles with translation fields added
    """
    print(f"\n[TRANSLATE] Translating {len(articles)} articles to {len(TARGET_LANGUAGES)} languages (DeepL)...")

    for i, article in enumerate(articles, 1):
        title = article.get("headline", article.get("title", "No title"))
        source_name = article.get("source_name", article.get("source_id", "Unknown"))
        print(f"   [{i}/{len(articles)}] [{source_name}] {title[:40]}...")

        translate_article(article)

    # Stats
    translated_count = sum(
        1 for a in articles if a.get("ai_summary_translations")
    )
    print(f"\n   [STATS] Translated: {translated_count}/{len(articles)} articles")

    return articles