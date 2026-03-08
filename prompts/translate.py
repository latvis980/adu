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
    Translate a single article's headline lines and summary to all target languages.

    Adds four new keys to the article dict:
    - headline_line_1_translations: {"es": "...", "fr": "...", "pt-br": "...", "ru": "..."}
    - headline_line_2_translations: {"es": "...", "fr": "...", "pt-br": "...", "ru": "..."}
    - ai_summary_translations: {"es": "...", "fr": "...", "pt-br": "...", "ru": "..."}

    Args:
        article: Article dict with 'headline_line_1', 'headline_line_2', and 'ai_summary' keys

    Returns:
        Article dict with translation fields added
    """
    line1 = article.get("headline_line_1", "")
    line2 = article.get("headline_line_2", "")
    summary = article.get("ai_summary", "")

    if not line1 and not line2 and not summary:
        article["headline_line_1_translations"] = {}
        article["headline_line_2_translations"] = {}
        article["ai_summary_translations"] = {}
        return article

    try:
        translator = _get_translator()

        line1_translations = {}
        line2_translations = {}
        summary_translations = {}

        for internal_code, deepl_code in TARGET_LANGUAGES.items():
            if line1:
                t = _translate_text(translator, line1, deepl_code)
                if t:
                    line1_translations[internal_code] = t

            if line2:
                t = _translate_text(translator, line2, deepl_code)
                if t:
                    line2_translations[internal_code] = t

            if summary:
                t = _translate_text(translator, summary, deepl_code)
                if t:
                    summary_translations[internal_code] = t

        article["headline_line_1_translations"] = line1_translations
        article["headline_line_2_translations"] = line2_translations
        article["ai_summary_translations"] = summary_translations

        lang_count = len(summary_translations)
        if lang_count > 0:
            print(f"      ✅ Translated to {lang_count} languages")
        else:
            print(f"      ⚠️  No translations produced")

    except Exception as e:
        print(f"      ❌ Translation failed: {e}")
        article["headline_line_1_translations"] = {}
        article["headline_line_2_translations"] = {}
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