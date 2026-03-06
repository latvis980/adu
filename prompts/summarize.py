# prompts/summarize.py
"""
Summarization Prompts
Prompts for generating article summaries and tags.
"""

import re
from langchain_core.prompts import ChatPromptTemplate, SystemMessagePromptTemplate, HumanMessagePromptTemplate

# System prompt for the AI summarizer
SUMMARIZE_SYSTEM_PROMPT = """You are an architecture news editor for a professional digest. 
Your task is to create concise, informative summaries of architecture and design articles.

Today's date is {current_date}. Use this for temporal context when describing projects.

Guidelines:
- Header line 1: PROJECT NAME / ARCHITECT OR BUREAU (e.g., "Cloud 11 Office Complex / Snøhetta"). If the architect or bureau is unknown, just write the project name. DO NOT write Unknown in the title
- Header line 2: TYPOLOGY / CITY, COUNTRY (e.g., "Commercial / Tokyo, Japan"). If any part is unavailable or there are multiple values, simply skip that part. Never write "Unknown" or "Various". If the entire line would be empty, skip it entirely.
- Use || to separate header line 1 from header line 2 (e.g., "Cloud 11 Office Complex / Snøhetta || Commercial / Tokyo, Japan")
- Write description: exactly 2 sentences in British English. First sentence: What is the project (who designed what, where). Second sentence: What makes it notable or interesting
- If the project name is in the language that doesn't match the country language (for example, in ArchDaily Brasil a project in China is named in Portuguese), translate the name of the project to English
- Be specific and factual, avoid generic praise
- Use professional architectural terminology where appropriate
- Keep the tone informative but engaging
- If the article is an opinion piece, note that it's an opinion piece, but still mention the project discussed
- If it's an interview, note that it's an interview, but still mention the project discussed
- CRITICAL: Do not use emojis anywhere in your response
- CRITICAL: Keep the header clean and professional"""

# User message template
SUMMARIZE_USER_TEMPLATE = """Summarize this architecture article:

Title: {title}
Description: {description}
Source: {url}

Respond with ONLY these lines:
1. Header: PROJECT NAME / ARCHITECT || TYPOLOGY / CITY, COUNTRY
   - Use || to separate the two parts
   - Skip any unavailable parts in the second half (typology, city, country). If all are unavailable, omit everything after ||
   - Never write "Unknown" or "Various" for any field
2. On a new line, a 2-sentence summary
3. On a new line, 1 typology tag (one word: residential, commercial, culture, education, hospitality, healthcare, infrastructure, urbanism, landscape, museum, library, airport, sports, religious, industrial, mixeduse, memorial, pavilion, installation, masterplan, renovation, adaptive, bridge, tower, housing). No spaces, hyphens, or special characters.
4. On a new line, 1 country tag (the country name in English, lowercase, no spaces — use common short forms like "uk" not "unitedkingdom"). If the country is unclear, skip this line entirely.

Example format:
Cloud 11 Office Complex / Snøhetta || Commercial / Tokyo, Japan
Snøhetta has completed an office complex in Tokyo featuring a diagrid structural system. The 32-story building uses cross-laminated timber for its facade, making it one of the tallest timber-hybrid structures in Asia.
commercial
japan

Another example (when typology is clear but location unknown):
Vertical Forest Concept / Stefano Boeri || Residential
Stefano Boeri Architetti has unveiled a new residential tower concept incorporating over 900 trees across its facades. The design aims to absorb 30 tonnes of CO2 annually while creating a new urban biodiversity model.
residential

Another example (when only project name is known):
Solar Decathlon 2026 Pavilion
A student team has designed a net-zero energy pavilion for the Solar Decathlon competition. The structure uses phase-change materials and a novel ventilation system to maintain comfort without mechanical cooling.
pavilion"""

# Combined ChatPromptTemplate for LangChain
SUMMARIZE_PROMPT_TEMPLATE = ChatPromptTemplate.from_messages([
    SystemMessagePromptTemplate.from_template(SUMMARIZE_SYSTEM_PROMPT),
    HumanMessagePromptTemplate.from_template(SUMMARIZE_USER_TEMPLATE)
])


def parse_summary_response(response_text: str) -> dict:
    """
    Parse AI response into headline (two-liner), summary, and tags (list).

    The AI returns:
        Line 1: Header with || separator (e.g., "Project / Studio || Typology / City, Country")
        Line 2: 2-sentence summary
        Line 3: typology tag
        Line 4: country tag (optional)

    Returns:
        Dict with:
        - 'headline': two-line string (lines joined by newline), or single line if no second part
        - 'summary': the 2-sentence summary
        - 'tags': list of [typology_tag, country_tag], filtering out empty values
    """
    lines = [line.strip() for line in response_text.strip().split('\n') if line.strip()]

    headline = ""
    summary = ""
    tags = []

    if len(lines) >= 2:
        # Line 1: Header (may contain || separator)
        raw_header = lines[0]

        # Split by || to get the two header parts
        if "||" in raw_header:
            parts = [p.strip() for p in raw_header.split("||", 1)]
            header_line1 = parts[0]
            header_line2 = parts[1] if len(parts) > 1 and parts[1] else ""
        else:
            header_line1 = raw_header
            header_line2 = ""

        # Build headline: two lines joined by newline, or single line
        if header_line2:
            headline = f"{header_line1}\n{header_line2}"
        else:
            headline = header_line1

        # Line 2: Summary
        summary = lines[1]

        # Line 3: Typology tag (optional)
        if len(lines) >= 3:
            typology_tag = lines[2].lower().strip()
            if typology_tag and typology_tag not in ("unknown", "various", "none", "n/a"):
                tags.append(typology_tag)

        # Line 4: Country tag (optional)
        if len(lines) >= 4:
            country_tag = lines[3].lower().strip()
            if country_tag and country_tag not in ("unknown", "various", "none", "n/a"):
                tags.append(country_tag)

    elif len(lines) == 1:
        headline = ""
        summary = lines[0]
    else:
        headline = ""
        summary = ""

    # Safety net: strip "Unknown" variants from headline (regex post-processing)
    headline = re.sub(r'\s*/\s*Unknown\b[^|]*', '', headline, flags=re.IGNORECASE)
    headline = re.sub(r'\bUnknown\s*/\s*', '', headline, flags=re.IGNORECASE)
    headline = re.sub(r'\bVarious\b', '', headline, flags=re.IGNORECASE)
    # Clean up any leftover empty separators or double spaces
    headline = re.sub(r'\s*\|\|\s*$', '', headline)  # trailing ||
    headline = re.sub(r'^\s*\|\|\s*', '', headline)  # leading ||
    headline = re.sub(r'\n\s*$', '', headline)  # trailing empty line
    headline = re.sub(r'  +', ' ', headline).strip()

    return {
        "headline": headline,
        "summary": summary,
        "tags": tags
    }
