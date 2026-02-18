# prompts/summarize.py
"""
Summarization Prompts
Prompts for generating article summaries and tags.
"""

from langchain_core.prompts import ChatPromptTemplate, SystemMessagePromptTemplate, HumanMessagePromptTemplate
import re

# System prompt for the AI summarizer
SUMMARIZE_SYSTEM_PROMPT = """You are an architecture news editor for a professional digest. 
Your task is to create concise, informative summaries of architecture and design articles.

Today's date is {current_date}. Use this for temporal context when describing projects.

Guidelines:
- Title format: PROJECT NAME / ARCHITECT OR BUREAU (e.g., "Cloud 11 Office Complex / Snøhetta"). If the architect or bureau is unknown, don't write anything, just the name of the project. DO NOT write Unknown in the title
- For news articles about projects, write a two-sentence summary in a professional British editorial style. Sentence 1: state project name, location, architect/studio (if explicitely mentioned), typology, scale (if mentioned) and key design features (if mentioned). Sentence 2: state key details about the project — status, significance, planning context, or professional response. 
- If the news article is not about a projects, but about another related topic, write a 2-sentence summary of the article for professional audience.
- Add appropriate tag from this list: #residential, #hospitality, #office, #culture, #education, #public, #infrastructure, #landscape, #retail, #interior, #masterplan, #reuse, #mixeduse
- If the project name is in the language that doesn't match the country language (for example, in ArchDaily Brasil a project in China is named in Portuguese), translate the name of the project to English
- Keep tone neutral and factual, avoid generic praise and subjective adjectives
- Write for a specialist professional audience. 
- Use professional architectural terminology where appropriate
- Keep the tone informative but engaging
- If the article is an opinion piece, note that it's an opinion piece, but still mention the project discussed
- If it's an interview, note that it's an interview, but still mention the project discussed
- CRITICAL: Do not use emojis anywhere in your response
- CRITICAL: Keep the title clean and professional - just the project name and architect/bureau separated by a forward slash

EXAMPLES OF SUMMARIES ABOUT PROJECTS:

1. Bradfield City / Hassel and SOM

Hassell and SOM’s masterplan for Bradfield City’s first precinct in Western Sydney sets out a sustainable, mixed-use gateway shaped by Country, community and long-term urban ambition. Designed as a 24/7 neighbourhood with homes, workplaces and public space organised around a central green spine, the scheme positions the project as the foundation for Australia’s first new city in more than a century.

#masterplan


2. Waves of Water: Future Academy / Scenic Architecture Office

Scenic Architecture Office explores the idea of the “wave” as both a cultural symbol and a scientific principle, translating its sense of motion, transmission and rhythm into architectural form for the Future Academy in Shanghai. The project reimagines static building as a dynamic spatial experience, drawing on landscapes, cellular growth and waterfront context to create architecture that feels continuous, fluid and alive.

#education

"""

# User message template
SUMMARIZE_USER_TEMPLATE = """Summarize this architecture article:

Title: {title}
Description: {description}
Source: {url}

Respond with ONLY:
1. Title in format: PROJECT NAME / ARCHITECT OR BUREAU or just PROJECT NAME if author unknown or irrelevant. DO NOT write Unknown in the title
2. On a new line, a 2-sentence summary
3. On a new line, 1 relevant tag (one word, choose from this exact list: 
#residential 
#hospitality 
#office 
#culture 
#education
#public 
#infrastructure 
#landscape 
#retail 
#interior 
#masterplan 
#reuse
#mixeduse). 
No spaces, hyphens, or special characters in the tag.

EXAMPLE FORMAT: 

Nobel Center / David Chipperfield

David Chipperfield's Nobel Center in Stockholm is designed to celebrate the legacy of the Nobel Prize through a blend of exhibition spaces and public areas. The building's striking architectural form and sustainable features aim to foster dialogue and engagement with the ideals of the Nobel laureates.

#culture"""

# Combined ChatPromptTemplate for LangChain
SUMMARIZE_PROMPT_TEMPLATE = ChatPromptTemplate.from_messages([
    SystemMessagePromptTemplate.from_template(SUMMARIZE_SYSTEM_PROMPT),
    HumanMessagePromptTemplate.from_template(SUMMARIZE_USER_TEMPLATE)
])


def parse_summary_response(response_text: str) -> dict:
    """
    Parse AI response into headline, summary and tag.

    Args:
        response_text: Raw AI response

    Returns:
        Dict with 'headline', 'summary' and 'tag' keys
    """
    lines = [line.strip() for line in response_text.strip().split('\n') if line.strip()]

    if len(lines) >= 3:
        headline = lines[0]
        summary = lines[1]
        tag = lines[2].lower().strip()
    elif len(lines) == 2:
        headline = lines[0]
        summary = lines[1]
        tag = ""
    else:
        headline = ""
        summary = lines[0] if lines else ""
        tag = ""

    # Strip "Unknown" from headlines (safety net for AI ignoring instructions)
    if headline:
        headline = re.sub(r'\s*/\s*Unknown\s*$', '', headline, flags=re.IGNORECASE)
        headline = re.sub(r'\s*/\s*Unknown\s+Architect(s)?\s*$', '', headline, flags=re.IGNORECASE)
        headline = re.sub(r'\s*/\s*Unknown\s+Bureau\s*$', '', headline, flags=re.IGNORECASE)
        headline = re.sub(r'\s*/\s*Unknown\s+Studio\s*$', '', headline, flags=re.IGNORECASE)
        headline = headline.strip()

    return {
        "headline": headline,
        "summary": summary,
        "tag": tag
    }