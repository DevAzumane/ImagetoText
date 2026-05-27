TRANSCRIPTION_PROMPT = """
You are an expert handwriting transcription engine.

Rules:
- Extract text exactly as written
- Preserve meaning
- No explanation
- No commentary
- Return only the extracted text
"""


def build_summary_prompt(text: str, num_lines: int):
    return f"""
You are a summarization engine.

TASK:
Summarize the following text clearly.

RULES:
- Keep important meaning
- No explanation
- No bullet points unless necessary
- Return ONLY the summary
- Summary must be EXACTLY within {num_lines} lines

TEXT:
{text}
"""