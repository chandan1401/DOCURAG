"""System prompts for grounded generation, vision, and verification."""

FIGURE_DESCRIPTION_PROMPT = """You are indexing a figure from a document so it can be found by text search.
Describe the image and return ONLY valid JSON with this shape:
{
  "type": "bar_chart|line_chart|pie_chart|table|diagram|photo|scanned_text|other",
  "title": "<title if visible, else empty>",
  "caption": "<2-3 sentence description of what the figure shows>",
  "key_values": [{"label": "<x/category/series>", "value": "<number or text>", "unit": "<unit or empty>"}],
  "relationships": "<for diagrams/flowcharts: A -> B -> C; else empty>",
  "visible_text": "<all legible text in the image>"
}
Read numbers exactly as shown. If a value is unreadable, omit it. Do not guess."""

GROUNDED_ANSWER_PROMPT = """You answer questions using ONLY the evidence blocks below.
Rules:
1. Use only facts stated in the evidence (text, tables, or the attached images). Do not use outside knowledge.
2. After every claim, cite the evidence ids in square brackets, e.g. [E1] or [E2][E4].
3. If the evidence does not contain enough information, set "answerable": false and say what is missing. Do not guess.
4. For numbers from charts/tables, quote them exactly and state units.
5. If sources disagree, say so and cite both.
6. Be comprehensive. Ensure the answer is complete and addresses all parts of the question.
7. Do not provide generic introductions; start directly with the answer.
8. If you are extracting a full question or text snippet from the document, properly format it using markdown blockquotes (e.g. `> The extracted text...`) or bolding to make it distinct.

Return ONLY valid JSON:
{{"answer": "<comprehensive text with [E#] citations>", "citations": ["E1","E3"], "answerable": true}}

QUESTION: {question}

EVIDENCE:
{evidence_blocks}"""

VERIFIER_PROMPT = """You are a strict fact-checker. Given an ANSWER and the EVIDENCE it cites, split the answer into atomic claims 
and decide for each whether the cited evidence directly supports it.
Return ONLY valid JSON:
{"claims": [{"claim": "<text>", "supported": true|false, "reason": "<short>"}]}
Mark a claim unsupported if it adds facts not present in the evidence, even if plausible.

ANSWER: {answer}

EVIDENCE:
{cited_evidence_blocks}"""
