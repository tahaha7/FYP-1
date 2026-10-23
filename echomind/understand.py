import json
import re

# words that sound like someone took a task
ACTION_MARKERS = [
    "will ", "should ", "needs to ", "need to ", "has to ", "have to ",
    "please ", "action item", "todo", "to do",
    # roman urdu ones
    "karna hai", "kar de", "kar dein", "bhej de", "bhej dein", "dekh le",
]

# words that sound like a decision was made
DECISION_MARKERS = [
    "decided", "we agree", "agreed", "final decision", "approved",
    "confirmed", "settled on",
]


def build_understanding_prompt(transcript_text):
    # give this prompt to any llm, it must answer with json only
    prompt = f"""You are reading a meeting transcript. Speakers switch between Urdu and English.

Read the transcript and reply with ONLY a JSON object, no extra text, in this shape:
{{
  "summary": "short summary of the meeting, 3-5 sentences",
  "decisions": [
    {{"text": "the decision", "evidence": {{"quote": "exact line from the transcript", "timestamp": 0.0}}}}
  ],
  "action_items": [
    {{"task": "what needs to be done", "owner": "person name or null", "deadline": "deadline or null",
      "evidence": {{"quote": "exact line from the transcript", "timestamp": 0.0}}}}
  ],
  "sentiment": "positive / neutral / negative / mixed"
}}

Rules:
- Every decision and every action item MUST have an evidence quote copied word for word from the transcript, and the timestamp (in seconds) where that line starts.
- Do not invent anything. If there is no evidence in the transcript, do not include the item.
- Owners and deadlines only if the transcript actually says them, otherwise null.

Transcript:
{transcript_text}
"""
    return prompt


def _split_sentences(text):
    parts = re.split(r"(?<=[.!?])\s+", text or "")
    return [p.strip() for p in parts if p.strip()]


def fallback_extract(lines):
    # no llm available, so just pick lines that look like tasks or decisions
    actions = []
    decisions = []
    for line in lines:
        for sent in _split_sentences(line.get("text", "")):
            low = sent.lower()
            ev = {"quote": sent, "timestamp": line.get("start", 0.0)}
            if any(m in low for m in ACTION_MARKERS):
                actions.append({
                    "task": sent,
                    "owner": None,
                    "deadline": None,
                    "evidence": ev,
                })
            elif any(m in low for m in DECISION_MARKERS):
                decisions.append({
                    "text": sent,
                    "evidence": ev,
                })

    texts = [l.get("text", "") for l in lines if l.get("text")]
    if texts:
        summary = " ".join(texts)[:400]
        if len(" ".join(texts)) > 400:
            summary += "..."
    else:
        summary = "No transcript text found."

    return {
        "summary": summary,
        "decisions": decisions,
        "action_items": actions,
        "sentiment": "unknown",
        "source": "rule-based fallback (no llm used)",
    }


def _parse_llm_reply(raw):
    # llms like adding text around the json, so find the { ... } part
    start = raw.find("{")
    end = raw.rfind("}")
    if start == -1 or end == -1:
        raise ValueError("no json found in llm reply")
    return json.loads(raw[start:end + 1])


def understand(lines, llm_call=None):
    if llm_call is None:
        return fallback_extract(lines)

    transcript_text = "\n".join(
        f"[{l.get('start', 0):.1f}s] {l.get('speaker', '')}: {l.get('text', '')}"
        for l in lines
    )
    try:
        result = _parse_llm_reply(llm_call(build_understanding_prompt(transcript_text)))
    except Exception:
        # llm failed or gave bad json, fall back instead of crashing
        return fallback_extract(lines)

    result.setdefault("summary", "")
    result.setdefault("decisions", [])
    result.setdefault("action_items", [])
    result.setdefault("sentiment", "unknown")
    result["source"] = "llm"
    return result
