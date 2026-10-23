import re


def normalize(text):
    text = (text or "").lower()
    text = re.sub(r"\s+", " ", text)
    return text.strip()


def transcript_text(lines):
    return " ".join(l.get("text", "") for l in lines)


def transcript_range(lines):
    if not lines:
        return 0.0, 0.0
    start = min(l.get("start", 0) for l in lines)
    end = max(l.get("end", 0) for l in lines)
    return start, end


def check_item(item, full_text_norm, t_start, t_end):
    ev = item.get("evidence") or {}
    quote = normalize(ev.get("quote", ""))
    if not quote:
        return False, "no evidence quote"
    if quote not in full_text_norm:
        return False, "quote not found in transcript"
    ts = ev.get("timestamp")
    if ts is not None:
        try:
            ts = float(ts)
        except (TypeError, ValueError):
            return False, "bad timestamp"
        if ts < t_start or ts > t_end:
            return False, "timestamp outside transcript"
    return True, "ok"


def validate_items(items, lines):
    full = normalize(transcript_text(lines))
    t_start, t_end = transcript_range(lines)
    verified, rejected = [], []
    for item in items:
        ok, reason = check_item(item, full, t_start, t_end)
        if ok:
            verified.append(item)
        else:
            bad = dict(item)
            bad["reject_reason"] = reason
            rejected.append(bad)
    return {"verified": verified, "rejected": rejected}


def validate_result(result, lines):
    # result is the dict from understand.py
    out = {}
    for key in ("action_items", "decisions"):
        out[key] = validate_items(result.get(key, []), lines)
    return out
