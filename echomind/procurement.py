import re
import statistics

# finds money amounts like Rs 50,000 / PKR 2.5 lakh / $900 / 5000 rupees
AMOUNT_RE = re.compile(
    r"(?P<prefix>rs\.?|pkr|usd|\$)?\s*"
    r"(?P<num>\d[\d,]*\.?\d*)"
    r"\s*(?P<unit>lakh|lac|crore|k|thousand|million)?"
    r"(?P<suffix>\s*(rupees|rs\.?|pkr))?",
    re.IGNORECASE,
)

UNIT_MULT = {
    "lakh": 100000,
    "lac": 100000,
    "crore": 10000000,
    "k": 1000,
    "thousand": 1000,
    "million": 1000000,
}


def _to_number(match):
    num = float(match.group("num").replace(",", ""))
    unit = (match.group("unit") or "").lower()
    if unit in UNIT_MULT:
        num *= UNIT_MULT[unit]
    return num


def _looks_like_money(match, text):
    # bare numbers with no rs/rupees/unit around them are probably not prices
    if match.group("prefix") or match.group("suffix") or match.group("unit"):
        return True
    return False


def scan_quotes(lines):
    quotes = []
    for line in lines:
        text = line.get("text", "")
        for m in AMOUNT_RE.finditer(text):
            if not _looks_like_money(m, text):
                continue
            quotes.append({
                "timestamp": line.get("start", 0.0),
                "speaker": line.get("speaker", ""),
                "amount_text": m.group(0).strip(),
                "amount": _to_number(m),
                "line": text,
            })
    return quotes


def scan_procurement(lines):
    # flags amounts that are far from the middle value of all quotes
    quotes = scan_quotes(lines)
    flags = []
    if len(quotes) >= 3:
        median = statistics.median(q["amount"] for q in quotes)
        if median > 0:
            for q in quotes:
                if q["amount"] > 1.5 * median:
                    flags.append({
                        **q,
                        "reason": f"amount {q['amount']:,.0f} is more than 1.5x the median quote {median:,.0f}",
                    })
                elif q["amount"] < 0.5 * median:
                    flags.append({
                        **q,
                        "reason": f"amount {q['amount']:,.0f} is less than half the median quote {median:,.0f}",
                    })
    return {"quotes": quotes, "flags": flags}
