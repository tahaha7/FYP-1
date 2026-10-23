import re

# this only suggests names, a human should confirm them


def _clean(name):
    return name.strip().lower()


def _self_intro(line_text, attendees):
    # "my name is X" / "i am X" / "mera naam X hai"
    low = line_text.lower()
    for name in attendees:
        n = _clean(name)
        patterns = [
            f"my name is {n}",
            f"i am {n}",
            f"i'm {n}",
            f"mera naam {n} hai",
            f"mera naam {n}",
        ]
        if any(p in low for p in patterns):
            return name
    return None


def _addressed_names(line_text, attendees):
    # names other people use while talking to someone
    low = " " + line_text.lower().strip() + " "
    found = []
    for name in attendees:
        n = _clean(name)
        patterns = [
            rf"\bhello {re.escape(n)}\b",
            rf"\bhi {re.escape(n)}\b",
            rf"\b{re.escape(n)} said\b",
            rf"\bthanks {re.escape(n)}\b",
            rf"\bthank you {re.escape(n)}\b",
            rf"^{re.escape(n)}[,:]",          # name at start, like "Ahmed, what do you think"
            rf"\b{re.escape(n)},",            # "so Ahmed, tell us..."
        ]
        if any(re.search(p, low.strip()) for p in patterns):
            found.append(name)
    return found


def suggest_names(lines, attendees):
    # returns {speaker: {name, reason, count}} - all suggestions, not confirmed
    suggestions = {}

    def add(speaker, name, reason):
        cur = suggestions.get(speaker)
        if cur and cur["name"] == name:
            cur["count"] += 1
        elif cur is None:
            suggestions[speaker] = {
                "name": name,
                "reason": reason,
                "count": 1,
                "status": "suggested, not confirmed",
            }
        # if a different name was already suggested, keep the first one

    for line in lines:
        speaker = line.get("speaker", "")
        text = line.get("text", "")

        name = _self_intro(text, attendees)
        if name:
            add(speaker, name, "self introduction in their own line")
            continue

        # people usually address the person they are talking to, not themselves
        for name in _addressed_names(text, attendees):
            others = {l.get("speaker", "") for l in lines} - {speaker}
            for other in others:
                add(other, name,
                    f"addressed by name in a line by {speaker}")

    return suggestions
