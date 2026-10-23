# tests for the parts of echomind that run without a gpu or a model
# run with: python -m pytest tests/ -q

from echomind import validate_evidence
from echomind.name_resolve import suggest_names
from echomind.prepare_data import split_rows
from echomind.procurement import scan_procurement
from echomind.understand import fallback_extract


def line(start, end, speaker, text):
    return {"start": start, "end": end, "speaker": speaker, "text": text}


# ---------- evidence validation ----------

TRANSCRIPT = [
    line(0.0, 4.0, "SPEAKER_00", "Ahmed will send the report by Friday"),
    line(4.0, 9.0, "SPEAKER_01", "We decided to approve the budget"),
]


def test_evidence_verified_item_passes():
    item = {
        "task": "Send the report",
        "evidence": {"quote": "Ahmed will send the report by Friday", "timestamp": 0.0},
    }
    out = validate_evidence.validate_items([item], TRANSCRIPT)
    assert len(out["verified"]) == 1
    assert out["verified"][0] is item
    assert out["rejected"] == []


def test_evidence_made_up_quote_rejected():
    item = {
        "task": "Book the flights",
        "evidence": {"quote": "We booked the flights to Dubai", "timestamp": 1.0},
    }
    out = validate_evidence.validate_items([item], TRANSCRIPT)
    assert out["verified"] == []
    assert len(out["rejected"]) == 1
    assert out["rejected"][0]["reject_reason"] == "quote not found in transcript"


def test_evidence_timestamp_out_of_range_rejected():
    item = {
        "task": "Send the report",
        "evidence": {"quote": "Ahmed will send the report by Friday", "timestamp": 999.0},
    }
    out = validate_evidence.validate_items([item], TRANSCRIPT)
    assert out["verified"] == []
    assert len(out["rejected"]) == 1
    assert out["rejected"][0]["reject_reason"] == "timestamp outside transcript"


def test_evidence_missing_quote_rejected():
    item = {"task": "Send the report", "evidence": {"timestamp": 0.0}}
    out = validate_evidence.validate_items([item], TRANSCRIPT)
    assert out["verified"] == []
    assert out["rejected"][0]["reject_reason"] == "no evidence quote"


# ---------- meeting understanding (rule-based fallback) ----------

MEETING_LINES = [
    line(0.0, 5.0, "SPEAKER_00", "Ahmed will send the report by Friday"),
    line(5.0, 10.0, "SPEAKER_01", "We decided to approve the budget"),
    line(10.0, 15.0, "SPEAKER_00", "The Lahore office is on the third floor"),
    line(15.0, 20.0, "SPEAKER_01", "ہم نے نئے دفتر کے بارے میں بات کی"),
]


def test_fallback_finds_action_with_evidence():
    out = fallback_extract(MEETING_LINES)
    assert len(out["action_items"]) == 1
    item = out["action_items"][0]
    assert item["task"] == "Ahmed will send the report by Friday"
    assert item["evidence"]["quote"] == "Ahmed will send the report by Friday"
    assert item["evidence"]["timestamp"] == 0.0


def test_fallback_finds_decision_with_evidence():
    out = fallback_extract(MEETING_LINES)
    assert len(out["decisions"]) == 1
    item = out["decisions"][0]
    assert item["text"] == "We decided to approve the budget"
    assert item["evidence"]["quote"] == "We decided to approve the budget"
    assert item["evidence"]["timestamp"] == 5.0


def test_fallback_plain_lines_produce_nothing():
    out = fallback_extract(MEETING_LINES)
    found = [i["task"] for i in out["action_items"]] + [d["text"] for d in out["decisions"]]
    assert "The Lahore office is on the third floor" not in found
    assert "ہم نے نئے دفتر کے بارے میں بات کی" not in found


def test_fallback_items_pass_evidence_validation():
    # what the fallback extracts should survive the evidence checker,
    # this is exactly what the orchestrator does with the result
    out = fallback_extract(MEETING_LINES)
    checks = validate_evidence.validate_result(out, MEETING_LINES)
    assert len(checks["action_items"]["verified"]) == 1
    assert len(checks["decisions"]["verified"]) == 1
    assert checks["action_items"]["rejected"] == []
    assert checks["decisions"]["rejected"] == []


# ---------- procurement ----------

QUOTE_LINES = [
    line(0.0, 4.0, "SPEAKER_00", "The first vendor quoted Rs 50,000 for the laptops"),
    line(4.0, 8.0, "SPEAKER_01", "Second vendor said PKR 52,000 for the same work"),
    line(8.0, 12.0, "SPEAKER_00", "Third one is asking Rs 180,000 which sounds too much"),
]


def test_procurement_finds_all_quotes():
    out = scan_procurement(QUOTE_LINES)
    amounts = sorted(q["amount"] for q in out["quotes"])
    assert len(out["quotes"]) == 3
    assert amounts == [50000.0, 52000.0, 180000.0]


def test_procurement_flags_only_the_unusual_one():
    out = scan_procurement(QUOTE_LINES)
    assert len(out["flags"]) == 1
    flag = out["flags"][0]
    assert flag["amount"] == 180000.0
    assert "median" in flag["reason"]


def test_procurement_bare_numbers_are_not_quotes():
    lines = [line(0.0, 4.0, "SPEAKER_00", "We have 12 people in the team and 3 rooms")]
    out = scan_procurement(lines)
    assert out["quotes"] == []
    assert out["flags"] == []


# ---------- name resolution ----------

def test_name_resolve_self_introduction():
    lines = [
        line(0.0, 4.0, "SPEAKER_00", "Hello everyone, my name is Ahmed"),
        line(4.0, 8.0, "SPEAKER_01", "Thanks, let us start with the agenda"),
    ]
    out = suggest_names(lines, ["Ahmed", "Sara"])
    assert "SPEAKER_00" in out
    assert out["SPEAKER_00"]["name"] == "Ahmed"
    assert out["SPEAKER_00"]["status"] == "suggested, not confirmed"


def test_name_resolve_roman_urdu_introduction():
    lines = [
        line(0.0, 4.0, "SPEAKER_01", "Assalamualaikum, mera naam Sara hai"),
    ]
    out = suggest_names(lines, ["Ahmed", "Sara"])
    assert out["SPEAKER_01"]["name"] == "Sara"


def test_name_resolve_no_names_gives_nothing():
    lines = [
        line(0.0, 4.0, "SPEAKER_00", "Let us go through the numbers one by one"),
        line(4.0, 8.0, "SPEAKER_01", "The total for this month looks fine"),
    ]
    out = suggest_names(lines, ["Ahmed", "Sara"])
    assert out == {}


# ---------- prepare_data splits ----------

def make_rows(per_speaker=10, speakers=("S1", "S2", "S3", "S4")):
    rows = []
    for sp in speakers:
        for i in range(per_speaker):
            rows.append({
                "audio_path": f"{sp}_{i}.wav",
                "transcription": f"clip {i} from {sp}",
                "speaker_id": sp,
                "category": "interview",
                "duration": 5.0,
                "confidence": "0.98",
            })
    return rows


def test_split_rows_speaker_aware_disjoint_and_complete():
    rows = make_rows()
    train, val, test = split_rows(rows, seed=42, speaker_aware=True)

    train_sp = {r["speaker_id"] for r in train}
    val_sp = {r["speaker_id"] for r in val}
    test_sp = {r["speaker_id"] for r in test}

    # no speaker appears in two splits
    assert train_sp.isdisjoint(val_sp)
    assert train_sp.isdisjoint(test_sp)
    assert val_sp.isdisjoint(test_sp)

    # all three splits got at least one speaker, and together cover everyone
    assert train_sp and val_sp and test_sp
    assert train_sp | val_sp | test_sp == {"S1", "S2", "S3", "S4"}

    # no rows lost or duplicated
    assert len(train) + len(val) + len(test) == len(rows)
    seen = [r["audio_path"] for r in train + val + test]
    assert len(seen) == len(set(seen))


def test_split_rows_plain_split_keeps_all_rows():
    rows = make_rows()
    train, val, test = split_rows(rows, seed=42, speaker_aware=False)
    assert len(train) + len(val) + len(test) == len(rows)
    assert train and val and test
