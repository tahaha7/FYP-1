import argparse
import json
import time
from pathlib import Path

import yaml

from echomind.name_resolve import suggest_names
from echomind.procurement import scan_procurement
from echomind.storage import save_json_encrypted
from echomind.transcribe import transcribe_file
from echomind.understand import understand
from echomind.validate_evidence import validate_result


def load_config(path="config.yaml"):
    with open(path) as f:
        return yaml.safe_load(f)


def run_with_retry(fn, tries=2):
    # simple retry, most failures here are one-off loading issues
    last_err = None
    for _ in range(tries):
        try:
            return fn(), None
        except Exception as e:
            last_err = e
            time.sleep(1)
    return None, last_err


def overlap(a_start, a_end, b_start, b_end):
    return max(0.0, min(a_end, b_end) - max(a_start, b_start))


def align_segments(segments, turns):
    # give each asr segment the speaker who overlaps it the most
    lines = []
    for seg in segments:
        speaker = "SPEAKER_00"
        best = 0.0
        for turn in turns:
            ov = overlap(seg["start"], seg["end"], turn["start"], turn["end"])
            if ov > best:
                best = ov
                speaker = turn["speaker"]
        lines.append({
            "start": seg["start"],
            "end": seg["end"],
            "speaker": speaker,
            "text": seg["text"],
        })
    lines.sort(key=lambda l: l["start"])
    return lines


def fmt_time(sec):
    m = int(sec // 60)
    s = int(sec % 60)
    return f"{m:02d}:{s:02d}"


class MeetingPipeline:
    def __init__(self, config_path="config.yaml"):
        self.cfg = load_config(config_path)
        self.status = {}

    def _step(self, name, fn):
        result, err = run_with_retry(fn)
        if err is not None:
            self.status[name] = f"failed: {err}"
            return None
        self.status[name] = "ok"
        return result

    def run(self, audio_path, out_dir, model_size="base", device="cpu",
            compute_type="int8", llm_call=None, attendees=None,
            enable_procurement=False):
        out_dir = Path(out_dir)
        out_dir.mkdir(parents=True, exist_ok=True)
        stem = Path(audio_path).stem

        # 1. transcribe
        asr = self._step("transcribe", lambda: transcribe_file(
            audio_path, model_size, device, compute_type))
        if asr is None:
            raise RuntimeError(f"transcription failed: {self.status['transcribe']}")

        # 2. diarize - optional, pipeline must still work without it
        def _diarize():
            from echomind.diarize import diarize_file
            return diarize_file(audio_path,
                                self.cfg["diarization"]["pipeline"])

        turns, err = run_with_retry(_diarize)
        if err is not None:
            turns = []
            self.status["diarize"] = (
                f"skipped ({err}) - using SPEAKER_00 for everyone")
        else:
            self.status["diarize"] = "ok"

        # 3+4. align and build lines
        lines = align_segments(asr["segments"], turns)
        self.status["align"] = "ok"

        # speaker name suggestions - only if we know who attended,
        # and these stay suggestions, they never rename anyone
        name_suggestions = {}
        if attendees:
            name_suggestions = suggest_names(lines, attendees)
            self.status["name_resolve"] = "ok"

        # optional procurement scan for prices/quotes
        procurement_result = None
        if enable_procurement:
            procurement_result = scan_procurement(lines)
            self.status["procurement"] = "ok"

        # 5. understand + check evidence
        result = self._step("understand", lambda: understand(lines, llm_call))
        if result is None:
            result = {"summary": "", "decisions": [], "action_items": [],
                      "sentiment": "unknown"}
        checks = validate_result(result, lines)
        self.status["validate"] = "ok"

        meeting = {
            "audio_path": str(audio_path),
            "language": asr.get("language"),
            "duration": asr.get("duration"),
            "lines": lines,
            "summary": result.get("summary", ""),
            "sentiment": result.get("sentiment", "unknown"),
            "understanding_source": result.get("source", ""),
            "decisions": checks["decisions"]["verified"],
            "action_items": checks["action_items"]["verified"],
            "rejected": {
                "decisions": checks["decisions"]["rejected"],
                "action_items": checks["action_items"]["rejected"],
            },
            "speaker_name_suggestions": name_suggestions,
            "procurement": procurement_result,
            "status": self.status,
        }

        # 6. write outputs
        json_path = out_dir / f"{stem}.json"
        with open(json_path, "w", encoding="utf-8") as f:
            json.dump(meeting, f, ensure_ascii=False, indent=2)

        md_path = out_dir / f"{stem}.md"
        md_path.write_text(self._minutes_md(meeting), encoding="utf-8")

        # encrypted copy, if we have somewhere to keep the key
        enc_path = None
        key_path = self.cfg.get("storage", {}).get("key_path")
        if key_path:
            enc_path = str(out_dir / f"{stem}.json.enc")
            save_json_encrypted(meeting, enc_path, key_path)

        return {"meeting": meeting, "json": str(json_path),
                "markdown": str(md_path), "encrypted": enc_path}

    def _minutes_md(self, meeting):
        out = []
        out.append(f"# Meeting minutes — {Path(meeting['audio_path']).name}\n")
        out.append(f"Duration: {meeting.get('duration')}s | "
                   f"Language: {meeting.get('language')} | "
                   f"Sentiment: {meeting.get('sentiment')}\n")

        out.append("## Summary\n")
        out.append((meeting.get("summary") or "No summary.") + "\n")

        out.append("## Decisions\n")
        if meeting["decisions"]:
            for d in meeting["decisions"]:
                out.append(f"- {d.get('text', '')}")
        else:
            out.append("- None found")
        out.append("")

        out.append("## Action items\n")
        if meeting["action_items"]:
            for a in meeting["action_items"]:
                owner = a.get("owner") or "unassigned"
                deadline = a.get("deadline") or "no deadline"
                out.append(f"- {a.get('task', '')} (owner: {owner}, {deadline})")
        else:
            out.append("- None found")
        out.append("")

        rejected = (meeting["rejected"]["decisions"] +
                    meeting["rejected"]["action_items"])
        if rejected:
            out.append("## Rejected by evidence check\n")
            for r in rejected:
                txt = r.get("text") or r.get("task", "")
                out.append(f"- {txt} — reason: {r.get('reject_reason', '')}")
            out.append("")

        suggestions = meeting.get("speaker_name_suggestions") or {}
        if suggestions:
            out.append("## Speaker name suggestions (not confirmed)\n")
            for speaker, info in suggestions.items():
                out.append(f"- {speaker}: maybe {info.get('name', '')} "
                           f"({info.get('reason', '')})")
            out.append("")

        proc = meeting.get("procurement")
        if proc and proc.get("flags"):
            out.append("## Procurement flags\n")
            for flag in proc["flags"]:
                out.append(f"- {flag.get('amount_text', '')} at "
                           f"{fmt_time(flag.get('timestamp', 0))} — "
                           f"{flag.get('reason', '')}")
            out.append("")

        out.append("## Transcript\n")
        for l in meeting["lines"]:
            out.append(f"[{fmt_time(l['start'])}] **{l['speaker']}**: {l['text']}")
        out.append("")
        return "\n".join(out)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("audio")
    ap.add_argument("--out-dir", default="outputs/meetings")
    ap.add_argument("--config", default="config.yaml")
    ap.add_argument("--model-size", default="base")
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--compute-type", default="int8")
    ap.add_argument("--attendees", default=None,
                    help="comma separated names, for speaker name suggestions")
    ap.add_argument("--procurement", action="store_true",
                    help="also scan the meeting for unusual prices/quotes")
    args = ap.parse_args()

    attendees = None
    if args.attendees:
        attendees = [a.strip() for a in args.attendees.split(",") if a.strip()]

    pipe = MeetingPipeline(args.config)
    out = pipe.run(args.audio, args.out_dir, args.model_size, args.device,
                   args.compute_type, attendees=attendees,
                   enable_procurement=args.procurement)
    print("step status:", json.dumps(out["meeting"]["status"], indent=2))
    print(f"minutes: {out['markdown']}")
    print(f"json:    {out['json']}")
    print(f"encrypted copy: {out['encrypted']}")


if __name__ == "__main__":
    main()
