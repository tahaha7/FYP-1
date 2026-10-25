# ECAPA-TDNN speaker enrolment, identification and EER evaluation
#
# enrol:     python -m echomind.speaker_id enroll --name rayyan data/enroll/rayyan/*.wav
# identify:  python -m echomind.speaker_id identify some_clip.wav
# list:      python -m echomind.speaker_id list
# delete:    python -m echomind.speaker_id delete --name rayyan
# evaluate:  python -m echomind.speaker_id eval data/verify
#            (data/verify/<name>/*.wav, clips NOT used for enrolment)

import argparse
import json
from pathlib import Path

import numpy as np
import yaml

from echomind.storage import encrypt_bytes, decrypt_bytes

SAMPLE_RATE = 16000  # speechbrain load_audio resamples to this
MIN_CLIP_SECONDS = 10.0  # shorter clips give unreliable voiceprints
# a clip this far from the person's other clips is probably bad audio
# or the wrong person
CONSISTENCY_WARN = 0.5

_classifier = None


def pick_device(device=None):
    if device:
        return device
    try:
        import torch
        return "cuda" if torch.cuda.is_available() else "cpu"
    except ImportError:
        return "cpu"


def _get_classifier(model_name="speechbrain/spkrec-ecapa-voxceleb", device=None):
    global _classifier
    if _classifier is None:
        # lazy import, speechbrain is heavy. ecapa is small enough for cpu
        from speechbrain.inference.speaker import EncoderClassifier
        _classifier = EncoderClassifier.from_hparams(
            source=model_name, run_opts={"device": pick_device(device)}
        )
    return _classifier


def _normalize(emb):
    norm = np.linalg.norm(emb)
    if norm > 0:
        emb = emb / norm
    return emb.astype(np.float32)


def get_embedding(audio_path, min_seconds=0.0):
    clf = _get_classifier()
    signal = clf.load_audio(str(audio_path))
    seconds = signal.shape[-1] / SAMPLE_RATE
    if seconds < min_seconds:
        raise ValueError(
            f"{audio_path} is {seconds:.1f}s, need at least {min_seconds:.0f}s "
            "of speech for a reliable voiceprint")
    emb = clf.encode_batch(signal)
    emb = emb.squeeze().detach().cpu().numpy()
    # normalize so cosine is just a dot product later
    return _normalize(emb)


def _store_file(store_dir, name):
    safe = "".join(c if c.isalnum() or c in "-_" else "_" for c in name)
    return Path(store_dir) / f"{safe}.npy.enc"


def enroll(audio_paths, name, store_dir, key_path=None,
           min_seconds=MIN_CLIP_SECONDS):
    # one clip or several - several (3-5) give a much steadier voiceprint
    if isinstance(audio_paths, (str, Path)):
        audio_paths = [audio_paths]
    if not audio_paths:
        raise ValueError("no enrolment clips given")

    embs = [get_embedding(p, min_seconds=min_seconds) for p in audio_paths]
    emb = _normalize(np.mean(embs, axis=0))
    # leave-one-out: each clip against the mean of the OTHER clips, so a bad
    # clip can't hide by pulling the average towards itself
    clip_scores = {}
    if len(embs) > 1:
        total = np.sum(embs, axis=0)
        for p, e in zip(audio_paths, embs):
            clip_scores[str(p)] = cosine(e, total - e)
    warnings = [p for p, s in clip_scores.items() if s < CONSISTENCY_WARN]

    Path(store_dir).mkdir(parents=True, exist_ok=True)
    raw = emb.tobytes()
    out = _store_file(store_dir, name)
    if key_path:
        out.write_bytes(encrypt_bytes(raw, key_path))
    else:
        # no key given, save plain (not great, use a key for real data)
        out.with_suffix(".npy").write_bytes(raw)
        out = out.with_suffix(".npy")
    return {"name": name, "embedding_path": str(out),
            "clips": len(embs), "clip_scores": clip_scores,
            "inconsistent_clips": warnings}


def load_enrolled(store_dir, key_path=None):
    people = {}
    for f in sorted(Path(store_dir).glob("*.npy*")):
        name = f.name.split(".")[0]
        raw = f.read_bytes()
        if f.suffix == ".enc" or str(f).endswith(".npy.enc"):
            if not key_path:
                continue  # cant read encrypted ones without the key
            raw = decrypt_bytes(raw, key_path)
        people[name] = np.frombuffer(raw, dtype=np.float32)
    return people


def delete(name, store_dir):
    # removes a person's voiceprint (consent withdrawn, re-enrolment, ...)
    enc = _store_file(store_dir, name)
    removed = []
    for f in (enc, enc.with_suffix(".npy")):
        if f.exists():
            f.unlink()
            removed.append(str(f))
    return removed


def cosine(a, b):
    denom = np.linalg.norm(a) * np.linalg.norm(b)
    if denom == 0:
        return 0.0
    return float(np.dot(a, b) / denom)


def score_all(emb, people):
    return {name: cosine(emb, ref) for name, ref in people.items()}


def identify(audio_path, store_dir, threshold=0.35, key_path=None):
    people = load_enrolled(store_dir, key_path)
    if not people:
        return None
    scores = score_all(get_embedding(audio_path), people)
    best_name = max(scores, key=scores.get)
    best_score = scores[best_name]
    if best_score < threshold:
        return None
    return {"name": best_name, "score": best_score}


# ---------- evaluation: FAR / FRR / EER ----------

def far_frr(genuine, impostor, threshold):
    # accept when score >= threshold
    genuine = np.asarray(genuine, dtype=float)
    impostor = np.asarray(impostor, dtype=float)
    far = float(np.mean(impostor >= threshold)) if impostor.size else 0.0
    frr = float(np.mean(genuine < threshold)) if genuine.size else 0.0
    return far, frr


def compute_eer(genuine, impostor):
    # sweep every observed score as a threshold, take the point where
    # FAR and FRR are closest. returns (eer, threshold)
    if len(genuine) == 0 or len(impostor) == 0:
        raise ValueError("need both genuine and impostor scores for EER")
    candidates = np.unique(np.concatenate([genuine, impostor]))
    best = None
    for t in candidates:
        far, frr = far_frr(genuine, impostor, t)
        gap = abs(far - frr)
        if best is None or gap < best[0]:
            best = (gap, (far + frr) / 2, float(t))
    return best[1], best[2]


def trial_scores(verify, people):
    # verify: {true_name: [embedding, ...]}. every clip is scored against
    # every enrolled voiceprint: same name = genuine, other name = impostor.
    # clips from people who are not enrolled only make impostor trials
    genuine, impostor = [], []
    for true_name, embs in verify.items():
        for emb in embs:
            for name, score in score_all(emb, people).items():
                (genuine if name == true_name else impostor).append(score)
    return genuine, impostor


def evaluate(verify_dir, store_dir, threshold, key_path=None):
    people = load_enrolled(store_dir, key_path)
    if not people:
        raise ValueError(f"no voiceprints found in {store_dir}")
    verify = {}
    for person_dir in sorted(p for p in Path(verify_dir).iterdir() if p.is_dir()):
        clips = sorted(person_dir.glob("*.wav"))
        if clips:
            verify[person_dir.name] = [get_embedding(c) for c in clips]
    genuine, impostor = trial_scores(verify, people)
    eer, eer_threshold = compute_eer(genuine, impostor)
    far, frr = far_frr(genuine, impostor, threshold)
    return {"genuine_trials": len(genuine), "impostor_trials": len(impostor),
            "eer": eer, "eer_threshold": eer_threshold,
            "current_threshold": threshold, "far": far, "frr": frr}


def main():
    ap = argparse.ArgumentParser(description="EchoMind speaker enrolment")
    ap.add_argument("--config", default="config.yaml")
    ap.add_argument("--device", default=None, help="cpu or cuda (default: auto)")
    sub = ap.add_subparsers(dest="cmd", required=True)

    p = sub.add_parser("enroll", help="make a voiceprint from 3-5 clips")
    p.add_argument("--name", required=True)
    p.add_argument("clips", nargs="+")
    p.add_argument("--min-seconds", type=float, default=MIN_CLIP_SECONDS)

    p = sub.add_parser("identify", help="who is speaking in this clip")
    p.add_argument("clip")

    sub.add_parser("list", help="list enrolled people")

    p = sub.add_parser("delete", help="remove a voiceprint")
    p.add_argument("--name", required=True)

    p = sub.add_parser("eval", help="FAR / FRR / EER on held-out clips")
    p.add_argument("verify_dir", help="folder with one sub-folder of .wav per person")

    args = ap.parse_args()
    with open(args.config, encoding="utf-8") as f:
        cfg = yaml.safe_load(f)
    sid = cfg.get("speaker_id", {})
    store_dir = sid.get("store_dir", "outputs/voiceprints")
    threshold = sid.get("threshold", 0.35)
    key_path = cfg.get("storage", {}).get("key_path")

    if args.cmd in ("enroll", "identify", "eval"):
        _get_classifier(sid.get("model", "speechbrain/spkrec-ecapa-voxceleb"),
                        args.device)

    if args.cmd == "enroll":
        out = enroll(args.clips, args.name, store_dir, key_path,
                     min_seconds=args.min_seconds)
        print(json.dumps(out, indent=2))
        if out["inconsistent_clips"]:
            print("warning: these clips do not match the others, "
                  "check the audio and re-record them:",
                  ", ".join(out["inconsistent_clips"]))
    elif args.cmd == "identify":
        out = identify(args.clip, store_dir, threshold, key_path)
        print(json.dumps(out or {"name": None, "note": "unknown speaker"}, indent=2))
    elif args.cmd == "list":
        for name in load_enrolled(store_dir, key_path):
            print(name)
    elif args.cmd == "delete":
        removed = delete(args.name, store_dir)
        print("removed: " + ", ".join(removed) if removed else "nothing to remove")
    elif args.cmd == "eval":
        out = evaluate(args.verify_dir, store_dir, threshold, key_path)
        print(json.dumps(out, indent=2))
        print(f"tip: set speaker_id.threshold to {out['eer_threshold']:.3f} "
              "in config.yaml to sit at the EER point")


if __name__ == "__main__":
    main()
