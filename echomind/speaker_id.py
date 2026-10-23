from pathlib import Path

import numpy as np

from echomind.storage import encrypt_bytes, decrypt_bytes

_classifier = None


def _get_classifier(model_name="speechbrain/spkrec-ecapa-voxceleb"):
    global _classifier
    if _classifier is None:
        # lazy import, speechbrain is heavy and gpu-ish
        from speechbrain.inference.speaker import EncoderClassifier
        _classifier = EncoderClassifier.from_hparams(
            source=model_name, run_opts={"device": "cuda"}
        )
    return _classifier


def get_embedding(audio_path):
    import torch
    clf = _get_classifier()
    signal = clf.load_audio(str(audio_path))
    emb = clf.encode_batch(signal)
    emb = emb.squeeze().detach().cpu().numpy()
    # normalize so cosine is just a dot product later
    norm = np.linalg.norm(emb)
    if norm > 0:
        emb = emb / norm
    return emb.astype(np.float32)


def _store_file(store_dir, name):
    safe = "".join(c if c.isalnum() or c in "-_" else "_" for c in name)
    return Path(store_dir) / f"{safe}.npy.enc"


def enroll(audio_path, name, store_dir, key_path=None):
    emb = get_embedding(audio_path)
    Path(store_dir).mkdir(parents=True, exist_ok=True)
    raw = emb.tobytes()
    out = _store_file(store_dir, name)
    if key_path:
        out.write_bytes(encrypt_bytes(raw, key_path))
    else:
        # no key given, save plain (not great, use a key for real data)
        out.with_suffix(".npy").write_bytes(raw)
        out = out.with_suffix(".npy")
    return {"name": name, "embedding_path": str(out)}


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


def cosine(a, b):
    denom = np.linalg.norm(a) * np.linalg.norm(b)
    if denom == 0:
        return 0.0
    return float(np.dot(a, b) / denom)


def identify(audio_path, store_dir, threshold=0.35, key_path=None):
    people = load_enrolled(store_dir, key_path)
    if not people:
        return None
    emb = get_embedding(audio_path)
    best_name, best_score = None, -1.0
    for name, ref in people.items():
        score = cosine(emb, ref)
        if score > best_score:
            best_name, best_score = name, score
    if best_score < threshold:
        return None
    return {"name": best_name, "score": best_score}
