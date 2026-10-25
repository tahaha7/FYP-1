# speaker id tests with a fake encoder, so no speechbrain or gpu needed

import numpy as np
import pytest

from echomind import speaker_id


class FakeSignal:
    def __init__(self, n):
        self.shape = (n,)


class FakeEmb:
    def __init__(self, vec):
        self.vec = np.asarray(vec, dtype=np.float32)

    def squeeze(self):
        return self

    def detach(self):
        return self

    def cpu(self):
        return self

    def numpy(self):
        return self.vec


class FakeClassifier:
    # clip name "<person>_<seconds>s_<n>.wav" -> a fixed vector per person,
    # with a little per-clip noise
    VOICES = {"ali": [1, 0, 0, 0], "sara": [0, 1, 0, 0], "omar": [0, 0, 1, 0]}

    def load_audio(self, path):
        seconds = float(path.split("/")[-1].split("_")[1].rstrip("s"))
        sig = FakeSignal(int(seconds * speaker_id.SAMPLE_RATE))
        sig.path = path
        return sig

    def encode_batch(self, signal):
        name = signal.path.split("/")[-1].split("_")[0]
        n = int(signal.path.split("_")[-1].split(".")[0])
        rng = np.random.default_rng(n)
        return FakeEmb(np.array(self.VOICES[name]) + rng.normal(0, 0.1, 4))


@pytest.fixture(autouse=True)
def fake_model(monkeypatch):
    monkeypatch.setattr(speaker_id, "_classifier", FakeClassifier())


def test_short_clip_rejected():
    with pytest.raises(ValueError, match="need at least"):
        speaker_id.enroll(["x/ali_4s_1.wav"], "ali", "unused")


def test_enroll_averages_clips_and_encrypts(tmp_path):
    key = tmp_path / "k.key"
    clips = [f"x/ali_15s_{i}.wav" for i in range(3)]
    out = speaker_id.enroll(clips, "ali", tmp_path / "vp", key_path=key)
    assert out["clips"] == 3
    assert out["embedding_path"].endswith("ali.npy.enc")
    assert out["inconsistent_clips"] == []
    people = speaker_id.load_enrolled(tmp_path / "vp", key_path=key)
    assert list(people) == ["ali"]
    assert np.isclose(np.linalg.norm(people["ali"]), 1.0, atol=1e-5)
    # no key = encrypted voiceprints are not readable
    assert speaker_id.load_enrolled(tmp_path / "vp") == {}


def test_single_path_still_works(tmp_path):
    out = speaker_id.enroll("x/sara_12s_1.wav", "sara", tmp_path / "vp",
                            key_path=tmp_path / "k.key")
    assert out["clips"] == 1


def test_inconsistent_clip_flagged(tmp_path):
    clips = ["x/ali_15s_1.wav", "x/ali_15s_2.wav", "x/omar_15s_3.wav"]
    out = speaker_id.enroll(clips, "ali", tmp_path / "vp",
                            key_path=tmp_path / "k.key")
    assert out["inconsistent_clips"] == ["x/omar_15s_3.wav"]


def test_identify_and_unknown_and_delete(tmp_path):
    key, vp = tmp_path / "k.key", tmp_path / "vp"
    speaker_id.enroll([f"x/ali_15s_{i}.wav" for i in range(3)], "ali", vp, key)
    speaker_id.enroll([f"x/sara_15s_{i}.wav" for i in range(3)], "sara", vp, key)
    hit = speaker_id.identify("x/sara_5s_9.wav", vp, 0.5, key)
    assert hit["name"] == "sara"
    # omar was never enrolled, so he must come back unknown
    assert speaker_id.identify("x/omar_5s_9.wav", vp, 0.5, key) is None
    assert len(speaker_id.delete("sara", vp)) == 1
    assert list(speaker_id.load_enrolled(vp, key)) == ["ali"]


def test_far_frr_and_eer():
    genuine = [0.9, 0.8, 0.7, 0.3]
    impostor = [0.1, 0.2, 0.4, 0.6]
    far, frr = speaker_id.far_frr(genuine, impostor, 0.5)
    assert far == 0.25 and frr == 0.25
    eer, t = speaker_id.compute_eer(genuine, impostor)
    assert eer == 0.25
    far, frr = speaker_id.far_frr(genuine, impostor, t)
    assert far == frr == 0.25


def test_perfect_separation_gives_zero_eer():
    eer, t = speaker_id.compute_eer([0.8, 0.9], [0.1, 0.2])
    assert eer == 0.0
    assert 0.2 < t <= 0.8


def test_evaluate_end_to_end(tmp_path):
    key, vp, verify = tmp_path / "k.key", tmp_path / "vp", tmp_path / "verify"
    speaker_id.enroll([f"x/ali_15s_{i}.wav" for i in range(3)], "ali", vp, key)
    speaker_id.enroll([f"x/sara_15s_{i}.wav" for i in range(3)], "sara", vp, key)
    for name in ("ali", "sara", "omar"):
        (verify / name).mkdir(parents=True)
        for i in range(2):
            (verify / name / f"{name}_8s_{10 + i}.wav").write_bytes(b"")
    out = speaker_id.evaluate(verify, vp, 0.5, key)
    # 2 clips x 2 enrolled people genuine; everything else impostor
    assert out["genuine_trials"] == 4
    assert out["impostor_trials"] == 8
    assert out["eer"] == 0.0
