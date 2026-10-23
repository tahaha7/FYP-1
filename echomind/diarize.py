import os


def diarize_file(audio_path, pipeline_name="pyannote/speaker-diarization-3.1"):
    # needs a gpu really, and the pyannote model is gated on hf
    token = os.environ.get("HF_TOKEN")
    if not token:
        raise RuntimeError("HF_TOKEN not set - pyannote model needs it")

    from pyannote.audio import Pipeline
    import torch

    pipeline = Pipeline.from_pretrained(pipeline_name, use_auth_token=token)
    if torch.cuda.is_available():
        pipeline.to(torch.device("cuda"))

    result = pipeline(str(audio_path))

    turns = []
    for turn, _, speaker in result.itertracks(yield_label=True):
        turns.append({
            "start": float(turn.start),
            "end": float(turn.end),
            "speaker": speaker,
        })
    turns.sort(key=lambda t: t["start"])
    return turns
