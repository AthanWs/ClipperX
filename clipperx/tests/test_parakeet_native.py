import pytest

from clipperx import transcription


def test_default_transcription_uses_parakeet_without_silent_whisper_fallback(monkeypatch, tmp_path):
    expected = [transcription.Segment(0, 1, "Hola", [{"start":0,"end":1,"word":"Hola"}])]
    monkeypatch.setattr(transcription, "_transcribe_parakeet", lambda *a, **k: expected, raising=False)
    monkeypatch.setattr(transcription, "_transcribe_whisperx", lambda *a, **k: pytest.fail("Unexpected WhisperX"))
    monkeypatch.setattr(transcription, "_transcribe_native", lambda *a, **k: pytest.fail("Unexpected Whisper fallback"))
    assert transcription.transcribe(tmp_path/"voice.wav") == expected


def test_native_tokens_are_joined_to_words_without_punctuation_extending_voice():
    from clipperx.parakeet_worker import timestamped_words
    result = timestamped_words("¿Hola, parcero?", [
        {"token":"¿", "start":0., "end":0.},
        {"token":"Ho", "start":.08, "end":.24},
        {"token":"la", "start":.24, "end":.4},
        {"token":",", "start":.4, "end":.4},
        {"token":" par", "start":1.2, "end":1.36},
        {"token":"cero", "start":1.36, "end":1.6},
        {"token":"?", "start":3., "end":3.},
    ], offset=30., duration=34.)
    assert [(w["word"],w["start"],w["end"]) for w in result] == [
        ("¿Hola,",30.08,30.4),("parcero?",31.2,31.6)]
    assert all(w["probability"] is None for w in result)


def test_native_token_text_mismatch_is_rejected_instead_of_inventing_times():
    from clipperx.parakeet_worker import timestamped_words
    with pytest.raises(ValueError, match="token|text"):
        timestamped_words("otra palabra",[{"token":"inventada","start":0.,"end":.8}],offset=0,duration=1.)


def test_chunk_ownership_keeps_source_clock_and_does_not_repeat_overlap():
    from clipperx.parakeet_worker import owned_words
    words=[{"word":"antes","start":29.6,"end":29.9},
        {"word":"después","start":30.1,"end":30.4}]
    assert [w["word"] for w in owned_words(words,core_start=0,core_end=30)] == ["antes"]
    assert [w["word"] for w in owned_words(words,core_start=30,core_end=60)] == ["después"]


def test_worker_rejects_invalid_native_timestamps():
    from clipperx.parakeet_worker import timestamped_words
    with pytest.raises(ValueError,match="time"):
        timestamped_words("hola",[{"token":"hola","start":float("nan"),"end":.4}],offset=0,duration=1)


def test_retranscription_preserves_previous_json_and_records_native_provenance(monkeypatch,tmp_path):
    from clipperx import parakeet
    import json
    import hashlib
    audio=tmp_path/"a.wav"
    audio.write_bytes(b"audio")
    old=tmp_path/"transcript.json"
    old.write_text("historical-original",encoding="utf-8")
    payload=dict(backend="parakeet",native_timestamps=True,request_signature="signature",
        segments=[dict(start=0,end=1,text="Hola",words=[dict(word="Hola",start=0,end=1,probability=None)])])
    monkeypatch.setattr(parakeet,"run_parakeet",lambda *a,**k:payload)
    transcription.transcribe_to_json(audio,output_dir=tmp_path,output_name="transcript")
    retained=list((tmp_path/"transcript-history").glob("*/transcript.json"))
    assert len(retained)==1 and retained[0].read_text()=="historical-original"
    metadata=json.loads(old.with_suffix(".provenance.json").read_text())
    assert metadata["native_timestamps"] and metadata["request_signature"]=="signature"
    assert metadata["transcript_sha256"]==hashlib.sha256(old.read_bytes()).hexdigest()


def test_old_or_tampered_transcript_is_not_a_parakeet_cache_hit(monkeypatch,tmp_path):
    from clipperx import parakeet
    import json
    audio=tmp_path/"a.flac"
    audio.write_bytes(b"audio")
    transcript=tmp_path/"transcript.json"
    transcript.write_text("[]")
    monkeypatch.setattr(parakeet,"request_signature",lambda _:"current")
    assert not parakeet.transcript_is_current(transcript,audio,language="es")
    metadata=dict(request_signature="current",audio_sha256=parakeet.audio_hash(audio),
        transcript_sha256=parakeet.audio_hash(transcript),backend="parakeet",native_timestamps=True)
    transcript.with_suffix(".provenance.json").write_text(json.dumps(metadata))
    assert parakeet.transcript_is_current(transcript,audio,language="es")
    transcript.write_text("[{}]")
    assert not parakeet.transcript_is_current(transcript,audio,language="es")


@pytest.mark.parametrize("current", [False, True])
def test_gui_pipeline_checks_native_provenance_before_reusing_transcript(monkeypatch,tmp_path,current):
    from types import SimpleNamespace
    from clipperx import pipeline, parakeet
    from clipperx.jobs import JobStore
    video=tmp_path/"source.mp4"
    video.write_bytes(b"source")
    job=JobStore.for_video(video,jobs_root=tmp_path/"jobs")
    job.audio_path.write_bytes(b"audio")
    job.transcript_path.write_text("[]")
    job.mark_complete("audio",artifacts=[job.audio_path])
    job.mark_complete("transcription",artifacts=[job.transcript_path])
    monkeypatch.setattr(pipeline.JobStore,"for_video",lambda _:job)
    monkeypatch.setattr(pipeline.settings,"llm_provider","test_no_provider_invoked")
    monkeypatch.setattr(pipeline.settings,"asr_backend","parakeet")
    monkeypatch.setattr(pipeline,"probe_media",lambda _:dict(duration_seconds=1))
    monkeypatch.setattr(pipeline,"TranscriptionEstimator",lambda **kwargs:
        SimpleNamespace(start=lambda:None,stop=lambda:None))
    checked=[]
    monkeypatch.setattr(parakeet,"transcript_is_current",lambda *a,**k:checked.append(k) or current)
    transcribed=[]
    monkeypatch.setattr(pipeline.transcription,"transcribe_to_json",lambda *a,**k:
        transcribed.append(k) or job.transcript_path)
    class StopBeforeInference(Exception):
        pass
    def stop(_):
        raise StopBeforeInference()
    monkeypatch.setattr(pipeline,"load_segments",stop)
    with pytest.raises(StopBeforeInference):
        pipeline.full_pipeline(video)
    assert len(checked)==1
    assert len(transcribed)==(0 if current else 1)
