"""Build a provider-agnostic technical timeline for long-form streams."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any

import cv2
import numpy as np

from .audio_features import AudioFeatureExtractor
from .face import detect_faces, load_face_model
from .jobs import JobStore
from .media import probe_media
from .progress import reporter
from .stages import ensure_audio

TIMELINE_VERSION = 2


def build_timeline(
    video_path: Path,
    *,
    sample_interval: float = 30.0,
    include_audio: bool = True,
    force: bool = False,
) -> Path:
    if sample_interval < 2.0:
        raise ValueError("sample_interval debe ser al menos 2 segundos")

    job = JobStore.for_video(video_path)
    if not force and _timeline_cache_matches(
        job.timeline_path,
        sample_interval=sample_interval,
        include_audio=include_audio,
    ):
        return job.timeline_path

    job.mark_started("timeline")
    try:
        metadata = probe_media(video_path)
        duration = float(metadata.get("duration_seconds") or 0.0)
        if duration <= 0:
            raise RuntimeError("ffprobe no devolvió una duración válida")

        audio_path: Path | None = None
        if include_audio:
            _, audio_path = ensure_audio(video_path, force=force)

        sample_times = _sample_times(0.0, duration, sample_interval)
        samples = _sample_video(
            video_path,
            duration=duration,
            sample_times=sample_times,
            window_end=duration,
            frames_dir=job.frames_dir,
            audio_path=audio_path,
            frame_prefix="sample",
        )
        timeline = {
            "version": TIMELINE_VERSION,
            "kind": "coarse_timeline",
            "source": str(video_path.resolve()),
            "fingerprint": job.fingerprint,
            "media": metadata,
            "sampling": {
                "interval_seconds": sample_interval,
                "include_audio": include_audio,
                "count": len(samples),
            },
            "visual_summary": _summarize_visual_layout(samples),
            "samples": samples,
        }
        temporary = job.timeline_path.with_suffix(".json.tmp")
        temporary.write_text(json.dumps(timeline, ensure_ascii=False, indent=2), encoding="utf-8")
        temporary.replace(job.timeline_path)
        job.mark_complete("timeline", artifacts=[job.timeline_path, job.frames_dir])
        return job.timeline_path
    except Exception as exc:
        job.mark_failed("timeline", str(exc))
        raise


def inspect_window(
    video_path: Path,
    *,
    start: float,
    end: float,
    sample_interval: float = 1.0,
    include_audio: bool = True,
    force: bool = False,
) -> Path:
    """Persist fine-grained evidence for a candidate, gap, or boundary window."""

    if sample_interval < 0.25:
        raise ValueError("sample_interval debe ser al menos 0.25 segundos")
    metadata = probe_media(video_path)
    duration = float(metadata.get("duration_seconds") or 0.0)
    start = max(0.0, float(start))
    end = min(duration, float(end))
    if end <= start:
        raise ValueError("La ventana debe tener una duración positiva")
    if end - start > 300.0:
        raise ValueError("Una inspección fina no puede superar 300 segundos")

    job = JobStore.for_video(video_path)
    signature = f"{start:.3f}|{end:.3f}|{sample_interval:.3f}|{include_audio}"
    probe_id = hashlib.sha256(signature.encode("utf-8")).hexdigest()[:12]
    output_path = job.probes_dir / f"window_{int(start * 1000):010d}_{int(end * 1000):010d}_{probe_id}.json"
    if not force and output_path.exists():
        return output_path

    audio_path: Path | None = None
    if include_audio:
        _, audio_path = ensure_audio(video_path)
    frame_dir = job.probes_dir / probe_id
    frame_dir.mkdir(parents=True, exist_ok=True)
    sample_times = _sample_times(start, end, sample_interval)
    samples = _sample_video(
        video_path,
        duration=duration,
        sample_times=sample_times,
        window_end=end,
        frames_dir=frame_dir,
        audio_path=audio_path,
        frame_prefix="probe",
    )
    payload = {
        "version": TIMELINE_VERSION,
        "kind": "window_probe",
        "source": str(video_path.resolve()),
        "fingerprint": job.fingerprint,
        "window": {"start": start, "end": end},
        "sampling": {
            "interval_seconds": sample_interval,
            "include_audio": include_audio,
            "count": len(samples),
        },
        "visual_summary": _summarize_visual_layout(samples),
        "samples": samples,
    }
    temporary = output_path.with_suffix(".json.tmp")
    temporary.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    temporary.replace(output_path)
    return output_path


def _sample_video(
    video_path: Path,
    *,
    duration: float,
    sample_times: list[float],
    window_end: float,
    frames_dir: Path,
    audio_path: Path | None,
    frame_prefix: str,
) -> list[dict[str, Any]]:
    capture = cv2.VideoCapture(str(video_path))
    if not capture.isOpened():
        raise RuntimeError(f"No se pudo abrir el video: {video_path}")

    detector = load_face_model()
    audio = AudioFeatureExtractor(audio_path) if audio_path else None
    previous_histogram: np.ndarray | None = None
    previous_gray: np.ndarray | None = None
    samples: list[dict[str, Any]] = []
    try:
        for index, timestamp in enumerate(sample_times):
            capture.set(cv2.CAP_PROP_POS_MSEC, timestamp * 1000.0)
            ok, frame = capture.read()
            if not ok or frame is None:
                continue

            analysis_frame = _analysis_frame(frame)
            gray = cv2.cvtColor(analysis_frame, cv2.COLOR_BGR2GRAY)
            histogram = cv2.calcHist([analysis_frame], [0, 1], None, [32, 32], [0, 256, 0, 256])
            cv2.normalize(histogram, histogram)
            visual_change = _visual_change(previous_histogram, histogram, previous_gray, gray)
            previous_histogram = histogram
            previous_gray = gray

            faces = [
                _face_evidence(box.x, box.y, box.width, box.height, analysis_frame)
                for box in detect_faces(analysis_frame, classifier=detector)
            ]
            frame_path = frames_dir / f"{frame_prefix}_{index:05d}_{int(timestamp * 1000):010d}ms.jpg"
            cv2.imwrite(str(frame_path), analysis_frame, [cv2.IMWRITE_JPEG_QUALITY, 82])

            if index + 1 < len(sample_times):
                end = min(duration, sample_times[index + 1])
            else:
                end = min(duration, window_end)
            audio_features = audio.window(timestamp, end).to_dict() if audio else None
            samples.append(
                {
                    "index": index,
                    "timestamp": round(timestamp, 3),
                    "end": round(end, 3),
                    "frame": str(frame_path),
                    "frame_width": int(analysis_frame.shape[1]),
                    "frame_height": int(analysis_frame.shape[0]),
                    "brightness": round(float(gray.mean()) / 255.0, 6),
                    "sharpness": round(float(cv2.Laplacian(gray, cv2.CV_64F).var()), 4),
                    "visual_change": round(visual_change, 6),
                    "faces": faces,
                    "audio": audio_features,
                }
            )
            reporter.set_percent(
                "timeline",
                100.0 * (index + 1) / len(sample_times),
                f"Muestra {index + 1}/{len(sample_times)}",
            )
    finally:
        capture.release()
        close = getattr(detector, "close", None)
        if callable(close):
            close()
    return samples


def _sample_times(start: float, end: float, interval: float) -> list[float]:
    times = list(np.arange(start, end, interval, dtype=float))
    final = max(start, end - min(0.25, interval / 2.0))
    if not times or final - times[-1] > max(0.1, interval * 0.1):
        times.append(final)
    return [round(float(value), 6) for value in times]


def _face_evidence(x: int, y: int, width: int, height: int, frame: np.ndarray) -> dict[str, Any]:
    frame_height, frame_width = frame.shape[:2]
    center_x = (x + width / 2.0) / max(frame_width, 1)
    center_y = (y + height / 2.0) / max(frame_height, 1)
    relative_area = (width * height) / max(frame_width * frame_height, 1)
    return {
        "x": x,
        "y": y,
        "width": width,
        "height": height,
        "normalized": {
            "x": round(x / max(frame_width, 1), 6),
            "y": round(y / max(frame_height, 1), 6),
            "width": round(width / max(frame_width, 1), 6),
            "height": round(height / max(frame_height, 1), 6),
        },
        "center": {"x": round(center_x, 6), "y": round(center_y, 6)},
        "relative_area": round(relative_area, 6),
        "region": _screen_region(center_x, center_y),
        "scale": "large" if relative_area >= 0.08 else "small",
    }


def _screen_region(center_x: float, center_y: float) -> str:
    horizontal = "left" if center_x < 0.4 else "right" if center_x > 0.6 else "center"
    vertical = "top" if center_y < 0.4 else "bottom" if center_y > 0.6 else "middle"
    return f"{vertical}-{horizontal}"


def _summarize_visual_layout(samples: list[dict[str, Any]]) -> dict[str, Any]:
    region_counts: dict[str, int] = {}
    small_region_counts: dict[str, int] = {}
    small_region_boxes: dict[str, list[dict[str, float]]] = {}
    transitions: list[dict[str, Any]] = []
    previous_signature: tuple[str, ...] | None = None
    samples_with_faces = 0
    for sample in samples:
        faces = sample.get("faces") or []
        if faces:
            samples_with_faces += 1
        regions = sorted({str(face["region"]) for face in faces})
        for face in faces:
            region = str(face["region"])
            region_counts[region] = region_counts.get(region, 0) + 1
            if face["scale"] == "small":
                small_region_counts[region] = small_region_counts.get(region, 0) + 1
                small_region_boxes.setdefault(region, []).append(face["normalized"])
        signature = tuple(regions)
        if previous_signature is not None and signature != previous_signature:
            transitions.append(
                {
                    "timestamp": sample["timestamp"],
                    "from": list(previous_signature),
                    "to": list(signature),
                }
            )
        previous_signature = signature

    minimum_persistence = max(2, int(round(len(samples) * 0.15)))
    camera_regions = [
        {
            "region": region,
            "detections": count,
            "normalized_face_box": {
                key: round(float(np.median([box[key] for box in small_region_boxes[region]])), 6)
                for key in ("x", "y", "width", "height")
            },
        }
        for region, count in sorted(small_region_counts.items(), key=lambda item: item[1], reverse=True)
        if count >= minimum_persistence
    ]
    return {
        "samples_with_faces": samples_with_faces,
        "face_region_counts": region_counts,
        "persistent_small_face_regions": camera_regions,
        "face_layout_transitions": transitions,
    }


def _timeline_cache_matches(
    timeline_path: Path,
    *,
    sample_interval: float,
    include_audio: bool,
) -> bool:
    if not timeline_path.exists():
        return False
    try:
        payload = json.loads(timeline_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return False
    sampling = payload.get("sampling", {})
    return (
        payload.get("version") == TIMELINE_VERSION
        and abs(float(sampling.get("interval_seconds", -1.0)) - sample_interval) < 1e-6
        and bool(sampling.get("include_audio")) is include_audio
    )


def _analysis_frame(frame: np.ndarray) -> np.ndarray:
    height, width = frame.shape[:2]
    scale = min(1.0, 640.0 / max(width, 1))
    if scale == 1.0:
        return frame
    return cv2.resize(frame, (int(width * scale), int(height * scale)), interpolation=cv2.INTER_AREA)


def _visual_change(
    previous_histogram: np.ndarray | None,
    histogram: np.ndarray,
    previous_gray: np.ndarray | None,
    gray: np.ndarray,
) -> float:
    if previous_histogram is None or previous_gray is None:
        return 1.0
    histogram_distance = max(
        0.0,
        min(1.0, 1.0 - float(cv2.compareHist(previous_histogram, histogram, cv2.HISTCMP_CORREL))),
    )
    if previous_gray.shape != gray.shape:
        pixel_distance = 1.0
    else:
        pixel_distance = float(cv2.absdiff(previous_gray, gray).mean()) / 255.0
    return 0.65 * histogram_distance + 0.35 * pixel_distance
