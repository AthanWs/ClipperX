"""Existing loudnorm engine through one file context and two owned HOST controls.

No source receipt, supervisor, subprocess fallback, or publication authority lives
here. The later HOST supplies private scratch and retires the output consumer.
"""

from dataclasses import dataclass
from fractions import Fraction
import hashlib
import math
import struct
import subprocess
from typing import Literal

from . import audio_mix as audio
from ._normalization_files import FileEvidence, NormalizationFiles
from ._source_host_ownership import HostedObservationControl, SourceHostJob
from .render_runtime import RenderRuntime

_PASS_NS = 45_000_000_000
_MAX_SAMPLES = 8_592_000
_MAX_PCM = 68_736_000
_LOG_CAP = 65536
_CHUNK = 65536


@dataclass(frozen=True, slots=True)
class NormalizationPassEvidence:
    pass_kind: Literal["measure", "process"]
    control_id: int
    generation: int
    canonical_pipe_argv: tuple[str, ...]
    actual_argv: tuple[str, ...]
    input_file: FileEvidence
    stdout_bytes: int
    stdout_sha256: str
    log_bytes: int
    natural_eof: bool
    exit_code: int


@dataclass(frozen=True, slots=True)
class PreparedNormalization:
    schedule_identity: str
    binding: audio.SourceClockBinding
    runtime_identity: tuple[str, ...]
    audio_format: audio.AudioFormat
    receipt: audio.NormalizationReceipt
    passes: tuple[NormalizationPassEvidence, NormalizationPassEvidence]
    normalized: audio.NormalizedPCM
    normalized_file: FileEvidence


def _guard(schedule: audio.AudioSchedule, pcm: bytes, binding: audio.SourceClockBinding | None,
           fmt: audio.AudioFormat, runtime: RenderRuntime) -> None:
    if type(schedule) is not audio.AudioSchedule or not audio._prepared_fields(schedule, audio.AudioSchedule):
        raise ValueError("exact audio schedule required")
    if (type(schedule.spans) is not tuple or not 1 <= len(schedule.spans) <= 4096
            or type(schedule.cues) is not tuple or len(schedule.cues) > 4096
            or any(type(s) is not audio.AudioSpan or not audio._prepared_fields(s, audio.AudioSpan)
                   for s in schedule.spans)
            or any(type(c) is not audio.ScheduledCue or not audio._prepared_fields(c, audio.ScheduledCue)
                   for c in schedule.cues)):
        raise ValueError("bounded exact schedule records required")
    audio._prepared_format(fmt)
    audio._prepared_runtime(runtime)
    if binding is not None:
        audio._prepared_binding(binding)
    audio._guard_known_input((schedule, binding, fmt, runtime),
        record_types=(audio.AudioSchedule, audio.AudioSpan, audio.ScheduledCue,
                      audio.SourceClockBinding, audio.AudioFormat, RenderRuntime),
        text_limit=2_000_000, entry_limit=32768)
    if type(schedule.source_fingerprint) is not str or not schedule.source_fingerprint:
        raise ValueError("source fingerprint required")
    previous = 0
    for span in schedule.spans:
        if (any(type(v) is not Fraction for v in
                (span.source_start, span.source_end, span.start, span.end))
                or any(type(v) is not int for v in (span.first, span.last))
                or any(type(v) is not bool for v in (span.normalize, span.limiter))
                or type(span.gain_db) not in (float, int)
                or not 0 <= span.source_start < span.source_end
                or not 0 <= span.start < span.end <= 179
                or span.first != previous or not span.first < span.last <= _MAX_SAMPLES
                or audio.quantize_time(span.start) != span.first
                or audio.quantize_time(span.end) != span.last):
            raise ValueError("invalid bounded schedule span")
        previous = span.last
    for cue in schedule.cues:
        if (type(cue.asset_id) is not str or not cue.asset_id
                or type(cue.start) is not Fraction or type(cue.end) is not Fraction
                or type(cue.first) is not int or type(cue.last) is not int
                or type(cue.gain_db) not in (int, float) or type(cue.duck_db) not in (int, float)
                or not 0 <= cue.start < cue.end <= 179
                or not 0 <= cue.first < cue.last <= previous
                or audio.quantize_time(cue.start) != cue.first or audio.quantize_time(cue.end) != cue.last):
            raise ValueError("invalid bounded schedule cue")
    if type(pcm) is not bytes or len(pcm) > _MAX_PCM:
        raise ValueError("bounded exact PCM bytes required")


def _finite(pcm: bytes) -> None:
    for offset in range(0, len(pcm), _CHUNK):
        if not all(math.isfinite(v[0]) for v in struct.iter_unpack("<f", memoryview(pcm)[offset:offset + _CHUNK])):
            raise audio.AudioPreparationError("malformed_pcm", "Nonfinite normalized PCM")


class OwnedNormalizationRunner:
    """Single-use PCMRunner backed by the existing sole-child/metadata lifecycle."""

    def __init__(self, control: HostedObservationControl, files: NormalizationFiles,
                 schedule: audio.AudioSchedule, binding: audio.SourceClockBinding,
                 audio_format: audio.AudioFormat, runtime: RenderRuntime,
                 receipt: audio.NormalizationReceipt | None = None) -> None:
        self._control, self._files = control, files
        self._schedule, self._binding, self._format, self._runtime = schedule, binding, audio_format, runtime
        self._receipt = receipt
        self._used = False
        self.evidence: NormalizationPassEvidence | None = None
        self.stdout: bytes | None = None
        self.input: bytes | None = None
        self.input_file: FileEvidence | None = None

    def __call__(self, argv: tuple[str, ...], data: bytes, *, measure: bool) -> subprocess.CompletedProcess[bytes]:
        control = self._control
        control._check()
        if self._used or type(measure) is not bool or measure != (self._receipt is None):
            raise ValueError("single-use exact normalization pass required")
        self._used = True
        if (type(data) is not bytes or len(data) != 4 * self._format.channels * self._schedule.enabled_samples
                or hashlib.sha256(data).hexdigest() != self._binding.selected_pcm_sha256):
            raise ValueError("runner input binding/count mismatch")
        if measure:
            filters = f"loudnorm={audio.TARGETS}:print_format=json"
        else:
            assert self._receipt is not None
            filters = audio.normalization_filter(self._receipt)
        expected = (str(self._runtime.ffmpeg), "-hide_banner", "-loglevel", "info", "-f", "f32le",
            "-ar", "48000", "-ac", str(self._format.channels), "-channel_layout", self._format.layout,
            "-i", "pipe:0", "-af", filters, *(("-f", "null", "-") if measure else
            ("-c:a", "pcm_f32le", "-f", "f32le", "pipe:1")))
        if (type(argv) is not tuple or len(argv) != len(expected)
                or any(type(v) is not str for v in argv) or argv != expected or self.input_file is None):
            raise ValueError("exact engine argv and verified file required")
        self.input = data
        actual = (*argv[:13], str(self._files._paths["enabled"]), *argv[14:])
        self._files._claim_enabled_input(control)
        control._note_payload(True, True)
        lease = control._launch("normalizer", actual, _PASS_NS)
        log = bytearray(_LOG_CAP)
        log_size = 0

        def read_log() -> None:
            nonlocal log_size
            while True:
                line = control._readline_stderr(lease, min(16385, _LOG_CAP - log_size + 1))
                if not line:
                    return
                # Owner hooks count and reject overflow before any log publication.
                log[log_size:log_size + len(line)] = line
                log_size += len(line)

        control._start_metadata(lease, read_log)
        size = 0 if measure else len(data)
        if measure:
            if control._read_stdout(lease, 1):
                control._fail("protocol", "normalizer")
                control._job._raise()
            stdout = b""
        else:
            accumulator = bytearray(size)
            offset = 0
            while offset < size:
                count = control._readinto_stdout(lease, memoryview(accumulator)[offset:min(size, offset + _CHUNK)])
                if not count:
                    raise audio.AudioPreparationError("sample_count_mismatch", "Short normalized PCM")
                offset += count
            if control._read_stdout(lease, 1):
                raise audio.AudioPreparationError("sample_count_mismatch", "Extra normalized PCM")
            stdout = bytes(accumulator)
            del accumulator
        outcome = control._finish_child(lease, True)
        diagnostic = control.ownership_snapshot().diagnostics[-1]
        self.stdout = stdout
        if (diagnostic.stdout_bytes != len(stdout)
                or diagnostic.stdout_sha256 != hashlib.sha256(stdout).hexdigest()):
            raise ValueError("owned diagnostic/stdout association mismatch")
        self.evidence = NormalizationPassEvidence("measure" if measure else "process",
            lease.observation_id, lease.generation, argv, actual, self.input_file,
            len(stdout), diagnostic.stdout_sha256, log_size, outcome.natural_eof, outcome.exit_code)
        return subprocess.CompletedProcess(actual, outcome.exit_code, stdout, bytes(memoryview(log)[:log_size]))


def prepare_normalized_audio(schedule: audio.AudioSchedule, enabled_pcm: bytes, *,
        binding: audio.SourceClockBinding | None, audio_format: audio.AudioFormat,
        runtime: RenderRuntime, job: SourceHostJob, files: NormalizationFiles) -> PreparedNormalization | None:
    _guard(schedule, enabled_pcm, binding, audio_format, runtime)
    if type(job) is not SourceHostJob or type(files) is not NormalizationFiles:
        raise ValueError("exact owned HOST job/files required")
    job.check()
    if schedule.enabled_samples == 0:
        if enabled_pcm != b"" or binding is not None:
            raise ValueError("empty enabled bus requires empty bytes and no binding")
        return None
    if binding is None:
        raise ValueError("enabled bus source binding required")
    if (binding.source_fingerprint != schedule.source_fingerprint
            or len(enabled_pcm) != 4 * audio_format.channels * schedule.enabled_samples
            or hashlib.sha256(enabled_pcm).hexdigest() != binding.selected_pcm_sha256):
        raise ValueError("enabled bus source/hash/count mismatch")
    files._begin(job, runtime)
    size, digest = len(enabled_pcm), binding.selected_pcm_sha256
    try:
        files._write("enabled", enabled_pcm)
        files._verify("enabled", size, digest)
        receipt: audio.NormalizationReceipt | None = None
        passes: list[NormalizationPassEvidence] = []
        normalized: audio.NormalizedPCM | None = None
        for measure in (True, False):
            job.check()
            control = job.new_normalization(deadline_ns=min(job._deadline, job._now() + _PASS_NS))
            control._enter("normalization", _LOG_CAP, 1 if measure else size, _PASS_NS, job._grace)
            try:
                evidence = files._verify("enabled", size, digest)
                runner = OwnedNormalizationRunner(control, files, schedule, binding, audio_format, runtime, receipt)
                runner.input_file = evidence
                if measure:
                    receipt = audio.measure_normalization(schedule, enabled_pcm, binding=binding,
                        audio_format=audio_format, runtime=runtime, _runner=runner, _runtime_files=files)
                    audio._prepared_receipt(receipt)
                    audio._validate_receipt(schedule, receipt, binding, audio_format, runtime)
                else:
                    assert receipt is not None
                    normalized = audio.process_normalized_pcm(schedule, enabled_pcm, receipt=receipt,
                        binding=binding, audio_format=audio_format, runtime=runtime, _runner=runner, _runtime_files=files)
                    _finite(normalized.pcm)
                    if normalized.pcm is not runner.stdout:
                        raise ValueError("normalization helper/runner stdout identity mismatch")
                control._check()
                post = files._verify("enabled", size, digest)
                if post != evidence or runner.input is not enabled_pcm or runner.evidence is None:
                    raise ValueError("pass/file/input association mismatch")
                passes.append(runner.evidence)
                control._note_payload(False, False, 0 if measure else size)
                control._leave(True)
                if control.ownership_snapshot().phase != "retired_ok":
                    raise ValueError("normalization pass did not certainly retire")
            except Exception:
                control._note_payload(False, False)
                if control._phase not in ("retired_ok", "stopped_clean"):
                    control._leave(False)
                raise
        assert receipt is not None and normalized is not None
        output_hash = hashlib.sha256(normalized.pcm).hexdigest()
        if passes[1].stdout_bytes != size or passes[1].stdout_sha256 != output_hash:
            raise ValueError("normalized stdout count/hash association mismatch")
        files._write("normalized", normalized.pcm)
        output = files._verify("normalized", size, output_hash)
        files._delete("enabled")
        result = PreparedNormalization(schedule.identity, binding, audio._runtime_identity(runtime),
            audio_format, receipt, (passes[0], passes[1]), normalized, output)
        files._seal(result)
        return result
    except Exception:
        if job._failure is None:
            job.fail("validation", "normalizer")
        files.retire()
        raise
