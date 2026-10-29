"""Task112 candidate-only component qualification; no default host activation.

The live runtime selector and its qualification identity are never consulted.
Actual publication/admission identities are mandatory external Root operands.
"""

import hashlib
import json
import math
import re
import struct
from dataclasses import dataclass
from pathlib import Path, PureWindowsPath

MiB = 1048576
WORKSPACE = "O:/Escritorio/ClipperX"
BASE = WORKSPACE + "/.superpowers/sdd/upstream-motion"
PUBLICATION_FIELDS = ("build_id source_sha256 recovery_sha256 reuse_sha256 continuation_sha256 "
    "parent_source_sha256 reuse_manifest_path iconv harfbuzz_source source_commit source_contract "
    "publisher_source_sha256 publisher_lock_sha256 license_mode status pair notices inspection_sha256 "
    "closure_review_sha256 qualification").split()


NOTICE_NAMES = ("ffmpeg-COPYING.GPLv2", "libass-COPYING", "harfbuzz-COPYING", "freetype-LICENSE.TXT",
    "fribidi-COPYING", "x264-COPYING", "dav1d-COPYING", "nv-codec-headers-README", "libiconv-COPYING", "libiconv-COPYING.LIB")
SOURCE_CONTRACT = dict(local_registered_file=True, one_video_one_mixed_audio=True, positive_even_dimensions=True,
    no_codec_allowlist=True, retained_native_families="mov_mp4 matroska_webm mpeg_ts flv avi common_audio h264 hevc vp8 vp9 mpeg png aac opus vorbis mp3 flac qtrle pcm".split(),
    new_external_decoder="libdav1d_cpu_av1", delivered_encoders=["h264_nvenc", "libx264", "aac"],
    known_external_omissions="libdavs2 libuavs3d libxevd libjxl libsvtjpegxs libcodec2 liblc3 libgme libmodplug libopenmpt".split(),
    omission_gate="compare accepted plan/source metadata and any ingest contract before candidate promotion; required missing codec is NEEDS_CONTEXT")
O9 = BASE + "/production-build-v9-hbft"
BUILD_ID = "ffmpeg-9.0.2-clipperx-production-v9-hbft"
RECORD_PATHS = {
    "publisher_source": O9 + "/recovery.py",
    "publisher_lock": O9 + "/recovery-lock.json",
    "publisher_review": WORKSPACE + "/.superpowers/sdd/task-125-source-review.md",
    "inspection": O9 + "/evidence/inspect/inspection.json",
    "closure_review": O9 + "/reviewed/publication-closure.json",
    "source_notice": O9 + "/publish/SOURCE-NOTICE.json",
    "ffmpeg_map": O9 + "/build/ffmpeg/ffmpeg-link.map",
    "ffprobe_map": O9 + "/build/ffmpeg/ffprobe-link.map",
}
RECORD_CAPS = {"inspection": 8 * MiB, "source_notice": 4 * MiB,
    "ffmpeg_map": 64 * MiB, "ffprobe_map": 64 * MiB}
SOURCE_PATHS = {"controller": WORKSPACE + "/scripts/qualify_production_runtime.py",
    "tests": WORKSPACE + "/clipperx/tests/test_production_runtime_qualification.py",
    "design": WORKSPACE + "/.superpowers/sdd/task-116-compact-component-design.md",
    "design_review": WORKSPACE + "/.superpowers/sdd/task-116-review.md"}
ACCEPTED_DESIGN_PINS = {"design": (12508, "960CACE54A3C9B02ED7A331E6C2EE169AE277F0CF87293D675F70133C9293CEC"),
    "design_review": (12562, "3E84A3A1FE7CDA1FA46A165AB24F30C55B8C59502317446466A37CD039A0B2D8")}
HELPER_SYMBOLS = {
    "_source_host_ownership": ("SourceHostJob",), "_retained_files": ("RetainedFiles",),
    "_render_services": ("OwnedRenderTools",),
    "_windows_pipe_io": (), "audio_mix": (), "filtergraph": (), "local_paths": (),
    "motion_compiler": (), "output_timeline": (), "overlay_ass": (), "planner_contracts": (),
    "schema": (), "visual_contracts": (),
    "render_clock": ("SourceTimeOrigin", "StreamClock", "source_decoder_identity", "video_decode_argv",
        "audio_decode_argv", "float_audio_decode_argv"),
    "render_contracts": ("RenderProfile",), "render_runtime": (),
    "visual_compiler": ("encoder_arguments",),
    "render_metrics": ("parse_frame_observations", "validate_output_observations"),
    "source_decode": ("parse_video", "parse_audio"),
    "font_staging": ("stage_original_fonts", "verify_font_stage"),
    "render_assets": ("prepare_brand_asset",),
}
LINEAGE_PATHS = (WORKSPACE + "/.superpowers/sdd/task-106-component-qualification-design.md",
    WORKSPACE + "/.superpowers/sdd/task-112-brief.md",
    WORKSPACE + "/.superpowers/sdd/task-120-ffmpeg-recovery-design.md",
    WORKSPACE + "/.superpowers/sdd/task-120-design-review.md",
    WORKSPACE + "/docs/superpowers/specs/2026-10-21-hbft-recovery-design.md",
    WORKSPACE + "/.superpowers/sdd/task-125-design-review.md")
LINEAGE_PINS = {
    LINEAGE_PATHS[2]: (12443, "3C50B97C24C22EF67C17EE05E826CA248E6FD16D9130DD7B433B47CDB3A04B1D"),
    LINEAGE_PATHS[3]: (10251, "51435A3253D5B47F20FA33DA34C5A5712D0A035DD5A50DC129EF6A8AF89B925C"),
    LINEAGE_PATHS[4]: (17969, "FF9A593DD3C7067ED25FE04590568DADF945867F3C4C1A9FDE09CBE9A22DB061"),
    LINEAGE_PATHS[5]: (15345, "D66FE38156E801398A9FD0B4A724C0C106BD96D82AEF361DA702C20315FEE217"),
}
ASSET_PATHS = {"headline_impact": "C:/Windows/Fonts/impact.ttf",
    "subtitle_bold": "C:/Windows/Fonts/arialbd.ttf",
    "kick_brand": WORKSPACE + "/assets/branding/KickAthanW.png"}


def closed(value, keys):
    if type(value) is not dict or set(value) != set(keys.split() if isinstance(keys, str) else keys):
        raise ValueError("missing/extra closed schema fields")
    return value


def integer(value, low=0, high=2**63 - 1):
    if type(value) is not int or not low <= value <= high:
        raise ValueError("bounded plain integer required")
    return value


def digest(raw):
    return hashlib.sha256(raw).hexdigest()


def canonical_json(value):
    return json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=True, allow_nan=False).encode("utf-8")


def require_equal(actual, expected, label):
    if canonical_json(actual) != canonical_json(expected):
        raise ValueError(label + " mismatch")


def same_hash(actual, expected):
    if (type(actual) is not str or not re.fullmatch(r"[a-fA-F0-9]{64}", actual)
            or type(expected) is not str or not re.fullmatch(r"[a-fA-F0-9]{64}", expected)
            or actual.lower() != expected.lower()):
        raise ValueError("provenance digest mismatch")


def canonical_path(value):
    if type(value) is not str or not 1 <= len(value) <= 2048 or "\0" in value:
        raise ValueError("bounded canonical path required")
    path = PureWindowsPath(value)
    if (not path.is_absolute() or not re.fullmatch(r"[A-Za-z]:", path.drive) or value.startswith(("\\\\", "//"))
            or ":" in value[2:] or any(p in (".", "..") or p.endswith((".", " ")) or "~" in p for p in value.replace("\\", "/").split("/")[1:])):
        raise ValueError("local canonical path required; no aliases/devices/ADS")
    if "//" in value[2:] or "\\\\" in value[2:]:
        raise ValueError("path separator alias")
    return path.as_posix()


@dataclass(frozen=True, slots=True)
class Pin:
    path: str
    sha256: str
    bytes: int

    @classmethod
    def parse(cls, value, cap=MiB):
        closed(value, "path sha256 bytes")
        canonical_path(value["path"])
        if type(value["sha256"]) is not str or not re.fullmatch(r"[a-fA-F0-9]{64}", value["sha256"]):
            raise ValueError("SHA256 required")
        # Producer bytes remain the authority. Normalize only comparison keys;
        # serializing a parsed Pin must preserve the admitted path spelling.
        return cls(value["path"], value["sha256"], integer(value["bytes"], 1, cap))

    def record(self):
        return dict(path=self.path, sha256=self.sha256, bytes=self.bytes)


def parse_document(raw, role="document"):
    """Default small-document JSON grammar; opaque records are never parsed."""
    cap = {"runtime_acceptance": 16384, "publication": 16384}.get(role, MiB)
    parsed = _decode_json(raw, cap)
    def walk(value, depth=0):
        if depth > 12:
            raise ValueError("JSON depth cap")
        if type(value) is dict:
            if len(value) > 128:
                raise ValueError("JSON object cap")
            for key, child in value.items():
                if len(key) > 2048:
                    raise ValueError("JSON key cap")
                walk(child, depth + 1)
        elif type(value) is list:
            if len(value) > 256:
                raise ValueError("JSON list cap")
            for child in value:
                walk(child, depth + 1)
        elif type(value) is str:
            if len(value) > 2048:
                raise ValueError("JSON string cap")
        elif type(value) is float and not math.isfinite(value):
            raise ValueError("nonfinite JSON scalar")
    walk(parsed)
    return parsed


def _decode_json(raw, cap):
    if type(raw) is not bytes or len(raw) > cap:
        raise ValueError("raw document byte cap")
    def pairs(rows):
        result = {}
        for key, value in rows:
            if key in result:
                raise ValueError("duplicate JSON key")
            result[key] = value
        return result
    def nonfinite(value):
        raise ValueError("nonfinite JSON scalar")
    try:
        parsed = json.loads(raw.decode("utf-8"), object_pairs_hook=pairs, parse_constant=nonfinite)
    except (UnicodeError, RecursionError) as exc:
        raise ValueError("bounded UTF-8 JSON required") from exc
    return parsed


def operation_limits():
    values = [("probe", 15, 1000000, 0)] * 8 + [
        ("encoder", 90, 1, 16 * MiB), ("probe", 45, 1000000, 0),
        ("spatial", 45, 184320, 184320), ("finalmix", 45, 192000, 192000),
        ("finalmix", 45, 384000, 384000), ("finalmix", 120, 384000, 384000),
        ("probe", 45, 65536, 65536), ("spatial", 45, 1024, 1024),
        ("spatial", 90, 13824000, 13824000), ("spatial", 90, 36864, 36864),
        ("encoder", 180, 1, 16 * MiB), ("probe", 45, 1000000, 0), ("probe", 45, 1, 0),
        ("encoder", 180, 1, 16 * MiB), ("probe", 45, 1000000, 0), ("probe", 45, 1, 0)]
    return [dict(index=i, kind=k, timeout_seconds=t, stdout_bytes=s, artifact_bytes=a)
        for i, (k, t, s, a) in enumerate(values, 1)]


def limits():
    return dict(children=24, campaign_seconds=1200, work_seconds=1185, cleanup_seconds=5, settlement_seconds=10,
        owned_bytes=268435456, capture_sidecar_bytes=33554432, stderr_bytes=1572864, max_paths=256,
        max_generations=4096, max_live=16, free_reserve_bytes=1073741824, guard_bytes=16384,
        operation_record_bytes=262144, terminal_bytes=2097152, probe_capture_bytes=1000000,
        stderr_child_bytes=65536, stderr_line_bytes=16384, graph_bytes=65536, ass_bytes=16384,
        list_bytes=4096, operations=operation_limits())


@dataclass(frozen=True, slots=True)
class ValidatedInputs:
    admission: Pin
    runtime_acceptance: Pin
    publication: Pin
    pair: object
    records: tuple
    notices: tuple
    source_pins: tuple
    assets: tuple
    run_id: str
    evidence_parent: str
    evidence_root: str
    build_id: str
    limits_json: bytes
    admission_raw: bytes
    acceptance_raw: bytes
    publication_raw: bytes
    started_ns: int


def _read_verified(pin, cap, *, io, clock, deadline, paths, identities, keep=False):
    """One bounded, closed stream per actual file; opaque records are not cached."""
    if type(pin) is not Pin:
        raise ValueError("Pin required")
    Pin.parse(pin.record(), cap)
    if clock() >= deadline:
        raise ValueError("campaign work deadline during admission")
    lexical = canonical_path(pin.path)
    if pin.path != lexical:
        raise ValueError("noncanonical input path alias")
    key = lexical.casefold()
    old = paths.get(key)
    if old is not None:
        if old.path != pin.path or old.bytes != pin.bytes or old.sha256.lower() != pin.sha256.lower():
            raise ValueError("conflicting same-path pin/alias")
        if keep and key not in identities["small"]:
            raise ValueError("repeated unretained input requested as JSON")
        return identities["small"].get(key)
    if len(paths) >= 288:
        raise ValueError("bounded distinct input documents")
    io.check_path(pin.path)
    observed = 0
    hasher = hashlib.sha256()
    chunks = [] if keep else None
    with io.open_read(pin.path) as stream:
        identity = io.stream_identity(stream)
        if identity in identities["files"]:
            raise ValueError("distinct-path file-identity alias")
        identities["files"].add(identity)
        paths[key] = pin
        while True:
            if clock() >= deadline:
                raise ValueError("campaign work deadline during input hashing")
            chunk = stream.read(min(MiB, pin.bytes + 1 - observed))
            if not chunk:
                break
            observed += len(chunk)
            if observed > pin.bytes:
                raise ValueError("input pin size mismatch")
            hasher.update(chunk)
            if keep:
                chunks.append(chunk)
    if observed != pin.bytes or hasher.hexdigest() != pin.sha256.lower():
        raise ValueError("input pin hash/size mismatch")
    if clock() >= deadline:
        raise ValueError("campaign work deadline after input hashing")
    if keep:
        raw = b"".join(chunks)
        identities["small"][key] = raw
        return raw
    return None


def _pin_at(value, path, cap):
    pin = Pin.parse(value, cap)
    if canonical_path(pin.path) != path or pin.path != path:
        raise ValueError("exact role path required")
    return pin


def _same_pin(value, expected, cap):
    pin = Pin.parse(value, cap)
    if (pin.path != expected.path or pin.bytes != expected.bytes
            or pin.sha256.lower() != expected.sha256.lower()):
        raise ValueError("pin identity mismatch")
    return pin


def _same_root(a, b):
    x, y = PureWindowsPath(a.casefold()), PureWindowsPath(b.casefold())
    return x == y or x in y.parents or y in x.parents


def _source_origins(source_pins):
    """Called only after all helper bytes have passed custody checks."""
    import importlib
    import importlib.util
    import inspect
    import sys
    pinned = {canonical_path(pin.path): pin for pin in source_pins}
    controller = sys.modules.get(__name__)
    expected_controller = SOURCE_PATHS["controller"]
    if (canonical_path(__file__) != expected_controller
            or canonical_path(validate_inputs.__code__.co_filename) != expected_controller
            or controller is not None and (
                canonical_path(controller.__file__) != expected_controller
                or controller.__spec__ is not None and canonical_path(controller.__spec__.origin) != expected_controller)):
        raise ValueError("controller module/spec/function origin mismatch")
    found = {}
    for short, symbols in HELPER_SYMBOLS.items():
        name = "clipperx." + short
        expected = WORKSPACE + "/clipperx/" + short + ".py"
        if expected not in pinned:
            raise ValueError("unpinned local helper import " + name)
        loaded = sys.modules.get(name)
        spec = loaded.__spec__ if loaded is not None else importlib.util.find_spec(name)
        if spec is None or spec.origin is None or canonical_path(spec.origin) != expected:
            raise ValueError("helper spec origin mismatch " + name)
        if loaded is not None and canonical_path(loaded.__file__) != expected:
            raise ValueError("loaded helper origin mismatch " + name)
        found[name] = loaded
    for short, symbols in HELPER_SYMBOLS.items():
        name = "clipperx." + short
        expected = WORKSPACE + "/clipperx/" + short + ".py"
        loaded = found[name] or importlib.import_module(name)
        if (canonical_path(loaded.__file__) != expected or loaded.__spec__ is None
                or canonical_path(loaded.__spec__.origin) != expected):
            raise ValueError("imported helper origin mismatch " + name)
        for symbol in symbols:
            obj = getattr(loaded, symbol, None)
            if obj is None or getattr(obj, "__module__", None) != name:
                raise ValueError("helper function origin mismatch " + name + "." + symbol)
            filename = inspect.getsourcefile(obj)
            if filename is None or canonical_path(filename) != expected:
                raise ValueError("helper source origin mismatch " + name + "." + symbol)


def _validate_compact_candidate(candidate, acceptance, records, pair, notices):
    closed(candidate, PUBLICATION_FIELDS)
    if (candidate["build_id"] != BUILD_ID or candidate["status"] != "candidate_published"
            or candidate["qualification"] != "pending separate original media/90-second/production gates"
            or candidate["license_mode"] != "GPL due to libx264; complete source/license publication needs review"):
        raise ValueError("candidate publication status/build identity")
    for field in ("source_sha256", "recovery_sha256", "reuse_sha256", "continuation_sha256",
            "parent_source_sha256", "publisher_source_sha256", "publisher_lock_sha256",
            "inspection_sha256", "closure_review_sha256"):
        if type(candidate[field]) is not str or not re.fullmatch(r"[a-fA-F0-9]{64}", candidate[field]):
            raise ValueError("candidate digest grammar " + field)
    for field, role in (("publisher_source_sha256", "publisher_source"),
            ("publisher_lock_sha256", "publisher_lock"), ("inspection_sha256", "inspection"),
            ("closure_review_sha256", "closure_review")):
        same_hash(candidate[field], records[role].sha256)
    if type(candidate["source_commit"]) is not str or not re.fullmatch(r"[a-f0-9]{40}", candidate["source_commit"]):
        raise ValueError("candidate source commit")
    for field in ("reuse_manifest_path", "harfbuzz_source"):
        if type(candidate[field]) is not str or canonical_path(candidate[field]) != candidate[field]:
            raise ValueError("candidate canonical metadata path " + field)
    iconv = closed(candidate["iconv"], "archive rejected_import header package copying copying_lib provenance_limit")
    if iconv["provenance_limit"] != "existing local MSYS2 package 1.17-4 metadata, not independently reverified distribution authenticity":
        raise ValueError("candidate iconv provenance limit")
    iconv_paths = []
    for role in ("archive", "rejected_import", "header", "package", "copying", "copying_lib"):
        pin = Pin.parse(iconv[role], 16 * MiB)
        canonical = canonical_path(pin.path)
        if pin.path != canonical:
            raise ValueError("iconv canonical recorded Pin path required")
        iconv_paths.append(canonical.casefold())
    if len(set(iconv_paths)) != 6:
        raise ValueError("six distinct iconv Pin metadata roles")
    require_equal(candidate["source_contract"], SOURCE_CONTRACT, "candidate media source contract")
    closed(candidate["pair"], "ffmpeg ffprobe")
    _same_pin(candidate["pair"]["ffmpeg"], pair.ffmpeg, 268435456)
    _same_pin(candidate["pair"]["ffprobe"], pair.ffprobe, 268435456)
    if type(candidate["notices"]) is not list or len(candidate["notices"]) != 10:
        raise ValueError("candidate exact ordered notices")
    for value, expected in zip(candidate["notices"], notices, strict=True):
        _same_pin(value, expected, MiB)
    if candidate["build_id"] != acceptance["build_id"]:
        raise ValueError("candidate/Root build identity mismatch")


def validate_inputs(cli_pin, *, io=None, clock=None):
    """Bind external v2 admission, Root acceptance, actual candidate, and raw custody."""
    import time
    clock = time.monotonic_ns if clock is None else clock
    io = DetachedIO() if io is None else io
    started_ns = clock()
    deadline = started_ns + 1185 * 10**9
    paths, identities = {}, {"files": set(), "small": {}}
    if type(cli_pin) is not Pin:
        raise ValueError("external admission Pin required")
    admission_raw = _read_verified(cli_pin, MiB, io=io, clock=clock, deadline=deadline,
        paths=paths, identities=identities, keep=True)
    admission = parse_document(admission_raw)
    closed(admission, "schema run_id evidence_parent evidence_root runtime_acceptance source_pins assets limits")
    if admission["schema"] != "clipperx-candidate-components-admission-v2":
        raise ValueError("admission v2 schema required")
    run_id = admission["run_id"]
    if type(run_id) is not str or not re.fullmatch(r"[a-z0-9][a-z0-9-]{0,63}", run_id):
        raise ValueError("bounded run identity")
    parent, evidence_root = map(canonical_path, (admission["evidence_parent"], admission["evidence_root"]))
    if (admission["evidence_parent"] != parent or admission["evidence_root"] != evidence_root
            or PureWindowsPath(evidence_root).parent != PureWindowsPath(parent)
            or PureWindowsPath(evidence_root).name != run_id):
        raise ValueError("exact absent immediate evidence child required")
    io.check_path(parent, directory=True)
    io.check_path(evidence_root, absent=True)
    if _same_root(evidence_root, O9):
        raise ValueError("evidence/runtime root overlap")
    accepted_pin = _pin_at(admission["runtime_acceptance"], O9 + "/reviewed/component-runtime-acceptance.json", 16384)
    acceptance_raw = _read_verified(accepted_pin, 16384, io=io, clock=clock, deadline=deadline,
        paths=paths, identities=identities, keep=True)
    acceptance = parse_document(acceptance_raw, role="runtime_acceptance")
    closed(acceptance, "schema status build_id runtime_root publication pair records notices")
    if (acceptance["schema"] != "clipperx-component-runtime-acceptance-v1"
            or acceptance["status"] != "accepted_for_component_qualification"
            or acceptance["build_id"] != BUILD_ID
            or acceptance["runtime_root"] != O9):
        raise ValueError("fixed Root acceptance identity")
    publication = _pin_at(acceptance["publication"], O9 + "/publish/candidate.json", 16384)
    closed(acceptance["pair"], "ffmpeg ffprobe")
    pair = CandidatePair(*(_pin_at(acceptance["pair"][name], O9 + "/publish/bin/" + name + ".exe", 268435456)
        for name in ("ffmpeg", "ffprobe")))
    closed(acceptance["records"], RECORD_PATHS)
    records = tuple((role, _pin_at(acceptance["records"][role], path, RECORD_CAPS.get(role, MiB)))
        for role, path in RECORD_PATHS.items())
    if type(acceptance["notices"]) is not list or len(acceptance["notices"]) != 10:
        raise ValueError("exact ordered ten notices required")
    notices = tuple(_pin_at(value, O9 + "/publish/notices/" + name, MiB)
        for value, name in zip(acceptance["notices"], NOTICE_NAMES, strict=True))
    if len({pin.path.casefold() for pin in notices}) != 10:
        raise ValueError("notice path alias")
    publication_raw = _read_verified(publication, 16384, io=io, clock=clock, deadline=deadline,
        paths=paths, identities=identities, keep=True)
    candidate = parse_document(publication_raw, role="publication")
    _validate_compact_candidate(candidate, acceptance, dict(records), pair, notices)
    closed(admission["source_pins"], "controller tests design design_review dependencies")
    sources = tuple((role, _pin_at(admission["source_pins"][role], path, MiB))
        for role, path in SOURCE_PATHS.items())
    for role, pin in sources:
        if role in ACCEPTED_DESIGN_PINS:
            size, sha = ACCEPTED_DESIGN_PINS[role]
            if pin.bytes != size or pin.sha256.lower() != sha.lower():
                raise ValueError("accepted design/review source pin drift")
    dependencies = admission["source_pins"]["dependencies"]
    if type(dependencies) is not list or not 1 <= len(dependencies) <= 64:
        raise ValueError("1..64 source dependency pins required")
    dependency_pins = tuple(Pin.parse(row) for row in dependencies)
    dep_paths = [canonical_path(pin.path) for pin in dependency_pins]
    if dep_paths != sorted(dep_paths) or len(set(p.casefold() for p in dep_paths)) != len(dep_paths):
        raise ValueError("unique sorted source dependencies required")
    needed = {WORKSPACE + "/clipperx/" + name + ".py" for name in HELPER_SYMBOLS} | set(LINEAGE_PATHS)
    if not needed.issubset(dep_paths):
        raise ValueError("unpinned helper or required Task106/Task112/Task120/Task125 lineage")
    for pin in dependency_pins:
        if pin.path in LINEAGE_PINS:
            size, sha = LINEAGE_PINS[pin.path]
            if pin.bytes != size or pin.sha256.lower() != sha.lower():
                raise ValueError("accepted Task120/Task125 design/review dependency pin drift")
    if any(path == SOURCE_PATHS[role] for path in dep_paths for role in SOURCE_PATHS):
        raise ValueError("duplicate named source/dependency role")
    closed(admission["assets"], ASSET_PATHS)
    assets = tuple((role, _pin_at(admission["assets"][role], path, 16 * MiB))
        for role, path in ASSET_PATHS.items())
    if canonical_json(admission["limits"]) != canonical_json(limits()):
        raise ValueError("exact plain-int campaign limits required")
    role_pins = (cli_pin, accepted_pin, publication, pair.ffmpeg, pair.ffprobe,
        *(pin for _, pin in records), *notices, *(pin for _, pin in sources),
        *dependency_pins, *(pin for _, pin in assets))
    if len({canonical_path(pin.path).casefold() for pin in role_pins}) != len(role_pins):
        raise ValueError("distinct mandatory input role pin paths required")
    if any(_same_root(evidence_root, pin.path) for pin in role_pins):
        raise ValueError("evidence/input namespace overlap")
    for pin, cap in ((pair.ffmpeg, 268435456), (pair.ffprobe, 268435456),
            *((pin, RECORD_CAPS.get(role, MiB)) for role, pin in records),
            *((pin, MiB) for pin in notices), *((pin, MiB) for _, pin in sources),
            *((pin, MiB) for pin in dependency_pins), *((pin, 16 * MiB) for _, pin in assets)):
        _read_verified(pin, cap, io=io, clock=clock, deadline=deadline,
            paths=paths, identities=identities)
    _source_origins(tuple(pin for _, pin in sources) + dependency_pins)
    result = ValidatedInputs(cli_pin, accepted_pin, publication, pair, records, notices,
        sources + (("dependencies", dependency_pins),), assets, run_id, parent, evidence_root,
        BUILD_ID, canonical_json(admission["limits"]), admission_raw, acceptance_raw, publication_raw, started_ns)
    guard_bytes(result, cli_pin)
    return result


def guard_bytes(inputs, cli_pin):
    if type(inputs) is not ValidatedInputs or type(cli_pin) is not Pin or inputs.admission != cli_pin:
        raise ValueError("guard external admission pin identity")
    if (len(inputs.admission_raw) != cli_pin.bytes or digest(inputs.admission_raw) != cli_pin.sha256.lower()
            or len(inputs.acceptance_raw) != inputs.runtime_acceptance.bytes
            or digest(inputs.acceptance_raw) != inputs.runtime_acceptance.sha256.lower()
            or len(inputs.publication_raw) != inputs.publication.bytes
            or digest(inputs.publication_raw) != inputs.publication.sha256.lower()):
        raise ValueError("guard bound raw pin hash/size drift")
    admission = parse_document(inputs.admission_raw)
    acceptance = parse_document(inputs.acceptance_raw, role="runtime_acceptance")
    _same_pin(admission["runtime_acceptance"], inputs.runtime_acceptance, 16384)
    _same_pin(acceptance["publication"], inputs.publication, 16384)
    for name in ("ffmpeg", "ffprobe"):
        _same_pin(acceptance["pair"][name], getattr(inputs.pair, name), 268435456)
    if (admission["run_id"] != inputs.run_id or admission["evidence_parent"] != inputs.evidence_parent
            or admission["evidence_root"] != inputs.evidence_root or acceptance["build_id"] != inputs.build_id
            or canonical_json(admission["limits"]) != inputs.limits_json):
        raise ValueError("guard field projection drift")
    value = dict(schema="clipperx-candidate-components-guard-v2", admission=cli_pin.record(),
        runtime_acceptance=inputs.runtime_acceptance.record(), publication=inputs.publication.record(),
        pair=dict(ffmpeg=inputs.pair.ffmpeg.record(), ffprobe=inputs.pair.ffprobe.record()),
        run_id=inputs.run_id, evidence_parent=inputs.evidence_parent, evidence_root=inputs.evidence_root,
        build_id=inputs.build_id, limits=admission["limits"])
    try:
        raw = canonical_json(value)
    except (TypeError, ValueError, OverflowError) as exc:
        raise ValueError("guard serialization failure") from exc
    if len(raw) > 16384:
        raise ValueError("pre-creation compact guard cap")
    if set(json.loads(raw)) != set("schema admission runtime_acceptance publication pair run_id evidence_parent evidence_root build_id limits".split()):
        raise ValueError("guard exact ten keys")
    return raw


def fixture_bytes():
    bgr = bytes(v for n in range(60) for y in range(32) for x in range(32) for v in (16 + x, 32 + y, 64 + n))
    words = (0, 0x80000000, 1, 0x80000001, 0x3E800000, 0xBE800000, 0x3FC00000, 0xBFC00000)
    floats = struct.pack("<8I", *words) * 12000
    tone = b"".join(struct.pack("<ff", (v := .25 if j % 96 < 48 else -.25), -v) for j in range(48000))
    alpha = bytes(v for y in range(16) for x in range(16) for v in (240, 40, 80, 0 if x < 5 else 128 if x < 11 else 255))
    ass = ("[Script Info]\nScriptType: v4.00+\nPlayResX: 320\nPlayResY: 180\nWrapStyle: 2\nScaledBorderAndShadow: yes\n\n"
        "[V4+ Styles]\nFormat: Name, Fontname, Fontsize, PrimaryColour, SecondaryColour, OutlineColour, BackColour, Bold, Italic, Underline, StrikeOut, ScaleX, ScaleY, Spacing, Angle, BorderStyle, Outline, Shadow, Alignment, MarginL, MarginR, MarginV, Encoding\n"
        "Style: Default,Arial,20,&H00FFFFFF,&H00FFFFFF,&H00000000,&H00000000,-1,0,0,0,100,100,0,0,1,1,0,2,10,10,8,1\n\n"
        "[Events]\nFormat: Layer, Start, End, Style, Name, MarginL, MarginR, MarginV, Effect, Text\n"
        "Dialogue: 0,0:00:00.25,0:00:00.75,Default,,0,0,0,,Ágil مرحبا\n").encode("utf-8")
    return {"source.bgr": bgr, "source.f32": floats, "tone.f32": tone, "alpha.rgba": alpha,
        "shots.list": b"raster.rgba\n", "text.ass": ass}


@dataclass(frozen=True, slots=True)
class CandidatePair:
    ffmpeg: Pin
    ffprobe: Pin

    def __post_init__(self):
        if (type(self.ffmpeg) is not Pin or type(self.ffprobe) is not Pin
                or self.ffmpeg.path.casefold() == self.ffprobe.path.casefold()
                or PureWindowsPath(self.ffmpeg.path).parent != PureWindowsPath(self.ffprobe.path).parent
                or PureWindowsPath(self.ffmpeg.path).name != "ffmpeg.exe"
                or PureWindowsPath(self.ffprobe.path).name != "ffprobe.exe"):
            raise ValueError("distinct canonical published candidate pair required")


def candidate_origin(path, size, mtime_ns, *, pair, video=None, audio=None):
    from fractions import Fraction
    from clipperx.render_clock import SourceTimeOrigin, StreamClock, source_decoder_identity
    if video is None:
        video = StreamClock(0, "video", Fraction(1, 60), Fraction(5, 4), Fraction(9, 4), width=32, height=32)
    if audio is None:
        audio = StreamClock(1, "audio", Fraction(1, 48000), Fraction(5, 4), Fraction(9, 4), sample_rate=48000, channels=2, sample_format="flt")
    fingerprint = digest(f"{path}|{size}|{mtime_ns}".encode())[:16]
    identity = source_decoder_identity(dict(runtime_id="candidate-components-v1", ffmpeg=pair.ffmpeg.path,
        ffprobe=pair.ffprobe.path, ffmpeg_sha256=pair.ffmpeg.sha256.upper(), ffprobe_sha256=pair.ffprobe.sha256.upper()))
    return SourceTimeOrigin(path, size, mtime_ns, fingerprint, Fraction(5, 4), Fraction(1), (video, audio), identity, {})


def operation_argv(pair, *, origin, brand, impact):
    from fractions import Fraction
    from clipperx.render_clock import video_decode_argv, audio_decode_argv, float_audio_decode_argv
    from clipperx.render_contracts import RenderProfile
    from clipperx.render_runtime import NATIVE_POLICY
    from clipperx.visual_compiler import encoder_arguments
    F, P = pair.ffmpeg.path, pair.ffprobe.path
    C = (F, "-hide_banner", "-nostdin", "-nostats", "-v", "info", "-xerror", "-n")
    def R(w, h, pix, path):
        return ("-f", "rawvideo", "-pixel_format", pix, "-video_size", f"{w}x{h}", "-framerate", "60", "-i", path)
    def A(path):
        return ("-f", "f32le", "-ar", "48000", "-ac", "2", "-i", path)
    def RAW(pix):
        return ("-an", "-sn", "-dn", "-threads", "1", "-c:v", "rawvideo", "-pix_fmt", pix,
            "-fps_mode", "passthrough", "-f", "rawvideo", "pipe:1")
    entries = "format=start_time,duration:stream=index,codec_name,codec_type,time_base,start_pts,duration_ts,width,height,pix_fmt,avg_frame_rate,nb_read_frames,sample_rate,channels,sample_fmt,color_range,color_space,color_transfer,color_primaries:frame=media_type,stream_index,pts,best_effort_timestamp_time,nb_samples"
    def Q(path):
        return (P, "-v", "error", "-count_frames", "-show_streams", "-show_format", "-show_frames", "-show_entries", entries, "-of", "json", path)
    def D(profile, path):
        return C + R(320, 180, "rgba", "concatf:shots.list") + A("tone.f32") + (
            "-map", "0:v:0", "-map", "1:a:0", "-vf", "scale=1080:1920:flags=bilinear,setsar=1", "-t", "1") + encoder_arguments(RenderProfile(video_encoder=profile)) + ("-fs", "8388608", path)
    V = ("format=gbrp,setparams=range=full:color_primaries=bt709:color_trc=linear:colorspace=gbr,"
        "libplacebo=w=48:h=48:format=rgba:crop_x='0*n':crop_y=0:crop_w=iw:crop_h=ih:pos_w=ow:pos_h=oh:"
        "pos_x=0:pos_y=0:fillcolor=black:alpha_mode=straight:reset_sar=1:" + NATIVE_POLICY)
    mix = "asplit=2[a][b];[a]volume=0.5[a1];[b]volume=0.25,adelay=24000S:all=1[b1];[a1][b1]amix=inputs=2:duration=first:normalize=0,apad=whole_len=48000,atrim=end_sample=48000,alimiter=limit=0.9:level=0:latency=1"
    def null(path):
        return C + ("-i", path, "-map", "0:v:0", "-map", "0:a:0", "-f", "null", "-")
    return (
        (F, "-version"), (P, "-version"), (F, "-hide_banner", "-buildconf"),
        (F, "-hide_banner", "-filters"), (F, "-hide_banner", "-codecs"), (F, "-hide_banner", "-formats"),
        (F, "-hide_banner", "-encoders"), (F, "-hide_banner", "-h", "filter=libplacebo"),
        C + R(32, 32, "bgr24", "source.bgr") + A("source.f32") + ("-map", "0:v:0", "-map", "1:a:0", "-c:v", "qtrle", "-c:a", "pcm_f32le", "-output_ts_offset", "1.25", "-use_editlist", "1", "-avoid_negative_ts", "disabled", "-fs", "8388608", "source.mov"),
        Q("source.mov"), tuple(video_decode_argv(F, origin, Fraction(0), Fraction(1))),
        tuple(audio_decode_argv(F, origin, Fraction(0), Fraction(1), "render48k-stereo")),
        tuple(float_audio_decode_argv(F, origin, Fraction(0), Fraction(1), "render48k-f32-stereo-v1")),
        C + A("tone.f32") + ("-af", mix, "-c:a", "pcm_f32le", "-f", "f32le", "pipe:1"),
        C + R(16, 16, "rgba", "alpha.rgba") + ("-frames:v", "1", "-c:v", "png", "-pix_fmt", "rgba", "-f", "image2pipe", "pipe:1"),
        C + ("-i", "alpha.png") + RAW("rgba"),
        C + ("-f", "lavfi", "-i", "color=c=0x102030:s=320x180:r=60:d=1", "-loop", "1", "-framerate", "60", "-i", "alpha.png", "-loop", "1", "-framerate", "60", "-i", brand, "-/filter_complex", "text.ffgraph", "-map", "[out]", "-t", "1") + RAW("rgba"),
        C[:5] + ("verbose",) + C[6:] + ("-init_hw_device", "vulkan=vk:0", "-filter_hw_device", "vk") + R(32, 32, "bgr24", "source.bgr") + ("-vf", V, "-frames:v", "4") + RAW("rgba"),
        D("libx264", "cpu.mp4"), Q("cpu.mp4"), null("cpu.mp4"), D("h264_nvenc", "gpu.mp4"), Q("gpu.mp4"), null("gpu.mp4"))


def assert_float_identity(raw, expected):
    if type(raw) is not bytes or len(raw) != 384000 or raw != expected:
        raise ValueError("float payload bit identity failure")


def partition_probe(raw):
    """Partition by observed stream identity, never option or array order."""
    value = parse_document(raw)
    closed(value, "streams format frames")
    streams = value["streams"]
    if type(streams) is not list or len(streams) != 2:
        raise ValueError("exactly one video and one audio stream required")
    by_type = {}
    for stream in streams:
        kind = stream.get("codec_type")
        if kind not in ("video", "audio") or kind in by_type:
            raise ValueError("undeclared/duplicate stream")
        integer(stream["index"], 0, 255)
        by_type[kind] = stream
    if by_type["video"]["index"] == by_type["audio"]["index"]:
        raise ValueError("distinct observed stream indices required")
    frames = {"video": [], "audio": []}
    if type(value["frames"]) is not list:
        raise ValueError("complete observed frame population required")
    for frame in value["frames"]:
        kind = frame.get("media_type")
        if kind not in frames or frame.get("stream_index") != by_type[kind]["index"]:
            raise ValueError("undeclared frame stream/media identity")
        frames[kind].append(frame)
    return value, by_type, frames


def parse_source_probe(raw, *, path, size, mtime_ns, pair):
    from fractions import Fraction
    from clipperx.render_clock import StreamClock
    value, streams, frames = partition_probe(raw)
    video, audio = streams["video"], streams["audio"]
    if (video.get("codec_name") != "qtrle" or (video.get("width"), video.get("height")) != (32, 32)
            or Fraction(video["avg_frame_rate"]) != 60 or int(video["nb_read_frames"]) != 60
            or audio.get("codec_name") != "pcm_f32le" or int(audio["sample_rate"]) != 48000
            or audio.get("channels") != 2 or audio.get("sample_fmt") != "flt"):
        raise ValueError("source fixture codec/geometry/channel identity")
    clocks = {}
    for kind, stream in streams.items():
        base = Fraction(stream["time_base"])
        start, span = integer(stream["start_pts"]), integer(stream["duration_ts"], 1)
        if base <= 0 or start * base != Fraction(5, 4) or span * base != 1:
            raise ValueError("observed original stream start/span mismatch")
        clocks[kind] = StreamClock(stream["index"], kind, base, start * base, (start + span) * base,
            **(dict(width=32, height=32) if kind == "video" else dict(sample_rate=48000, channels=2, sample_format="flt")))
    if len(frames["video"]) != 60:
        raise ValueError("natural 60 source frames required")
    for i, frame in enumerate(frames["video"]):
        if integer(frame["pts"]) * clocks["video"].time_base != Fraction(5, 4) + Fraction(i, 60):
            raise ValueError("source raw video PTS mismatch")
    count = 0
    for frame in frames["audio"]:
        if integer(frame["pts"]) * clocks["audio"].time_base != Fraction(5, 4) + Fraction(count, 48000):
            raise ValueError("source raw audio PTS discontinuity")
        count += integer(frame["nb_samples"], 1, 65536)
    if count != 48000:
        raise ValueError("natural 48000 source samples required")
    container_start, duration = Fraction(value["format"]["start_time"]), Fraction(value["format"]["duration"])
    if duration <= 0 or abs(container_start) > 1000 or duration > 1000:
        raise ValueError("finite bounded container timing required")
    origin = candidate_origin(path, size, mtime_ns, pair=pair, video=clocks["video"], audio=clocks["audio"])
    return origin, dict(container_start=str(container_start), container_duration=str(duration),
        stream_start="5/4", stream_span="1", stream_end="9/4", video_frames=60, audio_samples=count)


def assert_delivery(raw, mp4):
    from clipperx.render_metrics import parse_frame_observations, validate_output_observations
    value, streams, frames = partition_probe(raw)
    if not 0 < len(mp4) < 8388608:
        raise ValueError("delivery finite direct-file guardrail size")
    video = parse_frame_observations(canonical_json(dict(frames=frames["video"])), audio=False)
    audio = parse_frame_observations(canonical_json(dict(frames=frames["audio"])), audio=True)
    result = validate_output_observations(value, video, audio, expected_frames=60, expected_samples=48000,
        channels=2, actual_size=len(mp4), faststart=mp4_faststart(mp4))
    return result["decoded_observations"]


def assert_s16(raw, source):
    if len(raw) != 192000:
        raise ValueError("stereo s16 sample count failure")
    for (sample,), (original,) in zip(struct.iter_unpack("<h", raw), struct.iter_unpack("<f", source), strict=True):
        expected = max(-32768, min(32767, original * 32768))
        if abs(sample - expected) > 1:
            raise ValueError("s16 clipping/quantization mismatch")


def assert_mix(raw):
    if len(raw) != 384000:
        raise ValueError("mix sample count failure")
    for j, (left, right) in enumerate(struct.iter_unpack("<ff", raw)):
        if not math.isfinite(left) or not math.isfinite(right):
            raise ValueError("nonfinite mix")
        if j < 1024 or j >= 48000 - 1024 or abs(j - 24000) <= 1024:
            continue
        expected = .125 if j < 24000 else .1875
        if abs(abs(left) - expected) > 1e-6 or abs(abs(right) - expected) > 1e-6:
            raise ValueError("mix delay/amplitude mismatch")


def assert_png(raw, expected):
    from io import BytesIO
    from PIL import Image
    if not 0 < len(raw) <= 65536:
        raise ValueError("PNG output cap")
    try:
        with BytesIO(raw) as stream, Image.open(stream) as image:
            if image.format != "PNG" or image.mode != "RGBA" or image.size != (16, 16):
                raise ValueError("PNG format/geometry mismatch")
            image.load()
            if image.tobytes() != expected:
                raise ValueError("independent PNG RGBA mismatch")
    except OSError as exc:
        raise ValueError("independent PNG decode failure") from exc


def mp4_faststart(raw):
    offset, tags = 0, []
    while offset < len(raw):
        if len(tags) >= 256 or len(raw) - offset < 8:
            raise ValueError("bounded complete MP4 top-level boxes required")
        size, tag = struct.unpack_from(">I4s", raw, offset)
        header = 8
        if size == 1:
            if len(raw) - offset < 16:
                raise ValueError("short MP4 extended box")
            size, header = struct.unpack_from(">Q", raw, offset + 8)[0], 16
        elif size == 0:
            size = len(raw) - offset
        if size < header or size > len(raw) - offset:
            raise ValueError("MP4 box bounds")
        tags.append(tag)
        offset += size
    return tags.count(b"moov") == 1 and b"mdat" in tags and tags.index(b"moov") < tags.index(b"mdat")


def assert_video_tags(stderr, time_base):
    from fractions import Fraction
    from clipperx.source_decode import parse_video
    lines = stderr.decode("utf-8", errors="strict").splitlines()
    tags = [parse_video(line, time_base) for line in lines if line.startswith("[showinfo@cxv @") and " n:" in line]
    if len(tags) != 60:
        raise ValueError("complete 60 showinfo tags required")
    for i, tag in enumerate(tags):
        if tag.n != i or tag.pts * tag.time_base != Fraction(5, 4) + Fraction(i, 60) or (tag.width, tag.height) != (32, 32):
            raise ValueError("original video tagged clock/count mismatch")


def assert_vulkan(raw, stderr):
    if (len(raw) != 36864 or not 0 < len(stderr) <= 65536
            or any(len(line) > 16384 for line in stderr.splitlines())):
        raise ValueError("Vulkan payload/selected-device evidence missing")
    text = stderr.decode("utf-8", errors="strict")
    # These are the pinned original hwcontext_vulkan selection/use messages,
    # rather than extension listings, discovery rows or error descriptions.
    prefix = r"(?:\[[^\]\r\n]+\][ \t]+)?"
    selected = re.findall(r"(?m)^" + prefix +
        r"Device 0 selected: ([^\r\n]+) \([^()\r\n]+\) \(0x[0-9a-fA-F]+\)[ \t]*\r?$", text)
    used = re.findall(r"(?m)^" + prefix + r"Using device: ([^\r\n]+?)[ \t]*\r?$", text)
    if (len(selected) != 1 or len(used) != 1 or not selected[0].strip()
            or selected[0] != used[0] or len(used[0]) > 2048):
        raise ValueError("complete consistent Vulkan selected-device identity required")
    means = []
    for n in range(4):
        frame = raw[n * 9216:(n + 1) * 9216]
        pixels = list(zip(*[iter(frame)] * 4))
        if any(pixel[3] != 255 for pixel in pixels):
            raise ValueError("opaque Vulkan raster required")
        blue, green = [p[2] for p in pixels], [p[1] for p in pixels]
        if max(blue) - min(blue) < 20 or max(green) - min(green) < 20:
            raise ValueError("analytic Vulkan ramp span failure")
        if (sum(pixels[y * 48 + 36][2] - pixels[y * 48 + 12][2] for y in range(12, 36)) <= 0
                or sum(pixels[36 * 48 + x][1] - pixels[12 * 48 + x][1] for x in range(12, 36)) <= 0):
            raise ValueError("analytic Vulkan ramp slope failure")
        means.append(sum(pixels[y * 48 + x][0] for y in range(12, 36) for x in range(12, 36)) / 576)
    if abs((means[3] - means[0]) - 3) > 2:
        raise ValueError("analytic Vulkan temporal red ramp failure")
    return dict(selected_device=used[0])


def assert_audio_tags(stderr, *, floating):
    from clipperx.source_decode import parse_audio
    populations = {"cxain": [], "cxaout": []}
    for line in stderr.decode("utf-8", errors="strict").splitlines():
        if line.startswith(("[ashowinfo@cxain @", "[ashowinfo@cxaout @")):
            tag = parse_audio(line)
            populations[tag.stage].append(tag)
    if not populations["cxain"] or len(populations["cxain"]) != len(populations["cxaout"]):
        raise ValueError("paired complete PCM metadata populations required")
    for population, tags in populations.items():
        samples = 0
        for i, tag in enumerate(tags):
            if (tag.n != i or tag.pts != 60000 + samples or tag.rate != 48000 or tag.channels != 2
                    or not 1 <= tag.samples <= 65536 or tag.sample_format not in
                    (("flt", "fltp") if population == "cxain" else ("flt",) if floating else ("s16",))):
                raise ValueError("PCM tagged original clock/format/channel mismatch")
            samples += tag.samples
        if samples != 48000:
            raise ValueError("48000 tagged PCM samples required")
    if any(a.samples != b.samples for a, b in zip(populations["cxain"], populations["cxaout"], strict=True)):
        raise ValueError("unpaired PCM block populations")


def assert_raster(raw, brand_rgba, brand_size):
    if len(raw) != 13824000 or len(brand_rgba) != brand_size[0] * brand_size[1] * 4:
        raise ValueError("complete raster/prepared-brand geometry required")
    frames = [memoryview(raw)[n * 230400:(n + 1) * 230400] for n in (0, 30, 59)]
    def pixel(frame, x, y):
        return tuple(frame[(y * 320 + x) * 4:(y * 320 + x + 1) * 4])
    def within(x, y, roi):
        x0, y0, x1, y1 = roi
        return x0 <= x <= x1 and y0 <= y <= y1
    rois = ((8, 8, 23, 23), (248, 8, 311, 39), (8, 42, 110, 75), (10, 130, 309, 179))
    baseline = pixel(frames[0], 0, 0)
    if baseline[3] != 255:
        raise ValueError("opaque baseline required")
    for y in range(180):
        for x in range(320):
            if any(within(x, y, roi) for roi in rois):
                continue
            if any(pixel(frame, x, y) != baseline for frame in frames):
                raise ValueError("unexpected raster changes outside overlay ROIs")
    def over(source, background):
        alpha = source[3] / 255
        return tuple(source[i] * alpha + background[i] * (1 - alpha) for i in range(3))
    for frame in frames:
        for y in range(8, 24):
            for x in range(8, 24):
                foreground = (240, 40, 80, 0 if x - 8 < 5 else 128 if x - 8 < 11 else 255)
                if any(abs(a - b) > 2 for a, b in zip(pixel(frame, x, y)[:3], over(foreground, baseline), strict=True)):
                    raise ValueError("analytic alpha/source-over mismatch")
    width, height = brand_size
    samples, opaque_changed = 0, False
    for y in range(32):
        sy = (y + .5) * height / 32
        for x in range(64):
            sx = (x + .5) * width / 64
            if min(sx % 1, 1 - sx % 1, sy % 1, 1 - sy % 1) < .1:
                continue
            source = tuple(brand_rgba[(int(sy) * width + int(sx)) * 4:(int(sy) * width + int(sx) + 1) * 4])
            expected = over(source, baseline)
            for frame in frames:
                observed = pixel(frame, 248 + x, 8 + y)
                if any(abs(a - b) > 2 for a, b in zip(observed[:3], expected, strict=True)):
                    raise ValueError("independent nearest-neighbor brand/source-over mismatch")
            samples += 1
            opaque_changed |= source[3] == 255 and pixel(frames[0], 248 + x, 8 + y)[:3] != baseline[:3]
    if samples < 16 or not opaque_changed:
        raise ValueError("insufficient unambiguous opaque brand samples")
    measurements = {"brand_samples": samples}
    for name, roi in zip(("drawtext", "ass"), rois[2:], strict=True):
        changed, unchanged = [], 0
        for y in range(roi[1], roi[3] + 1):
            for x in range(roi[0], roi[2] + 1):
                if pixel(frames[0], x, y) != pixel(frames[2], x, y):
                    raise ValueError("text changed between inactive endpoint frames")
                if pixel(frames[1], x, y) != pixel(frames[0], x, y):
                    changed.append((x, y))
                else:
                    unchanged += 1
        if len(changed) < 20 or not unchanged:
            raise ValueError("active text requires background and >=20 glyph pixels")
        measurements[name] = dict(changed_pixels=len(changed), bbox=[min(p[i] for p in changed) for i in (0, 1)] + [max(p[i] for p in changed) for i in (0, 1)])
    return measurements


class DetachedIO:
    """Sequential, root-free input reads, closed before original owner creation."""

    def check_path(self, value, *, absent=False, directory=False):
        import stat
        lexical = canonical_path(value)
        path = Path(lexical)
        for node in reversed((path, *path.parents)):
            try:
                observed = node.lstat()
            except FileNotFoundError:
                if absent and node == path:
                    continue
                raise ValueError("required existing parent/input unavailable")
            if stat.S_ISLNK(observed.st_mode) or getattr(observed, "st_file_attributes", 0) & 0x400:
                raise ValueError("reparse ancestry refused")
        if absent:
            if path.exists():
                raise ValueError("consumed evidence namespace")
            resolved = path.parent.resolve(strict=True) / path.name
        else:
            resolved = path.resolve(strict=True)
            if directory != path.is_dir():
                raise ValueError("input role/type mismatch")
        if canonical_path(str(resolved)) != lexical:
            raise ValueError("case/short-name/resolution alias refused")
        return path

    def read(self, path, cap):
        actual = self.check_path(path)
        if actual.stat().st_size > cap:
            raise ValueError("input byte cap before read")
        with actual.open("rb", buffering=0) as stream:
            raw = stream.read(cap + 1)
        if len(raw) > cap:
            raise ValueError("input grew beyond cap")
        return raw

    def stream_identity(self, stream):
        import os
        observed = os.fstat(stream.fileno())
        return observed.st_dev, observed.st_ino

    def open_read(self, path):
        return self.check_path(path).open("rb", buffering=0)

    def write_exclusive(self, path, raw):
        """Root-confirmed controller settlement, after the original owner retires."""
        target = self.check_path(path, absent=True)
        with target.open("xb", buffering=0) as stream:
            count = stream.write(raw)
            stream.flush()
            import os
            os.fsync(stream.fileno())
        return count


def publish_terminal(path, payload, *, owner, files, started_ns, clock, io,
        reserved_bytes, charged_bytes, reserved_paths, used_paths):
    """Persistence consumes no process authority and cannot establish retirement.

    The controller first observes the original owner. Persisted bytes explicitly
    remain provisional until the exclusive write's post-return check succeeds.
    """
    from dataclasses import asdict, is_dataclass
    result = dict(status="STOPPED", candidate_component_passed=False, runtime_integrated=False,
        production_accepted=False, selected_font_file_identity="unverified")
    def plain(value):
        if is_dataclass(value):
            return plain(asdict(value))
        if isinstance(value, bytes):
            return dict(hex=value.hex(), bytes=len(value))
        if isinstance(value, dict):
            return {key: plain(child) for key, child in value.items()}
        if isinstance(value, (tuple, list)):
            return [plain(child) for child in value]
        return value
    try:
        canonical_path(path)
        if (not owner.own0 or not files.files_clear or owner.failure is not None
                or payload.get("operations") != 24 or payload.get("attempts") != 24):
            raise ValueError("actual complete operation/retirement observations required")
        integer(reserved_bytes, 1, 2097152)
        integer(charged_bytes, 0, 268435456 - reserved_bytes)
        integer(reserved_paths, 1, 256)
        integer(used_paths, 0, 256 - reserved_paths)
        elapsed = clock() - started_ns
        if not 0 <= elapsed < 1200 * 10**9:
            raise ValueError("terminal campaign deadline before serialization")
        record = dict(schema="clipperx-candidate-components-terminal-v1", provisional=True,
            acceptance="exclusive complete write plus controller post-return checks and later Root review required",
            owner=plain(owner), files=plain(files), payload=payload, elapsed_before_terminal_ns=elapsed,
            runtime_integrated=False, production_accepted=False, selected_font_file_identity="unverified",
            dynamic_runtime_scope="observed component operations only; no OS quota/descendant sandbox/font selection proof")
        raw = canonical_json(record)
        if len(raw) > reserved_bytes:
            raise ValueError("terminal serialization/reservation ceiling")
        if clock() - started_ns >= 1200 * 10**9:
            raise ValueError("terminal campaign deadline after serialization")
        result["status"] = "UNCERTAIN"
        if io.write_exclusive(path, raw) != len(raw):
            raise OSError("short exclusive terminal write")
        elapsed = clock() - started_ns
        if not 0 <= elapsed < 1200 * 10**9:
            raise ValueError("terminal post-return campaign deadline")
        result.update(status="candidate_component_passed", candidate_component_passed=True,
            terminal_bytes=len(raw), terminal_sha256=digest(raw), elapsed_after_terminal_ns=elapsed)
    except (ValueError, OSError, TypeError) as exc:
        result["error"] = str(exc)[:2048]
    return result


class OwnedAttemptSequence:
    """Finite controller policy around the original owner, not a process owner."""

    def __init__(self, job, files, tools, *, started_ns, plan, guard):
        from clipperx._source_host_ownership import SourceHostJob
        from clipperx._retained_files import RetainedFiles
        from clipperx._render_services import OwnedRenderTools
        if (type(job) is not SourceHostJob or type(files) is not RetainedFiles or type(tools) is not OwnedRenderTools
                or files._job is not job or tools._job is not job or tools._files is not files):
            raise ValueError("original same owner/files/tools required")
        if type(plan) is not tuple or len(plan) != 24 or any(type(argv) is not tuple for argv in plan):
            raise ValueError("exact 24-operation immutable argv plan required")
        self.job, self.files, self.tools = job, files, tools
        self.started_ns, self.plan, self.guard = started_ns, plan, guard
        self.work_deadline = min(job._deadline, started_ns + 1185 * 10**9)
        self.attempts = self.completed = self.stderr_bytes = self.sidecar_reservation = 0
        self.active = None
        self.failed = False
        self.seen_controls, self.seen_generations = set(), set()

    def process_factory(self, factory):
        def process(argv):
            if self.active is None or argv != self.active or self.attempts != self.completed or self.attempts >= 24:
                raise ValueError("undeclared/concurrent/extra process attempt")
            if self.job._clock() >= self.work_deadline:
                raise ValueError("campaign work deadline before process attempt")
            self.attempts += 1  # charge immediately before the injected Popen boundary
            return factory(argv)
        return process

    def run(self, index, argv, *, sink=None, inputs=(), outputs=()):
        if (self.failed or self.active is not None or type(index) is not int or index != self.completed + 1
                or not 1 <= index <= 24 or self.attempts != self.completed or argv != self.plan[index - 1]):
            raise ValueError("exact next sequential operation required")
        if self.job._clock() >= self.work_deadline:
            raise ValueError("campaign work deadline; cleanup/settlement reserved")
        row = operation_limits()[index - 1]
        capture = row["stdout_bytes"] if row["kind"] == "probe" and sink is None else 0
        reservation = capture + 65536 + 262144
        if self.sidecar_reservation + reservation > 33554432:
            raise ValueError("aggregate stdout/diagnostic/operation sidecar reservation")
        self.sidecar_reservation += reservation
        self.guard(index, argv, inputs, outputs)
        expected = (0 if row["kind"] == "encoder" or index in (21, 24) else
            row["stdout_bytes"] if row["kind"] in ("spatial", "finalmix") else None)
        self.active = argv
        try:
            result = self.tools.run(argv, kind=row["kind"], timeout=row["timeout_seconds"],
                max_stdout_bytes=row["stdout_bytes"], max_stderr_bytes=65536,
                expected_stdout_bytes=expected, sink=sink, inputs=inputs, outputs=outputs)
            diagnostic = self.job.snapshot().current.diagnostics[-1]
            self.stderr_bytes += result.stderr_bytes
            if (result.argv != argv or result.kind != row["kind"] or result.exit_code != 0
                    or not result.natural_eof or not result.retired or result.stderr_truncated
                    or result.control_id != self.job.snapshot().current.observation_id
                    or result.generation != diagnostic.generation or result.stdout_sha256 != diagnostic.stdout_sha256
                    or result.stdout_bytes != diagnostic.stdout_bytes or self.stderr_bytes > 1572864
                    or result.stderr_bytes > 65536 or len(result.stderr_tail) != result.stderr_bytes):
                raise ValueError("operation association/EOF/retirement/complete-stderr failure")
            if (result.control_id in self.seen_controls or result.generation in self.seen_generations
                    or (result.stdout is not None and
                        (len(result.stdout) != result.stdout_bytes or digest(result.stdout) != result.stdout_sha256))):
                raise ValueError("reused generation/control or incomplete captured bytes")
            if self.attempts != index or self.job._clock() >= self.work_deadline:
                raise ValueError("operation return attempt/deadline mismatch")
            self.seen_controls.add(result.control_id)
            self.seen_generations.add(result.generation)
            self.completed += 1
            return result
        except BaseException:
            self.failed = True
            raise
        finally:
            self.active = None


def _owned_write(files, path, raw, *, role, cap, deadline):
    if type(raw) is not bytes or len(raw) > cap:
        raise ValueError("bounded generated file required")
    files.register(path, role=role, owned=True, max_bytes=cap)
    with files.stage_deadline(deadline):
        with files.open_file(path, "xb") as stream:
            for offset in range(0, len(raw), 65536):
                part = memoryview(raw)[offset:offset + 65536]
                while part:
                    part = part[stream.write(part):]
            stream.flush()
            stream.fsync()


def _owned_read(files, path, cap, *, deadline):
    with files.stage_deadline(deadline):
        with files.open_binary(path) as stream:
            size = stream.fstat().st_size
            if not 0 < size <= cap:
                raise ValueError("owned output size/cap failure")
            chunks = []
            remaining = size
            while remaining:
                raw = stream.read(min(65536, remaining))
                if not raw:
                    raise ValueError("short owned output")
                chunks.append(raw)
                remaining -= len(raw)
            if stream.read(1):
                raise ValueError("growing owned output")
    return b"".join(chunks)


def _stage_candidate_assets(files, root, assets, *, deadline):
    """The production path uses the unchanged original font/brand services."""
    from io import BytesIO
    from PIL import Image
    from clipperx.font_staging import stage_original_fonts, verify_font_stage
    from clipperx.render_assets import RenderAssetRegistry, prepare_brand_asset

    registry = RenderAssetRegistry({"kick_brand": Path(assets["kick_brand"].path)},
        {name: Path(assets[name].path) for name in ("headline_impact", "subtitle_bold")}, {})
    with files.stage_deadline(deadline):
        stage = stage_original_fonts(registry, stage_dir=root / "fonts", _files=files)
        verify_font_stage(stage, _files=files)
        brand = prepare_brand_asset(stage.staged_registry, "kick_brand", cache_dir=root, _files=files)
    admitted_brand = assets["kick_brand"]
    if (brand.source_path != Path(admitted_brand.path)
            or brand.source_sha256 != admitted_brand.sha256.lower()):
        raise ValueError("prepared brand observed source path/hash differs from admitted Pin")
    raw = _owned_read(files, brand.path, 16 * MiB, deadline=deadline)
    with BytesIO(raw) as source, Image.open(source) as image:
        image.load()
        if image.format != "PNG" or image.mode != "RGBA" or image.size != brand.size:
            raise ValueError("prepared brand byte/geometry mismatch")
        rgba = image.tobytes()
    staged = tuple(font.staged_path for font in stage.fonts)
    impact = next(font.staged_path.name for font in stage.fonts if font.asset_id == "headline_impact")
    return brand.path.name, impact, rgba, brand.size, staged + (brand.path,)


def _raster_graph(impact):
    if not re.fullmatch(r"[A-Za-z0-9._-]+\.ttf", impact):
        raise ValueError("staged font basename required")
    return ("[0:v]format=rgba[bg];[1:v]format=rgba[alpha];"
        "[2:v]scale=64:32:flags=neighbor,format=rgba[brand];"
        "[bg][alpha]overlay=x=8:y=8:format=rgb:shortest=1[tmp];"
        "[tmp][brand]overlay=x=248:y=8:format=rgb:shortest=1[b];"
        "[b]ass=filename=text.ass:fontsdir=fonts,"
        f"drawtext=fontfile='fonts/{impact}':text='CLIP':x=8:y=42:fontsize=22:"
        "fontcolor=white:enable='gte(t,0.25)*lt(t,0.75)',format=rgba[out]").encode("utf-8")


def _probe_oracle(index, raw, prior):
    if type(raw) is not bytes or not raw or len(raw) > 1000000:
        raise ValueError("complete bounded probe text required")
    text = raw.decode("utf-8", errors="strict")
    if index in (1, 2):
        name = "ffmpeg" if index == 1 else "ffprobe"
        if len(re.findall(rf"(?m)^{name} version 9\.0\.2(?:[ \t]|$)", text)) != 1:
            raise ValueError("candidate 9.0.2 version required")
        configuration = re.findall(r"(?m)^[ \t]*configuration:[ \t]*([^\r\n]+)$", text)
        if len(configuration) != 1 or not configuration[0].strip():
            raise ValueError("one complete configuration identity required")
        expected = {"libavutil", "libavcodec", "libavformat", "libavdevice", "libavfilter",
            "libswscale", "libswresample"}
        rows = re.findall(r"(?m)^[ \t]*(lib[A-Za-z0-9_]+)[ \t]+"
            r"(\d+\.[ \t]*\d+\.[ \t]*\d+)[ \t]*/[ \t]*"
            r"(\d+\.[ \t]*\d+\.[ \t]*\d+)[ \t]*$", text)
        declared = re.findall(r"(?m)^[ \t]*(lib[A-Za-z0-9_]+)[ \t]+", text)
        if (len(rows) != len(expected) or len(declared) != len(expected)
                or {row[0] for row in rows} != expected or set(declared) != expected):
            raise ValueError("complete unique compiled library version rows required")
        libraries = tuple(sorted((name, re.sub(r"[ \t]", "", compiled), re.sub(r"[ \t]", "", linked))
            for name, compiled, linked in rows))
        if any(compiled != linked for _, compiled, linked in libraries):
            raise ValueError("compiled/linked library version mismatch")
        identity = (libraries, configuration[0].strip())
        if index == 2 and identity != prior:
            raise ValueError("FFmpeg/FFprobe library/configuration identity mismatch")
        return identity
    if index == 3:
        required = ("--enable-gpl", "--enable-libx264", "--enable-libplacebo", "--enable-libass",
            "--enable-libfreetype", "--enable-libharfbuzz", "--enable-libfribidi", "--enable-libdav1d")
        if any(flag not in text for flag in required) or "--disable-everything" in text:
            raise ValueError("reviewed static build configuration missing")
    elif index == 4:
        required = ("adelay aeval aformat alimiter amix anull apad aresample asetpts asettb "
            "ashowinfo asplit ass atrim boxblur color concat crop format fps "
            "libplacebo loudnorm null overlay scale setpts setsar settb showinfo "
            "split trim volume drawtext geq drawbox hstack select testsrc testsrc2 nullsrc").split()
        if not set(required).issubset(set(re.findall(r"(?m)^\s*[A-Z.]{3}\s+(\w+)\s", text))):
            raise ValueError("required filter union incomplete")
    elif index == 5:
        entries = {name: (flags, description) for flags, name, description in
            re.findall(r"(?m)^\s*([A-Z.]{6})\s+(\w+)\s+([^\r\n]+)", text)}
        required = set("h264 hevc vp8 vp9 mpeg1video mpeg2video png qtrle pcm_f32le pcm_s16le aac opus vorbis mp3 flac av1".split())
        if (not required.issubset(entries) or any(not entries[name][0].startswith("D") for name in required)
                or not re.search(r"\bdecoders:\s+[^)]*\blibdav1d\b", entries.get("av1", ("", ""))[1])):
            raise ValueError("broad CPU codec registration missing")
    elif index == 6:
        rows = re.findall(r"(?m)^[ \t]*([DE. ]{2})[ \t]+([a-z0-9_,]+)(?=[ \t]|$)", text)
        groups = [set(joined.split(",")) for flags, joined in rows if "D" in flags or "E" in flags]
        names = set().union(*groups) if groups else set()
        if (not set("mov mp4 matroska webm mpegts flv avi rawvideo f32le s16le wav null image2pipe".split()).issubset(names)
                or not any({"mov", "mp4"}.issubset(group) for group in groups)
                or not any({"matroska", "webm"}.issubset(group) for group in groups)):
            raise ValueError("required mux/demux families missing")
    elif index == 7:
        if not set("h264_nvenc libx264 aac qtrle pcm_f32le pcm_s16le png".split()).issubset(set(re.findall(r"(?m)^\s*[A-Z.]{6}\s+(\w+)\s", text))):
            raise ValueError("delivery/fixture encoders missing")
    elif index == 8:
        from clipperx.render_runtime import NATIVE_OPTIONS
        options = set(re.findall(r"(?m)^\s{1,8}([A-Za-z][A-Za-z0-9_]*)\s+<[^>\r\n]+>", text))
        if not set(NATIVE_OPTIONS).issubset(options):
            raise ValueError("libplacebo native policy option missing")
    return None


def _default_process(argv, root):
    import subprocess
    return subprocess.Popen(list(argv), stdin=subprocess.DEVNULL, stdout=subprocess.PIPE,
        stderr=subprocess.PIPE, bufsize=0, text=False, shell=False, close_fds=True, cwd=root)


def _new_owner(driver, *, deadline, clock, process):
    from clipperx._source_host_ownership import SourceHostJob
    return SourceHostJob(driver, deadline_ns=deadline, clock=clock, process_factory=process)


def _new_files(job, root, clock):
    from clipperx._retained_files import RetainedFiles, _path_action
    def namespace_action(operation, path, destination):
        # A successful exclusive mkdir consumes this namespace even if no leaf
        # could be opened. Let original retirement record the refused cleanup.
        if operation == "rmdir" and path == root:
            raise OSError("consumed evidence namespace cannot be removed")
        return _path_action(operation, path, destination)
    return RetainedFiles(job, root=root.parent, clock=clock, max_paths=256,
        max_live=16, max_generations=4096, _path_action=namespace_action)


class ComponentCampaign:
    """One original-owner candidate-only campaign from an immutable v2 admission."""

    def __init__(self, inputs, cli_pin, io, clock):
        self.inputs, self.cli_pin, self.io, self.clock = inputs, cli_pin, io, clock

    def execute(self):
        import threading
        from clipperx._render_services import OwnedRenderTools

        inputs, clock, io = self.inputs, self.clock, self.io
        guard = guard_bytes(inputs, self.cli_pin)  # all ten fields before a namespace action
        root = Path(inputs.evidence_root)
        deadline = inputs.started_ns + 1185 * 10**9
        state = dict(error=None, attempts=0, operations=0, records=[], keep=[])
        holder = {}

        def driver_work():
            job, files, tools = holder["job"], holder["files"], holder["tools"]
            keep = state["keep"]
            seq = None
            try:
                job.register_current_driver()
                files.register(root, role="directory", owned=True, max_bytes=0)
                with files.stage_deadline(deadline):
                    files.path_action("mkdir", root)
                _owned_write(files, root / "guard.json", guard, role="sidecar", cap=16384, deadline=deadline)
                keep.append(root / "guard.json")
                pins = (inputs.admission, inputs.runtime_acceptance, inputs.publication,
                    inputs.pair.ffmpeg, inputs.pair.ffprobe)
                for pin in pins:
                    files.register(Path(pin.path), role="runtime" if pin in (inputs.pair.ffmpeg, inputs.pair.ffprobe)
                        else "original", owned=False, max_bytes=pin.bytes)
                for _, pin in inputs.assets:
                    files.register(Path(pin.path), role="original", owned=False, max_bytes=pin.bytes)
                fixtures = fixture_bytes()
                for name, raw in fixtures.items():
                    role = "ass" if name.endswith(".ass") else "raw" if name.endswith((".bgr", ".rgba")) else "pcm" if name.endswith(".f32") else "graph"
                    cap = 16384 if role == "ass" else 4096 if name.endswith(".list") else len(raw)
                    _owned_write(files, root / name, raw, role=role, cap=cap, deadline=deadline)
                    if role in ("ass", "graph"):
                        keep.append(root / name)
                brand, impact, brand_rgba, brand_size, asset_keep = _stage_candidate_assets(
                    files, root, dict(inputs.assets), deadline=deadline)
                keep.extend(asset_keep)
                graph = _raster_graph(impact)
                _owned_write(files, root / "text.ffgraph", graph, role="graph", cap=65536, deadline=deadline)
                keep.append(root / "text.ffgraph")
                origin = candidate_origin(str(root / "source.mov"), 1, 1, pair=inputs.pair)
                plan = operation_argv(inputs.pair, origin=origin, brand=brand, impact=impact)

                def recheck(index, argv, child_inputs, outputs):
                    if argv != plan[index - 1] or any(path not in files._paths for path in (*child_inputs, *outputs)):
                        raise ValueError("unplanned argv or undeclared retained path")
                    for pin in pins:
                        path = Path(pin.path)
                        with files.stage_deadline(deadline):
                            if files.path_action("resolve", path) != path:
                                raise ValueError("pin path drift")
                            with files.open_binary(path) as stream:
                                stat = stream.fstat()
                                if stat.st_size != pin.bytes:
                                    raise ValueError("between-child pin size drift")
                                hasher, count = hashlib.sha256(), 0
                                while count < pin.bytes:
                                    raw = stream.read(min(65536, pin.bytes - count))
                                    if not raw:
                                        raise ValueError("short pinned read")
                                    hasher.update(raw)
                                    count += len(raw)
                                if stream.read(1) or hasher.hexdigest() != pin.sha256.lower():
                                    raise ValueError("between-child pin hash/growth drift")
                    with files.stage_deadline(deadline):
                        disk = files.path_action("disk_usage", root)
                    reserved = sum(p.max_bytes for p in files.snapshot().paths if p.owned) + 2097152
                    if (reserved > 268435456 or disk[2] < 1073741824 + 268435456 - reserved
                            or len(files.snapshot().paths) + 1 > 256):
                        raise ValueError("owned reservation/path/free-space limit")

                seq = OwnedAttemptSequence(job, files, tools, started_ns=inputs.started_ns, plan=plan, guard=recheck)
                job._processes = seq.process_factory(lambda argv: _default_process(argv, root))
                names = {9: "source.mov", 11: "video.bgr", 12: "audio.s16", 13: "audio.f32",
                    14: "mix.f32", 15: "alpha.png", 16: "decoded.rgba", 17: "raster.rgba",
                    18: "vulkan.rgba", 19: "cpu.mp4", 22: "gpu.mp4"}
                source_bytes = fixtures["source.bgr"]
                source_audio = fixtures["source.f32"]
                version_libraries = None
                observed_origin = None
                for index, argv in enumerate(plan, 1):
                    row = operation_limits()[index - 1]
                    if index == 17:
                        review_reservation = 3 * 262144
                        if seq.sidecar_reservation + review_reservation > 33554432:
                            raise ValueError("raster review aggregate sidecar reservation")
                        seq.sidecar_reservation += review_reservation
                        for frame in (0, 30, 59):
                            files.register(root / f"raster-review-{frame:02}.png", role="asset",
                                owned=True, max_bytes=262144)
                    if index == 18:
                        # This log uses this operation's existing 65536-byte
                        # stderr sidecar reservation, not an additional probe.
                        files.register(root / "vulkan-device.log", role="sidecar",
                            owned=True, max_bytes=65536)
                    out = root / names[index] if index in names else None
                    if out is not None:
                        role = "candidate" if index in (9, 19, 22) else "asset" if index == 15 else "raw" if index in (11, 16, 17, 18) else "pcm"
                        files.register(out, role=role, owned=True, max_bytes=row["artifact_bytes"])
                    declared = {9: ("source.bgr", "source.f32"), 10: ("source.mov",),
                        11: ("source.mov",), 12: ("source.mov",), 13: ("source.mov",),
                        14: ("tone.f32",), 15: ("alpha.rgba",), 16: ("alpha.png",),
                        17: ("alpha.png", brand, "text.ffgraph", "text.ass", *[str(path.relative_to(root)) for path in asset_keep if path.suffix == ".ttf"]),
                        18: ("source.bgr",), 19: ("raster.rgba", "shots.list", "tone.f32"),
                        20: ("cpu.mp4",), 21: ("cpu.mp4",), 22: ("raster.rgba", "shots.list", "tone.f32"),
                        23: ("gpu.mp4",), 24: ("gpu.mp4",)}
                    child_inputs = tuple(dict.fromkeys((Path(argv[0]), *(root / name for name in declared.get(index, ())))))
                    sink = None
                    if out is not None and index not in (9, 19, 22):
                        with files.stage_deadline(deadline):
                            sink = files.open_file(out, "xb")
                            sink.__enter__()
                    try:
                        result = seq.run(index, argv, sink=sink, inputs=child_inputs,
                            outputs=(out,) if index in (9, 19, 22) else ())
                    finally:
                        if sink is not None:
                            with files.stage_deadline(deadline):
                                sink.__exit__(None, None, None)
                    state["attempts"] = seq.attempts
                    if result.stdout:
                        capture_path = root / f"capture-{index:02}.bin"
                        _owned_write(files, capture_path, result.stdout, role="sidecar",
                            cap=row["stdout_bytes"], deadline=deadline)
                        keep.append(capture_path)
                    observation = None
                    if index <= 8:
                        observation = _probe_oracle(index, result.stdout, version_libraries)
                        if index == 1:
                            version_libraries = observation
                    elif index == 10:
                        source_mov = root / "source.mov"
                        with files.stage_deadline(deadline):
                            stamp = files.path_action("stat", source_mov)
                        observed_origin, observation = parse_source_probe(result.stdout,
                            path=str(source_mov), size=stamp.st_size, mtime_ns=stamp.st_mtime_ns, pair=inputs.pair)
                        if plan != operation_argv(inputs.pair, origin=observed_origin, brand=brand, impact=impact):
                            raise ValueError("observed candidate origin changed fixed argv plan")
                    elif index in (11, 12, 13, 14, 15, 16, 17, 18):
                        raw = _owned_read(files, out, row["artifact_bytes"], deadline=deadline)
                        if digest(raw) != result.stdout_sha256 or len(raw) != result.stdout_bytes:
                            raise ValueError("owned sink versus child stdout identity mismatch")
                        if index == 11:
                            if raw != source_bytes or observed_origin is None:
                                raise ValueError("whole analytic BGR ramp mismatch")
                            assert_video_tags(result.stderr_tail, observed_origin.streams[0].time_base)
                        elif index == 12:
                            assert_s16(raw, source_audio)
                            assert_audio_tags(result.stderr_tail, floating=False)
                        elif index == 13:
                            assert_float_identity(raw, source_audio)
                            assert_audio_tags(result.stderr_tail, floating=True)
                        elif index == 14:
                            assert_mix(raw)
                        elif index == 15:
                            assert_png(raw, fixtures["alpha.rgba"])
                        elif index == 16:
                            if raw != fixtures["alpha.rgba"]:
                                raise ValueError("candidate PNG decode mismatch")
                        elif index == 17:
                            observation = assert_raster(raw, brand_rgba, brand_size)
                            from io import BytesIO
                            from PIL import Image
                            observation["review_frames"] = []
                            for frame in (0, 30, 59):
                                pixels = raw[frame * 230400:(frame + 1) * 230400]
                                with BytesIO() as stream:
                                    Image.frombytes("RGBA", (320, 180), pixels).save(stream, format="PNG")
                                    png = stream.getvalue()
                                path = root / f"raster-review-{frame:02}.png"
                                _owned_write(files, path, png, role="asset", cap=262144, deadline=deadline)
                                keep.append(path)
                                observation["review_frames"].append(dict(frame=frame, path=str(path),
                                    sha256=digest(png), bytes=len(png), width=320, height=180))
                        elif index == 18:
                            log = result.stderr_tail
                            if digest(log) != result.stderr_sha256:
                                raise ValueError("complete Vulkan device log identity mismatch")
                            log_path = root / "vulkan-device.log"
                            _owned_write(files, log_path, log, role="sidecar", cap=65536, deadline=deadline)
                            keep.append(log_path)
                            observation = assert_vulkan(raw, log)
                            observation["device_log"] = dict(path=str(log_path), sha256=digest(log), bytes=len(log))
                    elif index in (20, 23):
                        mp4 = _owned_read(files, root / ("cpu.mp4" if index == 20 else "gpu.mp4"),
                            16 * MiB, deadline=deadline)
                        observation = assert_delivery(result.stdout, mp4)
                    elif index in (9, 19, 22):
                        with files.stage_deadline(deadline):
                            stamp = files.path_action("stat", out)
                        if not 0 < stamp.st_size < 8388608:
                            raise ValueError("direct MOV/MP4 output absent/oversize/truncated")
                    record = dict(index=index, kind=result.kind, argv=argv, control_id=result.control_id,
                        generation=result.generation, stdout_bytes=result.stdout_bytes,
                        stdout_sha256=result.stdout_sha256, stderr_bytes=result.stderr_bytes,
                        stderr_sha256=result.stderr_sha256, natural_eof=result.natural_eof,
                        exit_code=result.exit_code, oracle=observation)
                    raw_record = canonical_json(record)
                    if len(raw_record) > 262144:
                        raise ValueError("operation record byte cap")
                    record_path = root / f"operation-{index:02}.json"
                    _owned_write(files, record_path, raw_record, role="sidecar", cap=262144, deadline=deadline)
                    keep.append(record_path)
                    if index in (9, 15, 19, 22):
                        keep.append(out)
                    state["records"].append(dict(index=index, sha256=digest(raw_record), bytes=len(raw_record)))
                    state["operations"] = index
                state["attempts"] = seq.attempts
                if seq.completed != 24 or seq.attempts != 24:
                    raise ValueError("complete 24 original operation settlement required")
                recheck(24, plan[23], (), ())
            except BaseException as exc:
                if seq is not None:
                    state["attempts"] = seq.attempts
                state["error"] = str(exc)[:2048]
                if isinstance(exc, KeyboardInterrupt):
                    job.request_stop("cancelled")
                elif job.snapshot().failure is None:
                    job.fail("validation", "driver")
            finally:
                try:
                    files.retire(retain=tuple(keep))
                finally:
                    job.driver_done()

        driver = threading.Thread(target=driver_work, name="candidate-components-driver", daemon=False)
        job = _new_owner(driver, deadline=deadline, clock=clock,
            process=lambda argv: _default_process(argv, root))
        files = _new_files(job, root, clock)
        tools = OwnedRenderTools(job, files, clock=clock)
        holder.update(job=job, files=files, tools=tools)
        from clipperx._source_host_ownership import HostedObservationFailure
        try:
            job.start_driver()
        except HostedObservationFailure:
            # A synchronous injected driver can finish with an owned failure
            # before start_driver returns. The original owner still settles it.
            pass
        except KeyboardInterrupt:
            state["error"] = "controller cancelled"
            job.request_stop("cancelled")
        while True:
            try:
                owner = job.service_cleanup()
                if owner.own0:
                    break
                if owner.failure is not None and clock() >= owner.failure.cleanup_deadline_ns:
                    return dict(status="STOPPED", error=state["error"] or "cleanup custody uncertain",
                        attempts=state["attempts"], operations=state["operations"],
                        candidate_component_passed=False, runtime_integrated=False, production_accepted=False)
                job._driver.join(.01)
            except KeyboardInterrupt:
                state["error"] = "controller cancelled"
                job.request_stop("cancelled")
        final_files = files.snapshot()
        if (state["error"] is not None or state["attempts"] != state["operations"]
                or state["operations"] != 24 or owner.failure is not None or not final_files.files_clear):
            return dict(status="STOPPED", error=state["error"] or "incomplete owner/file settlement",
                attempts=state["attempts"], operations=state["operations"],
                candidate_component_passed=False, runtime_integrated=False, production_accepted=False)
        payload = dict(operations=24, attempts=24, records=state["records"],
            pair=dict(ffmpeg=inputs.pair.ffmpeg.record(), ffprobe=inputs.pair.ffprobe.record()),
            admission=inputs.admission.record(), publication=inputs.publication.record())
        reserved = sum(path.max_bytes for path in final_files.paths if path.owned)
        result = publish_terminal(str(root / "terminal.json"), payload, owner=owner, files=final_files,
            started_ns=inputs.started_ns, clock=clock, io=io, reserved_bytes=2097152,
            charged_bytes=reserved, reserved_paths=1, used_paths=len(final_files.paths))
        result.update(attempts=24, operations=24)
        return result


def admit(cli_pin, *, io=None, clock=None):
    inputs = validate_inputs(cli_pin, io=io, clock=clock)
    import time
    return ComponentCampaign(inputs, cli_pin, DetachedIO() if io is None else io,
        time.monotonic_ns if clock is None else clock)


def main(argv=None):
    import sys
    arguments = list(sys.argv[1:] if argv is None else argv)
    if not arguments:
        print("NOT_RUN")
        return 0
    import argparse
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument("--native", action="store_true")
    parser.add_argument("--confirm-host", action="store_true")
    parser.add_argument("--admission")
    parser.add_argument("--admission-sha256")
    parser.add_argument("--admission-bytes", type=int)
    try:
        args = parser.parse_args(arguments)
        if not args.native or not args.confirm_host or not all((args.admission, args.admission_sha256, args.admission_bytes)):
            raise ValueError("complete Root activation operands required")
        pin = Pin.parse(dict(path=args.admission, sha256=args.admission_sha256, bytes=args.admission_bytes))
        result = admit(pin).execute()
        print(result["status"])
        return 0 if result["candidate_component_passed"] else 2
    except (ValueError, OSError) as exc:
        print(f"NOT_ADMITTED: {exc}")
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
