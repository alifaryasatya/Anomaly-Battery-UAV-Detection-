"""Run or resume the final audit without modifying historical experiment evidence.

Independent phases may run in separate processes. No source engine/config edit,
model fitting or candidate selection is performed by this audit script.
"""

import argparse
import contextlib
import csv
import io
import json
from pathlib import Path
import runpy
import subprocess
import sys
from time import perf_counter
from unittest.mock import patch

try:
    from scripts.final_audit_support import (FinalAuditError, capture_preservation, core_bundle_audit, gate_rows,
        load, render_reports, require, sha, table, test_totals, verify_manifest, verify_preservation, write_json)
except ModuleNotFoundError:
    from final_audit_support import (FinalAuditError, capture_preservation, core_bundle_audit, gate_rows,
        load, render_reports, require, sha, table, test_totals, verify_manifest, verify_preservation, write_json)

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "artifacts/sprint13"
AUDITS = (
    "verify_sprint5.py", "verify_sprint6.py", "verify_sprint7_bundles.py",
    "verify_sprint8.py", "verify_sprint8_bundles.py", "verify_sprint9.py", "verify_sprint9_bundles.py",
    "verify_sprint10.py", "verify_sprint11.py", "verify_sprint12.py",
)


def program_files():
    return [ROOT / "main.py", *sorted((ROOT / "src").glob("*.py")),
            *sorted((ROOT / "scripts").glob("*.py")), *sorted((ROOT / "tests").glob("*.py"))]


def prepare():
    OUT.mkdir(parents=True, exist_ok=True)
    require((ROOT / "artifacts/sprint12/command_status.json").exists(), "Complete Sprint 12 before the final audit.")
    require(load(ROOT / "artifacts/sprint12/command_status.json")["status"] == "SPRINT_12_COMPLETE", "Sprint 12 is incomplete.")
    snapshot = capture_preservation(ROOT, OUT)
    write_json(OUT / "before_snapshot.json", snapshot)
    code = {p.relative_to(ROOT).as_posix(): sha(p) for p in program_files()}
    write_json(OUT / "pre_execution_manifest.json", code)
    plan = dict(stage=13, scope="final_software_and_research_evidence_audit", no_model_or_policy_selection=True,
                commands=["run_sprint0.py", "pytest full suite", "pip check", "research-evaluation", "replay", "repeat CLI cache checks"],
                independent_audits=list(AUDITS), saved_bundle_audit="103 original bundles, original fit identities/scaler/quantiles/score replay",
                report_rule="derive every metric from verified source CSV/JSON; research FAIL and limitations remain visible",
                latency_rule="verify/reuse Sprint12 original recorded host measurements; do not remeasure or imply hardware timing",
                preserve_scope="all completed pre-audit run manifests, Sprint0 freeze and Sprint7-12 evidence",
                required_outputs=["FINAL_RESEARCH_RESULTS.md", "FINAL_VERIFICATION.md"], software_success_is_not_promotion=True)
    write_json(OUT / "audit_plan.json", plan)
    print(json.dumps(dict(status="PREPARED", preserved_files=len(snapshot["files"]), prior_manifests=snapshot["prior_manifests"])))


def check_code():
    require((OUT / "pre_execution_manifest.json").exists(), "Run --phase prepare first.")
    for name, digest in load(OUT / "pre_execution_manifest.json").items():
        require(sha(ROOT / name) == digest, f"Program changed after final audit preparation: {name}")


def command(name, arguments, *, json_output=False, cwd=None):
    start = perf_counter()
    directory = ROOT.parent if cwd is None else cwd
    result = subprocess.run([sys.executable, *arguments], cwd=directory, capture_output=True, encoding="utf-8")
    (OUT / (name + ".stdout.txt")).write_text(result.stdout, encoding="utf-8")
    (OUT / (name + ".stderr.txt")).write_text(result.stderr, encoding="utf-8")
    require(result.returncode == 0, f"Final audit command {name} failed: {result.stderr[-1800:]}")
    parsed = json.loads(result.stdout) if json_output else None
    return dict(status="PASS", returncode=result.returncode, elapsed_seconds=perf_counter() - start,
                argv=[str(sys.executable), *arguments], cwd=str(directory), result=parsed)


def cache_state(directory):
    return {p: (sha(p), p.stat().st_mtime_ns) for p in directory.rglob("*") if p.is_file()}


def run_commands():
    check_code()
    records = {}
    records["sprint0"] = command("sprint0", [str(ROOT / "scripts/run_sprint0.py")], json_output=True)
    records["dependencies"] = command("pip_check", ["-m", "pip", "check"])
    records["research"] = command("research_cli", [str(ROOT / "main.py"), "research-evaluation"], json_output=True)
    research = records["research"]["result"]
    require(not research["training_stages_built_at_creation"] and not research["source_stages_built_at_creation"], "Final consolidation unexpectedly built/refit an experiment.")
    write_json(OUT / "research_result.json", research)
    old_replay = load(ROOT / "artifacts/sprint12/replay_result.json")
    replay_dir = Path(old_replay["output_directory"])
    original_timing_state = cache_state(replay_dir)
    records["replay"] = command("replay_cli", [str(ROOT / "main.py"), "replay"], json_output=True)
    replay = records["replay"]["result"]
    require(replay == old_replay, "Final audit must reuse the frozen replay measurements.")
    require(cache_state(replay_dir) == original_timing_state, "Replay remeasured or changed completed timing outputs.")
    write_json(OUT / "replay_result.json", replay)
    research_dir = Path(research["output_directory"])
    before = cache_state(research_dir)
    records["research_cache"] = command("research_cache_cli", [str(ROOT / "main.py"), "research-evaluation"], json_output=True)
    require(records["research_cache"]["result"] == research and cache_state(research_dir) == before, "Research cache changed content/mtime.")
    records["replay_cache"] = command("replay_cache_cli", [str(ROOT / "main.py"), "replay"], json_output=True)
    require(records["replay_cache"]["result"] == replay and cache_state(replay_dir) == original_timing_state, "Replay cache changed measurements.")
    records["tests"] = command("pytest", ["-m", "pytest", str(ROOT / "tests"), "-q", "--junitxml=" + str(OUT / "tests.xml")], cwd=ROOT)
    totals = test_totals(OUT / "tests.xml")
    check_code()
    write_json(OUT / "commands_verification.json", dict(status="PASS", tests=totals, commands=records,
                                                      model_training_by_final_audit=False, replay_latency_remeasured=False))
    print(json.dumps(dict(status="COMMANDS_PASS", tests=totals, research_run=research["run_id"], replay_run=replay["run_id"])))


def run_readonly_audit(script, *, pointer_overrides=None):
    """Capture legacy verifier writes in memory; historical evidence stays intact."""
    captured, output = [], io.StringIO()
    read_text = Path.read_text
    overrides = pointer_overrides or {}

    def redirected_read(path, *args, **kwargs):
        target = overrides.get(path.resolve(), path)
        return read_text(target, *args, **kwargs)

    def capture_write(path, value, *args, **kwargs):
        require(path.resolve().is_relative_to(ROOT / "artifacts") and path.suffix == ".json",
                f"Independent verifier attempted unexpected write: {path}")
        result = json.loads(value)
        require(result.get("status") == "PASS", f"Independent audit did not pass: {script}")
        captured.append(result)
        return len(value)

    # Audit code is loaded without its __main__, preserving its exact logic.
    with contextlib.redirect_stdout(output), patch.object(Path, "read_text", redirected_read), patch.object(Path, "write_text", capture_write):
        namespace = runpy.run_path(str(ROOT / "scripts" / script), run_name="sprint13_readonly_audit")
        if "verify" in namespace:
            result = namespace["verify"]()
        else:
            namespace["main"]()
            require(len(captured) == 1, f"Verifier {script} did not emit exactly one result.")
            result = captured[0]
    require(result.get("status") == "PASS", f"Independent audit failed: {script}")
    return result, output.getvalue()


def run_audits():
    check_code()
    results = {}
    for script in AUDITS:
        # New research consolidation has the same frozen source set. Audit it
        # using the original verifier without replacing the Sprint11 pointer.
        overrides = {}
        if script == "verify_sprint11.py":
            require((OUT / "research_result.json").exists(), "Finish the research CLI before its final independent audit.")
            overrides[(ROOT / "artifacts/sprint11/research_result.json").resolve()] = OUT / "research_result.json"
        result, text = run_readonly_audit(script, pointer_overrides=overrides)
        name = Path(script).stem
        write_json(OUT / "audits" / (name + ".json"), result)
        (OUT / "audits" / (name + ".stdout.txt")).write_text(text, encoding="utf-8")
        results[name] = result
        print(json.dumps(dict(audit=name, status="PASS")), flush=True)
    results["core_bundles"] = core_bundle_audit(ROOT)
    write_json(OUT / "audits/core_bundles.json", results["core_bundles"])
    check_code()
    write_json(OUT / "audits_verification.json", dict(status="PASS", audits=results, historical_evidence_written=False))


def finalize():
    check_code()
    commands = load(OUT / "commands_verification.json")
    audits = load(OUT / "audits_verification.json")
    require(commands["status"] == audits["status"] == "PASS", "Required final audit phases have not passed.")
    snapshot = load(OUT / "before_snapshot.json")
    verify_preservation(ROOT, snapshot)
    rs, replay = load(OUT / "research_result.json"), load(OUT / "replay_result.json")
    directory = Path(rs["output_directory"])
    verify_manifest(directory, ROOT)
    verify_manifest(Path(replay["output_directory"]), ROOT)
    research = load(directory / "results.json")
    old_dir = Path(load(ROOT / "artifacts/sprint11/research_result.json")["output_directory"])
    require(load(directory / "source_inventory.json") == load(old_dir / "source_inventory.json"), "Consolidated source set drifted from the frozen research inputs.")
    protocol = load(Path(replay["output_directory"]) / "replay_protocol.json")
    source_map = {r["command"]: r for r in research["source_runs"]}
    require(protocol["baseline_run_id"] == source_map["cross-validate"]["run_id"]
            and protocol["baseline_manifest_sha256"] == source_map["cross-validate"]["manifest_sha256"]
            and protocol["low_load_run_id"] == source_map["evaluate-low-load"]["run_id"]
            and protocol["low_load_manifest_sha256"] == source_map["evaluate-low-load"]["manifest_sha256"],
            "Research and replay lineage do not match.")
    require(replay["healthy_active"] == rs["baseline_healthy_active"] and replay["synthetic"] == rs["baseline_synthetic"], "Research/replay outcomes differ.")
    require(rs["selected_new_winner"] is None and not rs["operational_config_changed"] and not replay["threshold_changed"], "Final audit cannot select a new operational policy.")
    results = audits["audits"]
    bundle_count = sum(results[k]["bundles_verified"] for k in ("verify_sprint7_bundles", "verify_sprint8_bundles", "verify_sprint9_bundles", "core_bundles"))
    require(bundle_count == rs["model_bundle_references"], "Saved bundle audit coverage is incomplete.")
    data = dict(research_summary=rs, research=research, replay=replay,
                measurement_context=load(ROOT / "artifacts/sprint12/measurement_context.json"))
    technical = dict(raw_config_artifact_integrity=True, full_unit_integration_suite=True, CLI_research_and_replay=True,
                     predictor_split_scaler_threshold_provenance=True, synthetic_masks_and_observability=True,
                     validation_only_selection_and_pre_outer_freeze=True, readiness_and_metric_denominators=True,
                     independent_batch_stream_equivalence=True, reproducibility_and_no_remeasurement=True)
    gates = gate_rows(data, technical)
    verification = dict(software_audit_status="PASS", tests=commands["tests"], audits=results, bundles_verified=bundle_count,
                        preserved_files=len(snapshot["files"]), preserved_manifests=snapshot["prior_manifests"],
                        new_research_run_id=rs["run_id"], frozen_replay_run_id=replay["run_id"],
                        original_weights_thresholds_config_preserved=True, historical_evidence_written=False,
                        replay_latency_remeasured=False, research_gate_status=research["promotion_decision"]["status"],
                        deployment_authorized=False)
    write_json(OUT / "final_results.json", data)
    write_json(OUT / "verification.json", verification)
    with (OUT / "gate_checks.csv").open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=["check", "status", "scope", "detail"])
        writer.writeheader()
        writer.writerows(gates)
    final_results, final_verification = render_reports(data, gates, verification)
    (ROOT / "FINAL_RESEARCH_RESULTS.md").write_text(final_results, encoding="utf-8")
    (ROOT / "FINAL_VERIFICATION.md").write_text(final_verification, encoding="utf-8")
    verify_preservation(ROOT, snapshot)
    write_json(OUT / "preservation_verification.json", dict(status="PASS", files_checked=len(snapshot["files"]),
        prior_completed_manifests=snapshot["prior_manifests"], original_sprint0_frozen_files=snapshot["sprint0_frozen_files"],
        raw_config_protocol_weights_thresholds_prior_artifacts_evidence_unchanged=True))
    write_json(OUT / "document_manifest.json", {n: sha(ROOT / n) for n in ("FINAL_RESEARCH_RESULTS.md", "FINAL_VERIFICATION.md")})
    write_json(OUT / "source_manifest.json", {p.relative_to(ROOT).as_posix(): sha(p) for p in program_files()})
    write_json(OUT / "command_status.json", dict(status="SPRINT_13_COMPLETE", software_audit_status="PASS", tests=commands["tests"],
        independent_audits=len(results), model_bundles_reviewed=bundle_count, research_gate_status=research["promotion_decision"]["status"],
        progressive_target_status=research["baseline"]["research_target_checks"]["gradual_observable"]["status"],
        deployment_status="RESEARCH_AND_BASELINE_SOFTWARE_VALIDATION_ONLY", hardware_ready=False,
        final_documents=["FINAL_RESEARCH_RESULTS.md", "FINAL_VERIFICATION.md"]))
    evidence = {p.relative_to(ROOT).as_posix(): sha(p) for p in sorted(OUT.rglob("*")) if p.is_file() and p.name != "evidence_manifest.json"}
    evidence.update({n: sha(ROOT / n) for n in ("FINAL_RESEARCH_RESULTS.md", "FINAL_VERIFICATION.md")})
    write_json(OUT / "evidence_manifest.json", evidence)
    print(json.dumps(load(OUT / "command_status.json"), indent=2))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--phase", choices=("prepare", "commands", "audits", "finalize", "all"), default="all")
    phase = parser.parse_args().phase
    actions = dict(prepare=prepare, commands=run_commands, audits=run_audits, finalize=finalize)
    try:
        if phase == "all":
            for action in actions.values():
                action()
        else:
            actions[phase]()
    except Exception as exc:
        write_json(OUT / "command_error.json", dict(status="ERROR", phase=phase, type=type(exc).__name__, error=str(exc)))
        raise


if __name__ == "__main__":
    main()
