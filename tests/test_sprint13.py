"""Final audit must preserve failed gates, reject corrupt evidence, count real tests."""

from pathlib import Path

import pytest

from scripts.final_audit_support import (FinalAuditError, gate_rows, safe_path, sha, test_totals as read_test_totals,
                                        verify_manifest, verify_preservation, write_json)
from scripts.run_sprint13 import run_readonly_audit


def test_research_failures_stay_failed_when_software_passes():
    data = dict(research=dict(baseline=dict(research_target_checks={
        "abrupt_severe": dict(status="MET", rate=1., target=.8, applicable=36),
        "gradual_observable": dict(status="NOT_MET", rate=12 / 51, target=.7, applicable=51)}),
        promotion_decision=dict(gate_passed=False, status="NOT_PROMOTED")))
    gates = {r["check"]: r for r in gate_rows(data, dict(full_suite=True, integrity=True))}
    assert gates["full_suite"]["status"] == gates["integrity"]["status"] == "PASS"
    assert gates["research_target_abrupt_severe"]["status"] == "PASS"
    assert gates["research_target_gradual_observable"]["status"] == "FAIL"
    assert gates["dual_path_promotion"]["status"] == "FAIL"
    assert gates["hardware_emergency_readiness"]["status"] == "WARNING"


def test_undefined_research_rate_is_not_a_success_or_fabricated_zero():
    data = dict(research=dict(baseline=dict(research_target_checks={"x": dict(status="NOT_MET", rate=None, target=.7, applicable=0)}),
                              promotion_decision=dict(gate_passed=False, status="NOT_PROMOTED")))
    record = next(r for r in gate_rows(data, {}) if r["check"] == "research_target_x")
    assert record["status"] == "FAIL" and "measured=None" in record["detail"]


@pytest.mark.parametrize("attributes", ['tests="0"', 'tests="3" failures="1"', 'tests="3" errors="1"', 'tests="3" skipped="1"'])
def test_empty_failed_or_skipped_junit_cannot_pass_final_audit(tmp_path, attributes):
    path = tmp_path / "tests.xml"
    path.write_text(f"<testsuites><testsuite {attributes}/></testsuites>", encoding="utf-8")
    with pytest.raises(FinalAuditError):
        read_test_totals(path)


def test_multiple_junit_suites_are_aggregated_with_actual_denominators(tmp_path):
    path = tmp_path / "tests.xml"
    path.write_text('<testsuites><testsuite tests="3" time="1.25"/><testsuite tests="2" time="2.5"/></testsuites>', encoding="utf-8")
    totals = read_test_totals(path)
    assert totals["tests"] == 5 and totals["time_seconds"] == 3.75
    assert totals["failures"] == totals["errors"] == totals["skipped"] == 0


def test_corrupt_saved_artifact_and_snapshot_stop_the_audit(tmp_path):
    folder = tmp_path / "artifact"
    folder.mkdir()
    target = folder / "scores.csv"
    target.write_bytes(b"source")
    digest = sha(target)
    write_json(folder / "manifest.json", dict(outputs={"scores.csv": digest}))
    assert verify_manifest(folder, tmp_path)["outputs"]["scores.csv"] == digest
    snapshot = dict(files={"artifact/scores.csv": digest})
    verify_preservation(tmp_path, snapshot)
    target.write_bytes(b"altered")
    with pytest.raises(FinalAuditError, match="checksum mismatch"):
        verify_manifest(folder, tmp_path)
    with pytest.raises(FinalAuditError, match="Preserved input changed"):
        verify_preservation(tmp_path, snapshot)


def test_manifest_must_not_read_a_path_outside_workspace(tmp_path):
    with pytest.raises(FinalAuditError, match="leaves workspace"):
        safe_path(tmp_path, "../outside.txt")


def test_legacy_audit_writes_are_captured_without_overwriting_evidence(monkeypatch, tmp_path):
    import scripts.run_sprint13 as runner
    audit_dir = tmp_path / "scripts"
    audit_dir.mkdir()
    evidence = tmp_path / "artifacts/sprint1/verification.json"
    evidence.parent.mkdir(parents=True)
    evidence.write_text('{"historical":true}', encoding="utf-8")
    script = audit_dir / "verifier.py"
    script.write_text('from pathlib import Path\n'
        f'ROOT=Path({str(tmp_path)!r})\n'
        'def main():\n'
        '    (ROOT/"artifacts/sprint1/verification.json").write_text(\'{"status":"PASS","fresh_audit":true}\')\n', encoding="utf-8")
    before = (evidence.read_bytes(), evidence.stat().st_mtime_ns)
    monkeypatch.setattr(runner, "ROOT", tmp_path)
    result, _ = run_readonly_audit("verifier.py")
    assert result == dict(status="PASS", fresh_audit=True)
    assert (evidence.read_bytes(), evidence.stat().st_mtime_ns) == before


def test_legacy_verifier_cannot_write_arbitrary_non_evidence_files(monkeypatch, tmp_path):
    import scripts.run_sprint13 as runner
    folder = tmp_path / "scripts"
    folder.mkdir()
    (folder / "verifier.py").write_text('from pathlib import Path\n'
        f'ROOT=Path({str(tmp_path)!r})\n'
        'def main():\n'
        '    (ROOT/"raw.csv").write_text("changed")\n', encoding="utf-8")
    monkeypatch.setattr(runner, "ROOT", tmp_path)
    with pytest.raises(FinalAuditError, match="unexpected write"):
        run_readonly_audit("verifier.py")
    assert not (tmp_path / "raw.csv").exists()
