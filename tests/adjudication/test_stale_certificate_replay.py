"""Integration tests for stale-certificate replay through the live commit API."""
from __future__ import annotations

from pathlib import Path
import pytest

from scripts.run_stale_certificate_experiment import run_stale_certificate_experiment, write_replay_outputs

pytestmark = pytest.mark.adjudication


def test_stale_certificate_replay_experiment_all_scenarios_pass(tmp_path: Path):
    """Verify all 6 stale-certificate replay controls pass through CertificateSystem.commit_decision."""
    report = run_stale_certificate_experiment()
    assert report["experiment"] == "stale-certificate-replay-v1"
    assert report["total_scenarios"] == 7
    assert report["passed_scenarios"] == 7
    assert report["all_passed"] is True

    by_id = {s["scenario_id"]: s for s in report["scenarios"]}
    assert by_id["1_clean_positive_control"]["actual_verdict"] == "ACCEPTED"
    assert by_id["2_stale_commit_replay_label_changes"]["actual_verdict"] == "REFUSED"
    assert by_id["3_stale_authorization_replay_label_changes"]["actual_verdict"] == "REFUSED"
    assert by_id["4a_same_label_stale_replay_approve"]["actual_verdict"] == "REFUSED"
    assert by_id["4b_same_label_stale_replay_reject"]["actual_verdict"] == "REFUSED"
    assert by_id["5_unrelated_edit_control"]["actual_verdict"] == "ACCEPTED"
    assert by_id["6_post_recovery_positive_control"]["actual_verdict"] == "ACCEPTED"

    json_path, md_path = write_replay_outputs(report, tmp_path / "phase2")
    assert json_path.exists()
    assert md_path.exists()
