"""Offline verification tests for Phase 8 manuscript and audit bundle."""
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT / "scripts") not in sys.path:
    sys.path.insert(0, str(ROOT / "scripts"))

from verify_phase8_bundle import EXPECTED_TITLE, verify_phase8_bundle


def test_phase8_bundle_passes_all_static_and_parity_checks():
    """Offline bundle audit confirms title, citations, zero leaks, and Phase 2-7 numeric parity."""
    report = verify_phase8_bundle()
    assert report["status"] == "PASS"
    assert report["title"] == EXPECTED_TITLE
    assert report["citations_count"] == report["bib_entries_count"]
    assert report["citations_count"] == 18
