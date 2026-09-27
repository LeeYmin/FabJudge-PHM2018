import sys
import unittest
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from fabjudge.pipeline.claims import verify_claims
from fabjudge.pipeline.baselines import logistic_group_oof
from fabjudge.pipeline.calibration import fit_isotonic_probability
from fabjudge.pipeline.metrics import k_for_fraction, precision_at_k
from fabjudge.pipeline.routing import (
    compose_calibrated_queue_scores, random_audit_indices, route_score,
)
from fabjudge.pipeline.schemas import PACKET_FIELDS, make_packet, packet_issues, validate_packet


def example_packet():
    fields = {name: 0.25 for name in PACKET_FIELDS}
    fields["fault_type"] = 2
    return make_packet(fields)


class TestEvidencePacket(unittest.TestCase):
    def test_packet_matches_frozen_schema(self):
        packet = example_packet()
        validate_packet(packet)
        self.assertEqual(packet_issues(packet), [])

    def test_packet_identifier_or_label_field_is_rejected(self):
        packet = example_packet()
        packet["sequence_id"] = "must-not-be-present"
        with self.assertRaisesRegex(ValueError, "Forbidden"):
            packet_issues(packet)

    def test_nonfinite_value_is_data_review_without_imputation(self):
        packet = example_packet()
        packet["fields"]["rf_probability"] = None
        issues = packet_issues(packet)
        self.assertTrue(any("rf_probability" in issue for issue in issues))
        self.assertIsNone(packet["fields"]["rf_probability"])


class TestClaimsRoutingAndMetrics(unittest.TestCase):
    def test_claim_verifier_marks_mismatched_value(self):
        packet = example_packet()
        output = {
            "evidence_for": [{"field": "lstm_pred_seconds", "value": 0.25, "claim": "exact"}],
            "evidence_against": [{"field": "rf_probability", "value": 0.9, "claim": "not exact"}],
        }
        report = verify_claims(packet, output)
        self.assertEqual(report["claim_count"], 2)
        self.assertEqual(report["matching_claim_count"], 1)
        self.assertEqual(report["mismatched_claim_count"], 1)

    def test_router_never_cancels_alarm_and_routes_three_tiers(self):
        high = route_score(1.8, 0.5, 1.5)
        mid = route_score(1.0, 0.5, 1.5)
        low = route_score(0.2, 0.5, 1.5, audit_draw=0.05)
        self.assertEqual([high["route"], mid["route"], low["route"]], ["HIGH", "MID", "LOW"])
        self.assertFalse(high["llm_call"])
        self.assertTrue(mid["llm_call"])
        self.assertTrue(low["llm_call"])
        self.assertTrue(all(row["alarm_retained"] for row in (high, mid, low)))

    def test_precision_at_k_and_half_up_rounding(self):
        self.assertEqual(k_for_fraction(5, 0.5), 3)
        metric = precision_at_k([1, 0, 1, 0], [0.9, 0.8, 0.7, 0.1], fraction=0.5)
        self.assertEqual(metric["k"], 2)
        self.assertEqual(metric["precision_at_k"], 0.5)

    def test_random_router_has_fixed_count_and_seed(self):
        first = random_audit_indices(20, 6, seed=42, repetitions=100)
        second = random_audit_indices(20, 6, seed=42, repetitions=100)
        self.assertEqual(len(first), 100)
        for left, right in zip(first, second):
            np.testing.assert_array_equal(left, right)
            self.assertEqual(len(left), 6)

    def test_unavailable_scores_are_not_zero_filled_for_queue_ordering(self):
        calibration = fit_isotonic_probability([0, .1, .2, .8, .9, 1], [0, 0, 0, 1, 1, 1])
        rows = compose_calibrated_queue_scores(
            [.9, None, None], [.2, .3, .4], [True, True, False], calibration, calibration,
            row_statuses=["READY", "MODEL_UNAVAILABLE", "DATA_REVIEW"],
        )
        self.assertGreater(rows[0]["queue_score"], .5)
        self.assertEqual(rows[1], {"queue_score": None, "status": "MODEL_UNAVAILABLE", "score_source": None})
        self.assertEqual(rows[2], {"queue_score": None, "status": "DATA_REVIEW", "score_source": None})

    def test_logistic_baseline_predictions_are_group_oof(self):
        packets = []
        labels = []
        groups = []
        for index in range(12):
            fields = {name: (index + 1) / 12 for name in PACKET_FIELDS}
            fields["fault_type"] = index % 3 + 1
            packets.append(make_packet(fields))
            labels.append(index % 2)
            groups.append(f"sequence-group-{index}")
        predictions = logistic_group_oof(packets, labels, groups, n_splits=6)
        self.assertEqual(len(predictions), 12)
        self.assertTrue(predictions.logistic_probability_oof.between(0, 1).all())
        self.assertEqual(set(predictions.group_fold), {1, 2, 3, 4, 5, 6})


if __name__ == "__main__":
    unittest.main()
