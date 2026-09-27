import hashlib
import json
import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from fabjudge.jev_gate import canonical, request_spec


class TestJEVCache(unittest.TestCase):
 def test_v2_same_row_from_independent_serializations_has_same_body_and_key(self):
    question = {"type": "score", "instructions": "test", "criteria": ["a", "b", "c"]}
    original = {"lstm_pred_seconds": 4123.123456789012, "rf_probability": 0.512345678901234,
                "recent_lstm_delta": -12.1234567890123, "recent_lstm_std": 3.456789012345,
                "history_count": 5, "ROTATIONSPEED": -0.0032242732122540474}
    reloaded = json.loads(json.dumps(original))
    specs = [request_spec(row, question, ["ROTATIONSPEED"], include_history=False,
                          serialization_version="v2") for row in (original, reloaded)]
    bodies = [canonical(spec["body"]).encode("utf-8") for spec in specs]
    self.assertEqual(bodies[0], bodies[1])
    self.assertEqual(hashlib.sha256(bodies[0]).hexdigest(), hashlib.sha256(bodies[1]).hexdigest())
    self.assertEqual(specs[0]["body"]["state"]["features"]["rf_probability"], 0.512345678901)


 def test_v2_can_exclude_rf_without_changing_other_state(self):
    row = {"lstm_pred_seconds": 2000.0, "rf_probability": .5,
           "recent_lstm_delta": -2.0, "recent_lstm_std": 1.0, "history_count": 5}
    q = {"type": "score", "instructions": "test", "criteria": ["a", "b", "c"]}
    with_rf = request_spec(row, q, serialization_version="v2")["body"]["state"]["features"]
    without_rf = request_spec(row, q, serialization_version="v2", include_rf=False)["body"]["state"]["features"]
    self.assertEqual({k: v for k, v in with_rf.items() if k != "rf_probability"}, without_rf)

 def test_rf_field_changes_state_and_v2_cache_key(self):
    row = {"lstm_pred_seconds": 2000.0, "rf_probability": .8,
           "rf_probability_oof": .2, "recent_lstm_delta": -2.0,
           "recent_lstm_std": 1.0, "history_count": 5}
    q = {"type": "score", "instructions": "test", "criteria": ["a", "b", "c"]}
    specs = [request_spec(row, q, include_history=False, serialization_version="v2",
                          rf_field=field) for field in ("rf_probability", "rf_probability_oof")]
    states = [s["body"]["state"]["features"] for s in specs]
    keys = [hashlib.sha256(canonical(s["body"]).encode("utf-8")).hexdigest() for s in specs]
    self.assertEqual(states[0]["rf_probability"], .8)
    self.assertEqual(states[1]["rf_probability"], .2)
    self.assertNotEqual(keys[0], keys[1])


if __name__ == "__main__":
    unittest.main()
