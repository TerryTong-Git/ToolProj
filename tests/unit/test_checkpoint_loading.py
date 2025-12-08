from typing import Any

import pandas as pd

from src.exps_performance.logger import CheckpointManager, Record, make_request_id


def test_load_legacy_rows_preserves_request_id_and_fills_index(tmp_path: Any) -> None:
    """
    Legacy rows may be missing index_in_kind; we should not rewrite request_id on load,
    but we should assign indexes for convenience.
    """
    csv_path = tmp_path / "res.csv"
    legacy = [
        {
            "request_id": "legacy-1",
            "model": "m",
            "seed": 0,
            "exp_id": "run_x",
            "digit": 2,
            "kind": "add",
            "question": "q1",
            "answer": "a1",
            "index_in_kind": -1,
        },
        {
            "request_id": "legacy-2",
            "model": "m",
            "seed": 0,
            "exp_id": "run_x",
            "digit": 2,
            "kind": "add",
            "question": "q2",
            "answer": "a2",
            "index_in_kind": -1,
        },
    ]
    pd.DataFrame(legacy).to_csv(csv_path, index=False)

    ckpt = CheckpointManager(str(csv_path))
    recs = list(ckpt._records.values())

    assert {r.request_id for r in recs} == {"legacy-1", "legacy-2"}, "request_id should not be recomputed"
    # index_in_kind should be assigned in order per kind
    assert sorted(r.index_in_kind for r in recs) == [1, 2]


def test_get_by_question_and_upsert_migrates_to_index(tmp_path: Any) -> None:
    """
    If we find a legacy row by question hash, we upsert it so later lookups use the index-based id.
    """
    csv_path = tmp_path / "res.csv"
    question = "some question"
    legacy_id = "legacy-q"
    pd.DataFrame(
        [
            {
                "request_id": legacy_id,
                "model": "m",
                "seed": 0,
                "exp_id": "run_x",
                "digit": 2,
                "kind": "add",
                "question": question,
                "answer": "a",
                "index_in_kind": -1,
            }
        ]
    ).to_csv(csv_path, index=False)

    ckpt = CheckpointManager(str(csv_path))
    rec = ckpt.get_by_question("add", 2, question)
    assert rec is not None, "should find legacy by question"

    # simulate assigning index and new request id
    rec.index_in_kind = 1
    rec.request_id = make_request_id("add", 2, 1, rec.seed, rec.model)
    ckpt.upsert(rec, flush=False)
    ckpt.flush()

    df = pd.read_csv(csv_path)
    assert len(df) == 2  # legacy row + migrated row
    assert rec.request_id in set(df["request_id"]), "migrated id should be present"


def test_upsert_does_not_append_duplicate_request_id(tmp_path: Any) -> None:
    """
    Once a request_id is flushed, re-upserting the same record should not append another row.
    """
    csv_path = tmp_path / "res.csv"
    ckpt = CheckpointManager(str(csv_path))
    rec = Record(request_id="rid-1", model="m", seed=0, exp_id="run", digit=2, kind="add", question="q", answer="a")

    ckpt.upsert(rec, flush=True)
    ckpt.upsert(rec, flush=True)  # should not append

    df = pd.read_csv(csv_path)
    assert len(df) == 1, "duplicate request_id should not be appended twice"
