"""Recompute all three scores from saved predictions using the captured AISBench functions."""

import ast
import gzip
import hashlib
import json
import logging
import re
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
MODEL = "dsv41-bf16-cann930-gpqa"


def read(path):
    return json.loads(path.read_text())


def captured_function(filename, name):
    path = ROOT / "sources" / filename
    tree = ast.parse(path.read_text())
    function = next(node for node in tree.body if isinstance(node, ast.FunctionDef) and node.name == name)
    function.decorator_list = []
    module = ast.Module(body=[ast.ImportFrom(module="__future__", names=[ast.alias(name="annotations")], level=0),
                             function], type_ignores=[])
    ast.fix_missing_locations(module)
    namespace = {"re": re, "logger": logging.getLogger("saved_gpqa_audit")}
    exec(compile(module, str(path), "exec"), namespace)
    return namespace[name]


def main():
    summary = read(ROOT / "run/summary.json")
    evidence = read(ROOT / "run/results_summary.json")
    request = read(ROOT / "run/request.json")
    configurations = read(ROOT / "run/config_audit.json")
    dataset = read(ROOT / "dataset_reference.json")
    extract = captured_function("aisbench_model_postprocessors.py", "extract_non_reasoning_content")
    answer = captured_function("aisbench_gpqa.py", "GPQA_Simple_Eval_postprocess")
    assert summary["state"] == "COMPLETED" and len(summary["results"]) == 3
    assert configurations["status"] == "PASS" and configurations["source_hashes_match_request"]
    assert dataset["dataset_sha256"] == evidence["dataset_sha256"]
    assert len(dataset["answer_labels"]) == 198
    for file in request["files"]:
        path = ROOT / "scripts" / Path(file["path"]).name
        assert hashlib.sha256(path.read_bytes()).hexdigest() == file["sha256"], path

    results = []
    for row, config, recorded in zip(summary["results"], configurations["rounds"], evidence["round_evidence"]):
        number = row["round"]
        assert number == config["round"] == recorded["round"]
        assert row["exit_code"] == 0 and row["pass"]
        directory = ROOT / "results" / f"round_{number}"
        config_file = directory / "configs" / Path(config["config_path"]).name
        assert hashlib.sha256(config_file.read_bytes()).hexdigest() == config["config_sha256"]
        settings = config["generation_kwargs"]
        assert settings == {"temperature": 1.0, "top_p": 0.95, "top_k": 0, "seed": 42,
                            "ignore_eos": False,
                            "chat_template_kwargs": {"reasoning_effort": "low", "enable_thinking": True}}
        assert config["model_settings"]["max_out_len"] == 65536
        assert config["model_settings"]["batch_size"] == 8
        assert config["model_settings"]["request_timeout"] == 360000
        prediction = directory / "predictions" / MODEL / "GPQA_diamond.jsonl.gz"
        assert hashlib.sha256(gzip.decompress(prediction.read_bytes())).hexdigest() == recorded["predictions_sha256"]
        with gzip.open(prediction, "rt", encoding="utf-8") as stream:
            predictions = sorted((json.loads(line) for line in stream if line.strip()), key=lambda item: item["id"])
        assert [item["id"] for item in predictions] == list(range(198))
        assert all(item["success"] for item in predictions)
        assert [item["gold"] for item in predictions] == dataset["answer_labels"]
        final_answers = [answer(extract(item.get("content", item["prediction"]))) for item in predictions]
        raw_answers = [answer(extract(item["prediction"])) for item in predictions]
        correct = sum(value == gold for value, gold in zip(final_answers, dataset["answer_labels"]))
        raw_correct = sum(value == gold for value, gold in zip(raw_answers, dataset["answer_labels"]))
        unparsed = [item["id"] for item, value in zip(predictions, final_answers) if value is None]
        limit = [item["id"] for item in predictions if item["output_tokens"] >= 65536]
        differences = [item["id"] for item, final, raw, gold in
                       zip(predictions, final_answers, raw_answers, dataset["answer_labels"])
                       if (final == gold) != (raw == gold)]
        audit = read(directory / "gpqa_audit.json")
        original = read(directory / "results" / MODEL / "GPQA_diamond.json")
        assert correct == audit["correct"] == row["audit"]["correct"]
        assert audit["successful_requests"] == audit["questions"] == 198
        assert audit["max_out_len"] == 65536 and audit["scoring_validated"]
        assert abs(correct * 100 / 198 - audit["accuracy"]) < 1e-9
        assert abs(raw_correct * 100 / 198 - original["accuracy"]) < 1e-9
        assert unparsed == audit["unparsed_answer_ids"]
        assert limit == audit["output_limit_ids"]
        assert differences == audit["scoring_difference_ids"] == []
        assert sum(item["output_tokens"] for item in predictions) == recorded["output_tokens"]
        assert max(item["input_tokens"] for item in predictions) == recorded["max_input_tokens"]
        results.append({"round": number, "successful_requests": 198, "correct": correct,
                        "accuracy": correct * 100 / 198, "output_limit_ids": limit,
                        "unparsed_answer_ids": unparsed, "scoring_difference_ids": differences,
                        "prediction_sha256_verified": True, "config_sha256_verified": True})

    mean = sum(row["accuracy"] for row in results) / 3
    assert abs(mean - summary["mean_accuracy"]) < 1e-9
    for node in ("23", "24"):
        prefix = summary["server_logs_before_stop"][f"node{node}"]
        log = (ROOT / "run" / f"node{node}" / "launcher.log").read_bytes()
        assert hashlib.sha256(log[:prefix["bytes"]]).hexdigest() == prefix["sha256"]
        assert not re.search(rb"\bERROR\b|\bTraceback\b", log)
        release = read(ROOT / "run" / f"node{node}" / "release_check.json")
        assert release["npu_pids"] == release["remaining_owned"] == release["listeners"] == []
        assert read(ROOT / "run" / f"node{node}" / "shm_release_review.json")["delta_bytes"] == 0
    for logs in evidence["logs"].values():
        assert logs["errors_before_stop"] == logs["errors_during_shutdown"] == 0
        assert logs["engine_initializations"] == 1 and logs["worker_weight_load_completions"] == 8
    result = {"status": "passed", "scope": "CPU audit of saved predictions, scores, configs and release records",
              "rounds": results, "mean_accuracy": mean, "successful_requests": 594,
              "correct_across_rounds": sum(row["correct"] for row in results),
              "original_data_preserved": True}
    (ROOT / "evidence_validation.json").write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
