"""用法 python check_results.py AISBench时间戳结果目录 [预期题数]。"""
import json
from pathlib import Path
import sys

from mmengine.config import Config

from ais_bench.benchmark.datasets import (
    GPQADataset,
    GPQAEvaluator,
    GPQA_Simple_Eval_postprocess,
)
from ais_bench.benchmark.utils.postprocess.model_postprocessors import (
    extract_non_reasoning_content,
)

run_dir = Path(sys.argv[1])
expected = int(sys.argv[2]) if len(sys.argv) > 2 else 198
model = "dsv41-bf16-cann930-gpqa"
config_files = list((run_dir / "configs").glob("*.py"))
assert len(config_files) == 1, "结果目录应有一份AISBench配置快照"
model_config = Config.fromfile(str(config_files[0])).models
assert len(model_config) == 1 and model_config[0]["abbr"] == model, "模型配置不匹配"
max_out_len = int(model_config[0]["max_out_len"])
assert max_out_len > 0, "输出预算应为正数"
prediction_file = run_dir / "predictions" / model / "GPQA_diamond.jsonl"
with prediction_file.open(encoding="utf-8") as stream:
    rows = [json.loads(line) for line in stream if line.strip()]
rows.sort(key=lambda row: row["id"])
dataset = GPQADataset.load(
    "/mnt/share/c00954197/Workspace/dsv41/benchmark/ais_bench/datasets/gpqa",
    "gpqa_diamond.csv",
)
assert len(dataset) == 198 and 0 < expected <= len(dataset)
assert [row["id"] for row in rows] == list(range(expected)), "题号缺失、重复或题数不匹配"
assert all(row["success"] for row in rows), "存在请求失败"
references = dataset["answer"][:expected]
assert [row["gold"] for row in rows] == references, "答案与当前数据集不匹配"

# 推理字段存在时，content 保存服务返回的最终答案，空 content 计为无答案。
texts = [row.get("content", row["prediction"]) for row in rows]
answers = [GPQA_Simple_Eval_postprocess(extract_non_reasoning_content(text)) for text in texts]
score = GPQAEvaluator().score(answers, references)
aisbench_result = json.loads((run_dir / "results" / model / "GPQA_diamond.json").read_text())
raw_answers = [GPQA_Simple_Eval_postprocess(extract_non_reasoning_content(row["prediction"])) for row in rows]
raw_score = GPQAEvaluator().score(raw_answers, references)
assert abs(raw_score["accuracy"] - aisbench_result["accuracy"]) < 1e-6, "AISBench原始分数复算不一致"
difference_ids = [
    row["id"] for row, raw, final, gold in zip(rows, raw_answers, answers, references)
    if (raw == gold) != (final == gold)
]
assert all(rows[index].get("content") == "" for index in difference_ids), "存在需核查的非空最终答案评分差异"
audit = {
    "questions": expected,
    "max_out_len": max_out_len,
    "successful_requests": len(rows),
    "correct": sum(detail["correct"] for detail in score["details"]),
    "accuracy": score["accuracy"],
    "aisbench_accuracy": aisbench_result["accuracy"],
    "aisbench_raw_recomputed_accuracy": raw_score["accuracy"],
    "scoring_difference_ids": difference_ids,
    "scoring_validated": True,
    "unparsed_answer_ids": [row["id"] for row, answer in zip(rows, answers) if answer is None],
    "output_limit_ids": [row["id"] for row in rows if row["output_tokens"] >= max_out_len],
}
(run_dir / "gpqa_audit.json").write_text(json.dumps(audit, ensure_ascii=False, indent=2) + "\n")
print(json.dumps(audit, ensure_ascii=False, indent=2))
