"""用法 python check_results.py AISBench时间戳结果目录 [预期题数]。"""
import json
from pathlib import Path
import sys

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
prediction_file = run_dir / "predictions" / model / "GPQA_diamond.jsonl"
rows = [json.loads(line) for line in prediction_file.read_text().splitlines() if line.strip()]
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
audit = {
    "questions": expected,
    "successful_requests": len(rows),
    "correct": sum(detail["correct"] for detail in score["details"]),
    "accuracy": score["accuracy"],
    "aisbench_accuracy": aisbench_result["accuracy"],
    "unparsed_answer_ids": [row["id"] for row, answer in zip(rows, answers) if answer is None],
    "output_limit_ids": [row["id"] for row in rows if row["output_tokens"] >= 32768],
}
(run_dir / "gpqa_audit.json").write_text(json.dumps(audit, ensure_ascii=False, indent=2) + "\n")
print(json.dumps(audit, ensure_ascii=False, indent=2))
assert abs(audit["accuracy"] - audit["aisbench_accuracy"]) < 1e-6, "最终答案审计与AISBench分数不同，请检查推理输出"
