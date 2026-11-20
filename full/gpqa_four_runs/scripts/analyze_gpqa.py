"""Audit four saved GPQA runs and compare each later run with the first.

Uses only the Python standard library; does not generate new model responses.
The paired t tests compare per-question binary correctness, not run-level means.
"""

import csv
import gzip
import hashlib
import importlib.util
import json
import math
import statistics
from pathlib import Path


OUT = Path(__file__).resolve().parents[1]
REPO = OUT.parents[1]
MODEL = "dsv41-bf16-cann930-gpqa"
OLD = REPO / "full/cann930_20261117"
NEW = REPO / "full/cann930_20261119_precision"


def read(path):
    return json.loads(path.read_text(encoding="utf-8"))


def beta_fraction(a, b, x):
    """Continued fraction for the regularized incomplete beta function."""
    tiny = 1e-300
    qab, qap, qam = a + b, a + 1, a - 1
    c, d = 1.0, 1.0 - qab * x / qap
    d = 1 / (d if abs(d) > tiny else tiny)
    h = d
    for m in range(1, 10001):
        m2 = 2 * m
        aa = m * (b - m) * x / ((qam + m2) * (a + m2))
        d, c = 1 + aa * d, 1 + aa / c
        d = d if abs(d) > tiny else tiny
        c = c if abs(c) > tiny else tiny
        d = 1 / d
        h *= d * c
        aa = -(a + m) * (qab + m) * x / ((a + m2) * (qap + m2))
        d, c = 1 + aa * d, 1 + aa / c
        d = d if abs(d) > tiny else tiny
        c = c if abs(c) > tiny else tiny
        d = 1 / d
        delta = d * c
        h *= delta
        if abs(delta - 1) < 3e-14:
            return h
    raise ArithmeticError("Incomplete beta continued fraction did not converge")


def regularized_beta(x, a, b):
    if x <= 0:
        return 0.0
    if x >= 1:
        return 1.0
    factor = math.exp(math.lgamma(a + b) - math.lgamma(a) - math.lgamma(b)
                      + a * math.log(x) + b * math.log1p(-x))
    if x < (a + 1) / (a + b + 2):
        return factor * beta_fraction(a, b, x) / a
    return 1 - factor * beta_fraction(b, a, 1 - x) / b


def t_two_sided_p(t, df):
    return regularized_beta(df / (df + t * t), df / 2, 0.5)


def t_critical(df, alpha=0.05):
    lo, hi = 0.0, 100.0
    for _ in range(100):
        middle = (lo + hi) / 2
        if t_two_sided_p(middle, df) > alpha:
            lo = middle
        else:
            hi = middle
    return (lo + hi) / 2


def paired_test(reference, later):
    differences = [y - x for x, y in zip(reference, later)]
    n = len(differences)
    mean = statistics.mean(differences)
    se = statistics.stdev(differences) / math.sqrt(n)
    if se == 0:
        raise ValueError("Degenerate paired differences require separate handling")
    t = mean / se
    margin = t_critical(n - 1) * se
    gains = differences.count(1)
    losses = differences.count(-1)
    discordant = gains + losses
    exact_p = min(1.0, 2 * sum(math.comb(discordant, k)
                              for k in range(min(gains, losses) + 1)) / 2**discordant)
    return {"pairs": n, "df": n - 1, "gains": gains, "losses": losses,
            "mean_difference_percentage_points": 100 * mean,
            "t_statistic": t, "p_two_sided": t_two_sided_p(t, n - 1),
            "ci95_percentage_points": [100 * (mean - margin), 100 * (mean + margin)],
            "mcnemar_exact_p_two_sided": exact_p}


def main():
    # Analytic Cauchy (df=1) and published critical values check the CDF calculation.
    assert abs(t_two_sided_p(1, 1) - 0.5) < 1e-12
    assert abs(t_two_sided_p(2, 1) - (1 - 2 * math.atan(2) / math.pi)) < 1e-12
    assert abs(t_two_sided_p(1.959963984540054, 1000000) - 0.05) < 1e-6
    assert abs(t_two_sided_p(2.2281388519649385, 10) - 0.05) < 1e-10

    spec = importlib.util.spec_from_file_location("saved_gpqa_audit", NEW / "scripts/audit_saved_results.py")
    audit_module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(audit_module)
    extract = audit_module.captured_function("aisbench_model_postprocessors.py", "extract_non_reasoning_content")
    answer = audit_module.captured_function("aisbench_gpqa.py", "GPQA_Simple_Eval_postprocess")
    dataset = read(NEW / "dataset_reference.json")
    old_datasets = read(OLD / "combined_run/dataset_manifest.json")
    assert dataset["dataset_sha256"] in old_datasets.values()
    old_record = read(OLD / "combined_run/clients/gpqa_198_result.json")
    new_records = read(NEW / "run/results_summary.json")["round_evidence"]
    paths = [OLD / "precision_results/gpqa_198"] + [NEW / f"results/round_{i}" for i in (1, 2, 3)]
    all_rows, all_scores, summaries = [], [], []
    for i, folder in enumerate(paths):
        prediction = folder / "predictions" / MODEL / "GPQA_diamond.jsonl.gz"
        decompressed = gzip.decompress(prediction.read_bytes())
        sha = hashlib.sha256(decompressed).hexdigest()
        expected_sha = old_record["prediction_sha256"] if i == 0 else new_records[i - 1]["predictions_sha256"]
        assert sha == expected_sha
        # splitlines() also splits valid U+2028 inside JSON strings; use LF only.
        rows = sorted((json.loads(line) for line in decompressed.decode("utf-8").split("\n")
                       if line.strip()), key=lambda row: row["id"])
        assert [row["id"] for row in rows] == list(range(198))
        assert all(row["success"] and row["data_abbr"] == "GPQA_diamond" for row in rows)
        assert [row["gold"] for row in rows] == dataset["answer_labels"]
        if all_rows:
            assert all(row["origin_prompt"] == original["origin_prompt"]
                       for row, original in zip(rows, all_rows[0]))
        finals = [answer(extract(row["content"])) for row in rows]
        raw = [answer(extract(row["prediction"])) for row in rows]
        scores = [int(value == row["gold"]) for row, value in zip(rows, finals)]
        audit = read(folder / "gpqa_audit.json")
        assert sum(scores) == audit["correct"]
        limit = 32768 if i == 0 else 65536
        unparsed = [row["id"] for row, value in zip(rows, finals) if value is None]
        capped = [row["id"] for row in rows if row["output_tokens"] >= limit]
        assert unparsed == audit["unparsed_answer_ids"] and capped == audit["output_limit_ids"]
        summaries.append({"run": i + 1, "date": "2026-11-17" if i == 0 else "2026-11-19",
                          "source_round": None if i == 0 else i, "questions": 198,
                          "successful_requests": 198, "correct": sum(scores),
                          "accuracy_percent": 100 * sum(scores) / 198,
                          "raw_correct": sum(value == row["gold"] for row, value in zip(rows, raw)),
                          "output_budget": limit, "unparsed_answer_ids": unparsed,
                          "output_limit_ids": capped, "prediction_sha256": sha,
                          "source": str(prediction.relative_to(REPO))})
        all_rows.append(rows)
        all_scores.append(scores)

    comparisons = [{"reference_run": 1, "later_run": i + 1,
                    **paired_test(all_scores[0], scores)} for i, scores in enumerate(all_scores[1:], 1)]
    # Holm adjustment remains valid when comparisons share the same reference.
    maximum = 0.0
    for rank, index in enumerate(sorted(range(3), key=lambda j: comparisons[j]["p_two_sided"])):
        maximum = max(maximum, min(1.0, (3 - rank) * comparisons[index]["p_two_sided"]))
        comparisons[index]["p_holm_three_comparisons"] = maximum
        comparisons[index]["significant_after_holm_at_0_05"] = maximum < 0.05

    official = read(OUT / "official_reference.json")
    official_percent = official["flash_1017_deepinfer"]["gpqa_pass_at_1_percent"]
    latest_scores = [row["accuracy_percent"] for row in summaries[1:]]
    mean_latest = statistics.mean(latest_scores)
    se_latest = statistics.stdev(latest_scores) / math.sqrt(len(latest_scores))
    reference_t = (mean_latest - official_percent) / se_latest
    one_sample = {
        "method": "exploratory one-sample t on the three latest run-level accuracies",
        "reference_percent": official_percent, "n_runs": 3, "df": 2,
        "observed_mean_percent": mean_latest,
        "difference_percentage_points": mean_latest - official_percent,
        "t_statistic": reference_t, "nominal_p_two_sided": t_two_sided_p(reference_t, 2),
        "ci95_percent": [mean_latest - t_critical(2) * se_latest,
                         mean_latest + t_critical(2) * se_latest],
        "acceptance_status": "not established",
        "limitations": ["The official percentage is treated as a fixed rounded reference, ignoring its uncertainty.",
                        "Only three runs, sharing seed, questions and service; independence is not established.",
                        "Reasoning effort, sampling count, top_p, token budget, scoring and backend differ.",
                        "This is not a two-sample, paired official comparison, equivalence or non-inferiority test."]}

    OUT.mkdir(parents=True, exist_ok=True)
    with (OUT / "paired_scores.csv").open("w", newline="", encoding="utf-8") as stream:
        writer = csv.writer(stream, lineterminator="\n")
        writer.writerow(["question_id", "prompt_sha256", "gold", "run1_correct", "run2_correct",
                         "run3_correct", "run4_correct"])
        for i, row in enumerate(all_rows[0]):
            prompt = json.dumps(row["origin_prompt"], sort_keys=True, ensure_ascii=False).encode("utf-8")
            writer.writerow([i, hashlib.sha256(prompt).hexdigest(), row["gold"],
                             *(scores[i] for scores in all_scores)])
    result = {"analysis": "paired_question_correctness_t_tests", "alpha": 0.05,
              "dataset_sha256": dataset["dataset_sha256"], "all_prompts_and_gold_match": True,
              "scoring": "final content; unparsed answers counted incorrect", "runs": summaries,
              "four_run_descriptive_mean_percent": statistics.mean(x["accuracy_percent"] for x in summaries),
              "latest_three_mean_percent": statistics.mean(x["accuracy_percent"] for x in summaries[1:]),
              "comparisons": comparisons, "confidence_intervals": "individual, not simultaneous",
              "official_fixed_reference_exploratory_t": one_sample,
              "limitations": ["Generation configuration differs between run 1 and runs 2-4.",
                              "All runs use the same 198 questions and seed 42; not 792 independent questions.",
                              "No matched competitor or official reference responses were supplied.",
                              "A difference test is not an equivalence or non-inferiority acceptance test.",
                              "Binary paired differences use a large-sample t approximation; exact McNemar is also reported."],
              "numerics": "standard-library incomplete beta; analytic and critical-value checks passed"}
    (OUT / "results.json").write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"correct": [x["correct"] for x in summaries],
                      "mean_four": result["four_run_descriptive_mean_percent"],
                      "official_comparison": one_sample,
                      "comparisons": comparisons}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
