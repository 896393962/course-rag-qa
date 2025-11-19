from __future__ import annotations

import json
import math
from pathlib import Path
from statistics import mean
from typing import Any

from .retriever import CourseKnowledgeBase


def reciprocal_rank(ranked_ids: list[str], relevant_ids: set[str]) -> float:
    for index, doc_id in enumerate(ranked_ids, start=1):
        if doc_id in relevant_ids:
            return 1.0 / index
    return 0.0


def dcg(relevance_scores: list[int]) -> float:
    return sum((2**score - 1) / math.log2(index + 2) for index, score in enumerate(relevance_scores))


def ndcg(ranked_ids: list[str], relevance: dict[str, int], k: int) -> float:
    predicted_scores = [relevance.get(doc_id, 0) for doc_id in ranked_ids[:k]]
    ideal_scores = sorted(relevance.values(), reverse=True)[:k]
    ideal_dcg = dcg(ideal_scores)
    if ideal_dcg == 0:
        return 0.0
    return dcg(predicted_scores) / ideal_dcg


def evaluate(top_k: int = 5) -> dict[str, Any]:
    project_root = Path(__file__).resolve().parents[1]
    kb = CourseKnowledgeBase(project_root / "data" / "sample_documents.json")
    seeds = json.loads((project_root / "data" / "eval_queries.json").read_text(encoding="utf-8"))
    queries = expand_eval_queries(seeds)

    rows = []
    for item in queries:
        hits = kb.search(item["query"], top_k=top_k, filters=item.get("filters", {}))
        ranked_ids = [hit.document.id for hit in hits]
        relevance = item["relevance"]
        relevant_ids = set(relevance)
        rows.append(
            {
                "query": item["query"],
                "ranked_ids": ranked_ids,
                "mrr": reciprocal_rank(ranked_ids, relevant_ids),
                "ndcg": ndcg(ranked_ids, relevance, top_k),
            }
        )

    return {
        "top_k": top_k,
        "query_count": len(rows),
        "mrr_at_k": round(mean(row["mrr"] for row in rows), 4),
        "ndcg_at_k": round(mean(row["ndcg"] for row in rows), 4),
        "details": rows,
    }


def expand_eval_queries(seeds: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Expand curated seeds into an 80+ item evaluation set.

    The variants simulate student phrasing differences while preserving the
    same relevance labels, which is enough for repeatable retrieval regression
    checks in this demo project.
    """

    prefixes = ["", "请解释：", "复习时怎么理解：", "考试会怎么问：", "用一句话说明：", "举例说明：", "对比分析："]
    rows = []
    for seed in seeds:
        for prefix in prefixes:
            query = seed["query"] if not prefix else prefix + seed["query"]
            rows.append({**seed, "query": query})
    return rows


def main() -> None:
    print(json.dumps(evaluate(top_k=5), ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
