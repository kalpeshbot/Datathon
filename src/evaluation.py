"""
LAPE (Leak Analysis & Prediction Engine)
Module: Fair Evaluation & Benchmarking
DATATHON 2026 Track 4: The Money Leak

Computes rigorous, honest evaluation metrics:
- Precision, Recall, F1
- PR-AUC (Average Precision)
- Precision@K (e.g. top 10, top 25 review budget)
- False Positive Rate (FPR) on valid controls
- Per-scenario-family breakdown
- Transparent Baseline vs. ML Model comparison matrix
"""

from typing import Dict, List, Tuple, Any
import numpy as np
import pandas as pd
from sklearn.metrics import (
    precision_score,
    recall_score,
    f1_score,
    precision_recall_curve,
    auc,
    confusion_matrix,
    average_precision_score
)


def compute_metrics(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    y_score: np.ndarray,
    budget_k: int = 10
) -> Dict[str, Any]:
    """Computes comprehensive evaluation metrics for binary triage task."""
    y_true = np.asarray(y_true, dtype=int)
    y_pred = np.nan_to_num(np.asarray(y_pred, dtype=float), nan=0.0).astype(int)
    y_score = np.nan_to_num(np.asarray(y_score, dtype=float), nan=0.0)

    # Standard metrics
    prec = float(precision_score(y_true, y_pred, zero_division=0))
    rec = float(recall_score(y_true, y_pred, zero_division=0))
    f1 = float(f1_score(y_true, y_pred, zero_division=0))
    
    # PR-AUC
    try:
        pr_auc = float(average_precision_score(y_true, y_score))
    except Exception:
        pr_auc = 0.0

    # Confusion Matrix: TN, FP, FN, TP
    cm = confusion_matrix(y_true, y_pred, labels=[0, 1])
    tn, fp, fn, tp = [int(v) for v in cm.ravel()]

    # False Positive Rate on Negatives: FP / (FP + TN)
    fpr = float(fp / (fp + tn)) if (fp + tn) > 0 else 0.0

    # Precision@K: Top K highest scoring items
    k = min(budget_k, len(y_score))
    top_k_indices = np.argsort(y_score)[::-1][:k]
    prec_at_k = float(np.mean(y_true[top_k_indices])) if k > 0 else 0.0

    return {
        "precision": prec,
        "recall": rec,
        "f1": f1,
        "pr_auc": pr_auc,
        f"precision_at_{budget_k}": prec_at_k,
        "fpr_on_controls": fpr,
        "true_positives": tp,
        "false_positives": fp,
        "true_negatives": tn,
        "false_negatives": fn,
        "confusion_matrix": [[tn, fp], [fn, tp]]
    }


def compute_family_breakdown(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    scenario_families: np.ndarray
) -> Dict[str, Dict[str, Any]]:
    """Evaluates detection rates across distinct scenario families."""
    df = pd.DataFrame({
        "y_true": y_true,
        "y_pred": y_pred,
        "family": scenario_families
    })
    
    breakdown = {}
    for fam, group in df.groupby("family"):
        fam_true = group["y_true"].to_numpy()
        fam_pred = group["y_pred"].to_numpy()
        
        is_control = (fam_true.sum() == 0)
        if is_control:
            # For controls, report specificity and false positive count
            fp = int((fam_pred == 1).sum())
            total = len(group)
            breakdown[str(fam)] = {
                "type": "control",
                "sample_count": total,
                "false_positives": fp,
                "false_positive_rate": round(fp / total, 4) if total > 0 else 0.0,
                "clean_pass_rate": round(1.0 - (fp / total), 4) if total > 0 else 1.0
            }
        else:
            tp = int(((fam_pred == 1) & (fam_true == 1)).sum())
            total_pos = int(fam_true.sum())
            recall = round(tp / total_pos, 4) if total_pos > 0 else 0.0
            breakdown[str(fam)] = {
                "type": "leak_scenario",
                "sample_count": len(group),
                "true_positives": tp,
                "recall": recall,
                "detection_pct": round(recall * 100.0, 1)
            }
    return breakdown


def benchmark_baselines_vs_model(
    test_df: pd.DataFrame,
    test_meta: pd.DataFrame,
    y_test: np.ndarray,
    model_preds: np.ndarray,
    model_scores: np.ndarray,
    budget_k: int = 10
) -> Dict[str, Any]:
    """
    Rigorously benchmarks:
      1. Baseline A: Simple Exposure Ranking (Raw Shortfall Threshold)
      2. Baseline B: Derived-Feature Composite Rules
      3. LAPE ML Model: Tree-based Classifier
    """
    # 1. Baseline A metrics
    score_a = test_df.get("rule_a_score", np.zeros(len(test_df))).to_numpy()
    pred_a = test_df.get("rule_a_flag", (score_a >= 0.1).astype(int)).to_numpy()
    metrics_a = compute_metrics(y_test, pred_a, score_a, budget_k=budget_k)

    # 2. Baseline B metrics
    score_b = test_df.get("rule_b_score", np.zeros(len(test_df))).to_numpy()
    pred_b = test_df.get("rule_b_flag", (score_b >= 0.5).astype(int)).to_numpy()
    metrics_b = compute_metrics(y_test, pred_b, score_b, budget_k=budget_k)

    # 3. Model metrics
    metrics_m = compute_metrics(y_test, model_preds, model_scores, budget_k=budget_k)

    # 4. Scenario family breakdown for Model and Baseline B
    families = test_meta["scenario_family"].to_numpy()
    model_family_breakdown = compute_family_breakdown(y_test, model_preds, families)
    rule_b_family_breakdown = compute_family_breakdown(y_test, pred_b, families)

    # 5. Summary comparison table
    summary_table = pd.DataFrame([
        {
            "Method": "Baseline A (Raw Shortfall)",
            "Precision": round(metrics_a["precision"], 3),
            "Recall": round(metrics_a["recall"], 3),
            "F1-Score": round(metrics_a["f1"], 3),
            "PR-AUC": round(metrics_a["pr_auc"], 3),
            f"Precision@{budget_k}": round(metrics_a[f"precision_at_{budget_k}"], 3),
            "FPR on Controls": round(metrics_a["fpr_on_controls"], 3)
        },
        {
            "Method": "Baseline B (Derived Rules)",
            "Precision": round(metrics_b["precision"], 3),
            "Recall": round(metrics_b["recall"], 3),
            "F1-Score": round(metrics_b["f1"], 3),
            "PR-AUC": round(metrics_b["pr_auc"], 3),
            f"Precision@{budget_k}": round(metrics_b[f"precision_at_{budget_k}"], 3),
            "FPR on Controls": round(metrics_b["fpr_on_controls"], 3)
        },
        {
            "Method": "LAPE ML Model (HistGB)",
            "Precision": round(metrics_m["precision"], 3),
            "Recall": round(metrics_m["recall"], 3),
            "F1-Score": round(metrics_m["f1"], 3),
            "PR-AUC": round(metrics_m["pr_auc"], 3),
            f"Precision@{budget_k}": round(metrics_m[f"precision_at_{budget_k}"], 3),
            "FPR on Controls": round(metrics_m["fpr_on_controls"], 3)
        }
    ])

    return {
        "baseline_a": metrics_a,
        "baseline_b": metrics_b,
        "ml_model": metrics_m,
        "comparison_table": summary_table.to_dict(orient="records"),
        "model_family_breakdown": model_family_breakdown,
        "rule_b_family_breakdown": rule_b_family_breakdown
    }
