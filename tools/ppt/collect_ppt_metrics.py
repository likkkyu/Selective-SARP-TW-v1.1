#!/usr/bin/env python3
"""
Collect presentation-ready metrics for current Selective SARP-TW results.

Inputs:
- outputs_cmp/service_delay_table_by_type.csv
- outputs_cmp/*eval500.json (selected files)
- outputs/review/n100_vehicle_summary_seed1234.csv
- outputs/gantt_charts/gantt_100_rw1_logs_academic.csv

Outputs (under outputs/ppt_assets):
- metrics_summary.json
- metrics_table.csv
- metrics_vehicle_n100.json
- metrics_gantt_n100.json
"""

import argparse
import csv
import json
import os
from statistics import mean


def _to_float(v):
    if v is None:
        return None
    s = str(v).strip()
    if s == "":
        return None
    try:
        return float(s)
    except ValueError:
        return None


def load_service_delay_table(path):
    rows = []
    with open(path, "r", encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            row_parsed = {
                "graph_size": int(row["graph_size"]),
                "source": row.get("source", ""),
                "checkpoint": row.get("checkpoint", ""),
                "passenger_orders_total": _to_float(row.get("passenger_orders_total")),
                "cargo_orders_total": _to_float(row.get("cargo_orders_total")),
                "passenger_served_total": _to_float(row.get("passenger_served_total")),
                "cargo_served_total": _to_float(row.get("cargo_served_total")),
                "passenger_late_total": _to_float(row.get("passenger_late_total")),
                "cargo_late_total": _to_float(row.get("cargo_late_total")),
                "passenger_service_rate": _to_float(row.get("passenger_service_rate")),
                "cargo_service_rate": _to_float(row.get("cargo_service_rate")),
                "passenger_delay_rate": _to_float(row.get("passenger_delay_rate")),
                "cargo_delay_rate": _to_float(row.get("cargo_delay_rate")),
                "overall_service_rate": _to_float(row.get("overall_service_rate")),
                "overall_delay_rate": _to_float(row.get("overall_delay_rate")),
                "note": row.get("note", ""),
            }
            rows.append(row_parsed)
    rows.sort(key=lambda x: x["graph_size"])
    return rows


def load_eval_json(path):
    with open(path, "r", encoding="utf-8") as f:
        payload = json.load(f)

    run = payload.get("run", {})
    agg = payload.get("aggregate", {})
    audit = payload.get("audit", {})
    biz = payload.get("business_acceptance", {})

    return {
        "path": path,
        "graph_size": int(run.get("graph_size")),
        "checkpoint": run.get("checkpoint", ""),
        "checkpoint_role": run.get("checkpoint_role", ""),
        "selection_rule": run.get("selection_rule", ""),
        "business_clean_run": bool(run.get("business_clean", False)),
        "service_rate_mean": _to_float(agg.get("service_rate_mean")),
        "unfulfilled_rate_mean": _to_float(agg.get("unfulfilled_rate_mean")),
        "rejected_rate_mean": _to_float(agg.get("rejected_rate_mean")),
        "total_cost_raw_mean": _to_float(agg.get("total_cost_raw_mean")),
        "cargo_delay_cost_mean": _to_float(agg.get("cargo_delay_cost_mean")),
        "used_vehicles_mean": _to_float(agg.get("used_vehicles_mean")),
        "partition_consistent_samples": int(audit.get("partition_consistent_samples", 0)),
        "core_aggregate_match_samples": int(audit.get("core_aggregate_match_samples", 0)),
        "total_samples": int(audit.get("total_samples", 0)),
        "business_clean_gate": bool(biz.get("clean", False)),
    }


def load_vehicle_summary(path):
    rows = []
    with open(path, "r", encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            rows.append(
                {
                    "vehicle_id": int(row["vehicle_id"]),
                    "serve_count": int(float(row["serve_count"])),
                    "pickup_count": int(float(row["pickup_count"])),
                    "delivery_count": int(float(row["delivery_count"])),
                    "reject_count": int(float(row["reject_count"])),
                    "late_count": int(float(row["late_count"])),
                    "max_late_min": _to_float(row.get("max_late_min")) or 0.0,
                }
            )
    return rows


def load_gantt_logs(path):
    rows = []
    with open(path, "r", encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            start_t = _to_float(row.get("start_time"))
            end_t = _to_float(row.get("end_time"))
            tw_end = _to_float(row.get("tw_end"))
            action = row.get("action", "")
            is_serve = action == "serve"
            is_late = bool(is_serve and tw_end is not None and end_t is not None and end_t > tw_end + 1e-9)
            rows.append(
                {
                    "vehicle_id": int(row["vehicle_id"]),
                    "action": action,
                    "order_kind": row.get("order_kind", ""),
                    "start_time": start_t,
                    "end_time": end_t,
                    "tw_end": tw_end,
                    "is_serve": is_serve,
                    "is_late": is_late,
                }
            )
    return rows


def summarize(service_rows, eval_rows, vehicle_rows, gantt_rows):
    eval_by_n = {r["graph_size"]: r for r in eval_rows}

    merged = []
    for row in service_rows:
        n = row["graph_size"]
        extra = eval_by_n.get(n, {})
        out = dict(row)
        out.update(
            {
                "eval_service_rate_mean": extra.get("service_rate_mean"),
                "eval_unfulfilled_rate_mean": extra.get("unfulfilled_rate_mean"),
                "eval_rejected_rate_mean": extra.get("rejected_rate_mean"),
                "eval_total_cost_raw_mean": extra.get("total_cost_raw_mean"),
                "eval_business_clean_run": extra.get("business_clean_run"),
                "eval_business_clean_gate": extra.get("business_clean_gate"),
                "eval_partition_consistent_samples": extra.get("partition_consistent_samples"),
                "eval_total_samples": extra.get("total_samples"),
            }
        )
        merged.append(out)

    # Strengths
    passenger_late_all_zero = all((r.get("passenger_late_total") or 0.0) == 0.0 for r in merged)
    best_service_row = max(merged, key=lambda r: r.get("overall_service_rate") or -1)

    # Risks
    cargo_delay_max_row = max(merged, key=lambda r: r.get("cargo_delay_rate") or -1)

    vehicle_used = [r for r in vehicle_rows if r["serve_count"] > 0]
    vehicle_late = [r for r in vehicle_rows if r["late_count"] > 0]

    serve_rows = [r for r in gantt_rows if r["is_serve"]]
    late_serve_rows = [r for r in serve_rows if r["is_late"]]

    summary = {
        "service_rows": merged,
        "strengths": {
            "passenger_late_all_zero": passenger_late_all_zero,
            "best_overall_service_graph_size": best_service_row["graph_size"],
            "best_overall_service_rate": best_service_row.get("overall_service_rate"),
        },
        "risks": {
            "max_cargo_delay_graph_size": cargo_delay_max_row["graph_size"],
            "max_cargo_delay_rate": cargo_delay_max_row.get("cargo_delay_rate"),
        },
        "vehicle_n100": {
            "vehicle_total": len(vehicle_rows),
            "vehicle_used": len(vehicle_used),
            "vehicle_late_any": len(vehicle_late),
            "late_count_mean_used": mean([r["late_count"] for r in vehicle_used]) if vehicle_used else 0.0,
            "max_late_min": max([r["max_late_min"] for r in vehicle_rows]) if vehicle_rows else 0.0,
        },
        "gantt_n100": {
            "vehicle_total": len(set(r["vehicle_id"] for r in gantt_rows)),
            "serve_event_count": len(serve_rows),
            "late_serve_event_count": len(late_serve_rows),
            "start_hour": min([r["start_time"] for r in gantt_rows if r["start_time"] is not None]) if gantt_rows else None,
            "end_hour": max([r["end_time"] for r in gantt_rows if r["end_time"] is not None]) if gantt_rows else None,
        },
    }
    return summary


def write_metrics_table(path, rows):
    fieldnames = [
        "graph_size",
        "source",
        "checkpoint",
        "passenger_orders_total",
        "cargo_orders_total",
        "passenger_served_total",
        "cargo_served_total",
        "passenger_late_total",
        "cargo_late_total",
        "passenger_service_rate",
        "cargo_service_rate",
        "passenger_delay_rate",
        "cargo_delay_rate",
        "overall_service_rate",
        "overall_delay_rate",
        "eval_unfulfilled_rate_mean",
        "eval_rejected_rate_mean",
        "eval_total_cost_raw_mean",
    ]
    with open(path, "w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow({k: row.get(k) for k in fieldnames})


def parse_args():
    p = argparse.ArgumentParser(description="Collect PPT metrics from existing artifacts")
    p.add_argument("--service-delay-csv", default="outputs_cmp/service_delay_table_by_type.csv")
    p.add_argument(
        "--eval-jsons",
        nargs="+",
        default=[
            "outputs_cmp/n25_ours_rw1_fixed_eval500.json",
            "outputs_cmp/n50_ours_rw1_eval500.json",
            "outputs_cmp/n100_ours_rw1_fixed_eval500.json",
            "outputs_cmp/n200_phaseC_rw1_e10_eval500.json",
        ],
    )
    p.add_argument("--vehicle-summary-csv", default="outputs/review/n100_vehicle_summary_seed1234.csv")
    p.add_argument("--gantt-logs-csv", default="outputs/gantt_charts/gantt_100_rw1_logs_academic.csv")
    p.add_argument("--out-dir", default="outputs/ppt_assets")
    return p.parse_args()


def main():
    args = parse_args()
    os.makedirs(args.out_dir, exist_ok=True)

    service_rows = load_service_delay_table(args.service_delay_csv)
    eval_rows = [load_eval_json(p) for p in args.eval_jsons if os.path.exists(p)]
    vehicle_rows = load_vehicle_summary(args.vehicle_summary_csv)
    gantt_rows = load_gantt_logs(args.gantt_logs_csv)

    summary = summarize(service_rows, eval_rows, vehicle_rows, gantt_rows)

    summary_path = os.path.join(args.out_dir, "metrics_summary.json")
    with open(summary_path, "w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)

    eval_path = os.path.join(args.out_dir, "metrics_eval500_curated.json")
    with open(eval_path, "w", encoding="utf-8") as f:
        json.dump(eval_rows, f, ensure_ascii=False, indent=2)

    table_path = os.path.join(args.out_dir, "metrics_table.csv")
    write_metrics_table(table_path, summary["service_rows"])

    vehicle_path = os.path.join(args.out_dir, "metrics_vehicle_n100.json")
    with open(vehicle_path, "w", encoding="utf-8") as f:
        json.dump(summary["vehicle_n100"], f, ensure_ascii=False, indent=2)

    gantt_path = os.path.join(args.out_dir, "metrics_gantt_n100.json")
    with open(gantt_path, "w", encoding="utf-8") as f:
        json.dump(summary["gantt_n100"], f, ensure_ascii=False, indent=2)

    print(f"[done] wrote {summary_path}")
    print(f"[done] wrote {table_path}")
    print(f"[done] wrote {eval_path}")
    print(f"[done] wrote {vehicle_path}")
    print(f"[done] wrote {gantt_path}")


if __name__ == "__main__":
    main()
