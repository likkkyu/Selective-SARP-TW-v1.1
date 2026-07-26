#!/usr/bin/env python3
"""Append a parameter-meaning table slide to an existing report PPT."""

import argparse
import json
import os
import shutil
from pptx import Presentation
from pptx.util import Pt, Inches
from pptx.dml.color import RGBColor


PARAM_DESCRIPTIONS = {
    "graph_sizes": "实验规模（订单数 N）",
    "num_samples": "评估样本数（每个规模）",
    "batch_size": "推理批大小",
    "seed": "随机种子（保证可复现）",
    "decode": "解码策略",
    "checkpoint_role": "模型选择角色（按服务率优先）",
    "selection_rule": "模型筛选规则",
    "max_concurrent_open_orders": "单车最大并行在途订单数",
    "min_orders_per_dispatch": "单次发车最小订单触发阈值",
    "enable_delivery_viability": "启用送达可行性约束",
    "enable_viability_fallback": "可行性回退机制",
    "relax_pickup_commitment_trip_time": "放宽接载承诺行程时长约束",
    "decode_pickup_urgency_bias": "接载紧迫性偏置",
    "decode_pickup_urgency_horizon_hours": "接载紧迫性时间窗（小时）",
    "operation_window": "运营时间窗",
}


def _fmt(v):
    if isinstance(v, bool):
        return "True" if v else "False"
    return str(v)


def add_parameter_table_slide(ppt_path, output_path, eval_json_paths):
    if not os.path.exists(ppt_path):
        raise FileNotFoundError(f"PPT not found: {ppt_path}")

    eval_payloads = []
    for p in eval_json_paths:
        if os.path.exists(p):
            with open(p, "r", encoding="utf-8") as f:
                eval_payloads.append(json.load(f))

    if not eval_payloads:
        raise FileNotFoundError("No eval JSON found for parameter extraction")

    # Use first payload as canonical (configs are aligned); collect graph sizes across all.
    base_run = eval_payloads[0].get("run", {})
    state_kwargs = base_run.get("state_kwargs", {})
    graph_sizes = sorted({p.get("run", {}).get("graph_size") for p in eval_payloads if p.get("run", {}).get("graph_size") is not None})

    rows = [
        ("graph_sizes", f"{graph_sizes}", "n25/n50/n100/n200 跨规模对比"),
        ("num_samples", _fmt(base_run.get("num_samples", "N/A")), "eval500"),
        ("batch_size", _fmt(base_run.get("batch_size", "N/A")), "推理吞吐与显存折中"),
        ("seed", _fmt(base_run.get("seed", "N/A")), "固定随机性"),
        ("decode", _fmt(base_run.get("decode", "N/A")), "通常 greedy"),
        ("checkpoint_role", _fmt(base_run.get("checkpoint_role", "N/A")), "best_service"),
        ("selection_rule", _fmt(base_run.get("selection_rule", "N/A")), "service_priority"),
        ("max_concurrent_open_orders", _fmt(state_kwargs.get("max_concurrent_open_orders", "N/A")), "控制并发服务复杂度"),
        ("min_orders_per_dispatch", _fmt(state_kwargs.get("min_orders_per_dispatch", "N/A")), "发车最低订单门槛"),
        ("enable_delivery_viability", _fmt(state_kwargs.get("enable_delivery_viability", "N/A")), "确保可送达"),
        ("enable_viability_fallback", _fmt(state_kwargs.get("enable_viability_fallback", "N/A")), "异常回退"),
        ("relax_pickup_commitment_trip_time", _fmt(state_kwargs.get("relax_pickup_commitment_trip_time", "N/A")), "是否放宽承诺时长"),
        ("decode_pickup_urgency_bias", _fmt(base_run.get("decode_pickup_urgency_bias", "N/A")), "紧迫性偏置"),
        ("decode_pickup_urgency_horizon_hours", _fmt(base_run.get("decode_pickup_urgency_horizon_hours", "N/A")), "紧迫窗口"),
        ("operation_window", "10:00–16:00", "平峰运营时段"),
    ]

    prs = Presentation(ppt_path)
    # Try title+content layout; fallback to blank
    layout = prs.slide_layouts[1] if len(prs.slide_layouts) > 1 else prs.slide_layouts[0]
    slide = prs.slides.add_slide(layout)

    # title
    if slide.shapes.title:
        slide.shapes.title.text = "实验参数含义与当前取值"
        tf = slide.shapes.title.text_frame
        if tf.paragraphs and tf.paragraphs[0].runs:
            run = tf.paragraphs[0].runs[0]
            run.font.size = Pt(28)
            run.font.bold = True
            run.font.color.rgb = RGBColor(31, 78, 121)
    else:
        tb = slide.shapes.add_textbox(Inches(0.6), Inches(0.2), Inches(12), Inches(0.6))
        tb.text_frame.text = "实验参数含义与当前取值"

    # remove default content placeholder text if any
    for shp in list(slide.shapes):
        if shp.is_placeholder and shp.placeholder_format.idx != 0 and shp.has_text_frame:
            shp.text_frame.clear()

    # table
    n_rows = len(rows) + 1
    n_cols = 4
    table_shape = slide.shapes.add_table(n_rows, n_cols, Inches(0.35), Inches(1.0), Inches(12.6), Inches(5.8))
    table = table_shape.table

    table.columns[0].width = Inches(2.9)
    table.columns[1].width = Inches(3.7)
    table.columns[2].width = Inches(2.0)
    table.columns[3].width = Inches(4.0)

    headers = ["参数名", "含义", "当前取值", "备注"]
    for j, h in enumerate(headers):
        cell = table.cell(0, j)
        cell.text = h
        p = cell.text_frame.paragraphs[0]
        p.runs[0].font.bold = True
        p.runs[0].font.size = Pt(13)
        p.runs[0].font.color.rgb = RGBColor(31, 78, 121)

    for i, (k, v, note) in enumerate(rows, start=1):
        table.cell(i, 0).text = k
        table.cell(i, 1).text = PARAM_DESCRIPTIONS.get(k, "")
        table.cell(i, 2).text = v
        table.cell(i, 3).text = note
        for j in range(4):
            p = table.cell(i, j).text_frame.paragraphs[0]
            if p.runs:
                p.runs[0].font.size = Pt(11)

    # footer note
    footer = slide.shapes.add_textbox(Inches(0.45), Inches(6.75), Inches(12.3), Inches(0.35))
    footer.text_frame.text = "Source: outputs_cmp/*eval500.json (run/state_kwargs), service_delay_table_by_type.csv"
    run = footer.text_frame.paragraphs[0].runs[0]
    run.font.size = Pt(10)
    run.font.color.rgb = RGBColor(90, 90, 90)

    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    prs.save(output_path)
    print(f"[done] wrote PPT: {output_path}")


def parse_args():
    p = argparse.ArgumentParser(description="Append parameter table slide")
    p.add_argument("--ppt", required=True, help="Input PPT path")
    p.add_argument("--output", required=True, help="Output PPT path")
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
    return p.parse_args()


if __name__ == "__main__":
    args = parse_args()
    add_parameter_table_slide(args.ppt, args.output, args.eval_jsons)
