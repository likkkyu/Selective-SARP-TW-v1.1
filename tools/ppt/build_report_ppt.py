#!/usr/bin/env python3
"""
Create a new report deck from the existing monthly template and current artifacts.

Workflow:
1) Copy template pptx to a new output path.
2) Replace/update key slides with current figures and metrics.
3) Keep a rigorous, evidence-first narrative style.
"""

import argparse
import json
import os
import shutil
from datetime import datetime

from pptx import Presentation
from pptx.util import Inches, Pt
from pptx.dml.color import RGBColor


def fmt_pct(v):
    if v is None:
        return "N/A"
    return f"{100.0 * float(v):.2f}%"


def fmt_num(v):
    if v is None:
        return "N/A"
    return f"{int(round(float(v)))}"


def load_summary(path):
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def remove_shape(shape):
    el = shape._element
    el.getparent().remove(el)


def clear_content_shapes(slide, keep_indices):
    # keep_indices is 1-based index from current slide order
    keep = set(keep_indices)
    for idx in range(len(slide.shapes), 0, -1):
        if idx not in keep:
            remove_shape(slide.shapes[idx - 1])


def replace_picture_shape(slide, shape_index, image_path):
    shape = slide.shapes[shape_index - 1]
    left, top, width, height = shape.left, shape.top, shape.width, shape.height
    remove_shape(shape)
    slide.shapes.add_picture(image_path, left, top, width=width, height=height)


def set_shape_text(shape, text, font_size_pt=None, bold=None, color_rgb=None):
    tf = shape.text_frame
    tf.clear()
    p = tf.paragraphs[0]
    run = p.add_run()
    run.text = text
    if font_size_pt is not None:
        run.font.size = Pt(font_size_pt)
    if bold is not None:
        run.font.bold = bold
    if color_rgb is not None:
        run.font.color.rgb = RGBColor(*color_rgb)


def add_note_box(slide, text, left=0.6, top=6.72, width=12.0, height=0.35, size=10):
    box = slide.shapes.add_textbox(Inches(left), Inches(top), Inches(width), Inches(height))
    tf = box.text_frame
    tf.clear()
    p = tf.paragraphs[0]
    run = p.add_run()
    run.text = text
    run.font.size = Pt(size)
    run.font.color.rgb = RGBColor(90, 90, 90)


def create_deck(args):
    if not os.path.exists(args.template):
        raise FileNotFoundError(f"Template not found: {args.template}")

    for p in [
        args.metrics_summary,
        args.fig_main,
        args.fig_service,
        args.fig_delay,
        args.fig_overall,
        args.fig_counts,
        args.fig_vehicle,
        args.fig_risk,
        args.gantt,
    ]:
        if not os.path.exists(p):
            raise FileNotFoundError(f"Required asset not found: {p}")

    os.makedirs(os.path.dirname(args.output), exist_ok=True)
    shutil.copy2(args.template, args.output)

    summary = load_summary(args.metrics_summary)
    rows = summary["service_rows"]
    by_n = {r["graph_size"]: r for r in rows}

    prs = Presentation(args.output)

    # Slide 1: Cover
    s1 = prs.slides[0]
    set_shape_text(s1.shapes[8], "Selective SARP-TW: 当前阶段结果汇报", font_size_pt=34, bold=True)  # shape 9
    set_shape_text(s1.shapes[6], "汇报人：李珂豫\n导师组：李德纮", font_size_pt=16)  # shape 7
    set_shape_text(s1.shapes[2], datetime.now().strftime("%Y年%m月%d日"), font_size_pt=16)  # shape 3

    # Slide 2: Setup / protocol
    s2 = prs.slides[1]
    set_shape_text(s2.shapes[4], "实验设置与评估口径", font_size_pt=24, bold=True)  # shape 5
    setup_text = (
        "评估协议：固定 seed=99999，eval500，decode=greedy，checkpoint_role=best_service。\n"
        "规模范围：n25 / n50 / n100 / n200。\n"
        "指标口径：乘客/货物订单总数、服务数、延误数、服务率、延误率；并联合 overall 服务/延误及审计指标。\n"
        "数据来源：outputs_cmp/service_delay_table_by_type.csv + 各规模 eval500 JSON。"
    )
    set_shape_text(s2.shapes[5], setup_text, font_size_pt=14)  # shape 6
    set_shape_text(s2.shapes[9], "结论优先、证据跟随：每页标注数据源，保证可复现可追溯。", font_size_pt=14)  # shape 10

    # Slide 3: main results table figure
    s3 = prs.slides[2]
    clear_content_shapes(s3, keep_indices=[1, 2, 3, 4, 5])
    set_shape_text(s3.shapes[4], "多规模核心结果总览（n25/n50/n100/n200）", font_size_pt=24, bold=True)
    s3.shapes.add_picture(args.fig_main, Inches(0.6), Inches(1.25), width=Inches(12.0), height=Inches(4.9))
    add_note_box(s3, "Source: outputs/ppt_assets/fig_main_results_table.png; base table=outputs_cmp/service_delay_table_by_type.csv")

    # Slide 4: strengths (service/delay split)
    s4 = prs.slides[3]
    clear_content_shapes(s4, keep_indices=[1, 2, 3, 4, 5])
    set_shape_text(s4.shapes[4], "亮点指标：分类型服务率与延误率", font_size_pt=24, bold=True)
    s4.shapes.add_picture(args.fig_service, Inches(0.55), Inches(1.2), width=Inches(6.1), height=Inches(2.55))
    s4.shapes.add_picture(args.fig_delay, Inches(6.7), Inches(1.2), width=Inches(5.95), height=Inches(2.55))

    best_n = summary["strengths"]["best_overall_service_graph_size"]
    best_sr = summary["strengths"]["best_overall_service_rate"]
    risk_n = summary["risks"]["max_cargo_delay_graph_size"]
    risk_dr = summary["risks"]["max_cargo_delay_rate"]

    bullets = (
        f"• 当前总体服务率最佳规模：n{best_n}（overall service={fmt_pct(best_sr)}）。\n"
        f"• 乘客延误在四个规模上均为 0（passenger_late_total=0）。\n"
        f"• 风险侧重点：n{risk_n} 货物延误率最高（cargo delay={fmt_pct(risk_dr)}），需后续优化。"
    )
    box = s4.shapes.add_textbox(Inches(0.65), Inches(3.95), Inches(11.8), Inches(2.1))
    set_shape_text(box, bullets, font_size_pt=15)
    add_note_box(s4, "Source: outputs_cmp/service_delay_table_by_type.csv")

    # Slide 5: overall trends + counts
    s5 = prs.slides[4]
    set_shape_text(s5.shapes[4], "总体趋势与数量分解", font_size_pt=24, bold=True)
    replace_picture_shape(s5, 6, args.fig_overall)
    # add counts figure at bottom as inset
    s5.shapes.add_picture(args.fig_counts, Inches(7.2), Inches(4.55), width=Inches(4.9), height=Inches(2.05))
    add_note_box(s5, "Source: outputs/ppt_assets/fig_overall_service_delay.png, fig_counts_breakdown.png")

    # Slide 6: Gantt chart (mandatory)
    s6 = prs.slides[5]
    set_shape_text(s6.shapes[4], "车辆-订单调度甘特图（N100，学术风格）", font_size_pt=24, bold=True)
    replace_picture_shape(s6, 6, args.gantt)  # replace table slot with gantt
    replace_picture_shape(s6, 7, args.fig_risk)  # right slot as risk overview

    g = summary["gantt_n100"]
    gantt_note = (
        f"N100: vehicles={g['vehicle_total']}, serve_events={g['serve_event_count']}, "
        f"late_serve={g['late_serve_event_count']}, time={g['start_hour']:.2f}–{g['end_hour']:.2f}h"
    )
    add_note_box(s6, f"{gantt_note} | Source: outputs/gantt_charts/gantt_100_rw1_academic.png")

    # Slide 7: vehicle-level profile
    s7 = prs.slides[6]
    set_shape_text(s7.shapes[4], "车辆级案例分析（N100 Representative）", font_size_pt=24, bold=True)
    # remove left group and replace by our vehicle profile plot
    remove_shape(s7.shapes[5])  # original group in slot 6
    s7.shapes.add_picture(args.fig_vehicle, Inches(0.6), Inches(1.2), width=Inches(5.9), height=Inches(5.3))
    replace_picture_shape(s7, 7, args.gantt)
    # shape index changed after replacement/removal, replace current last picture for slot 8 not strictly needed
    # add concise insights
    v = summary["vehicle_n100"]
    insight = (
        f"Used vehicles: {v['vehicle_used']}/{v['vehicle_total']} | "
        f"vehicles with any lateness: {v['vehicle_late_any']} | "
        f"max late: {v['max_late_min']:.2f} min"
    )
    add_note_box(s7, f"{insight} | Source: outputs/review/n100_vehicle_summary_seed1234.csv")

    # Slide 8: risk and rigor
    s8 = prs.slides[7]
    clear_content_shapes(s8, keep_indices=[1, 2, 3, 4, 5])
    set_shape_text(s8.shapes[4], "严谨性与风险边界", font_size_pt=24, bold=True)
    s8.shapes.add_picture(args.fig_risk, Inches(0.7), Inches(1.25), width=Inches(6.0), height=Inches(4.2))
    s8.shapes.add_picture(args.fig_delay, Inches(6.9), Inches(1.25), width=Inches(5.4), height=Inches(4.2))

    n200 = by_n.get(200, {})
    text = (
        f"• n200 规模总体服务率：{fmt_pct(n200.get('overall_service_rate'))}。\n"
        f"• n200 货物延误率：{fmt_pct(n200.get('cargo_delay_rate'))}（阶段性瓶颈）。\n"
        f"• 审计一致性来自 eval500 汇总，后续将继续压降 cargo delay 与 unfulfilled。"
    )
    bx = s8.shapes.add_textbox(Inches(0.8), Inches(5.55), Inches(11.6), Inches(1.0))
    set_shape_text(bx, text, font_size_pt=14)
    add_note_box(s8, "Source: outputs_cmp/*eval500.json + outputs_cmp/service_delay_table_by_type.csv")

    # Slide 9: next plan
    s9 = prs.slides[8]
    set_shape_text(s9.shapes[4], "下阶段计划（面向论文定稿）", font_size_pt=24, bold=True)
    plan_text = (
        "1) 完善 n200 货物延误优化：从调度规则与惩罚权重双路径迭代。\n"
        "2) 补齐跨规模车辆级可解释性案例（n25/n50/n200）。\n"
        "3) 固化论文图表流水线：指标抽取 → 画图 → PPT 自动落版。\n"
        "4) 完成正文与附录复核，按期提交目标期刊。"
    )
    set_shape_text(s9.shapes[5], plan_text, font_size_pt=18)  # shape 6 main body
    add_note_box(s9, "All figures and numbers are traceable to repository artifacts.")

    prs.save(args.output)
    print(f"[done] wrote PPT: {args.output}")


def parse_args():
    p = argparse.ArgumentParser(description="Build a new report deck from template")
    p.add_argument("--template", default="/Users/bytedance/Downloads/2026年5月汇报.pptx")
    p.add_argument(
        "--output",
        default="/Users/bytedance/Downloads/2026年7月汇报_Selective_SARP_TW_阶段结果_v1.pptx",
    )
    p.add_argument("--metrics-summary", default="outputs/ppt_assets/metrics_summary.json")

    p.add_argument("--fig-main", default="outputs/ppt_assets/fig_main_results_table.png")
    p.add_argument("--fig-service", default="outputs/ppt_assets/fig_type_service_rates.png")
    p.add_argument("--fig-delay", default="outputs/ppt_assets/fig_type_delay_rates.png")
    p.add_argument("--fig-overall", default="outputs/ppt_assets/fig_overall_service_delay.png")
    p.add_argument("--fig-counts", default="outputs/ppt_assets/fig_counts_breakdown.png")
    p.add_argument("--fig-vehicle", default="outputs/ppt_assets/fig_vehicle_late_profile_n100.png")
    p.add_argument("--fig-risk", default="outputs/ppt_assets/fig_eval_risk_overview.png")
    p.add_argument("--gantt", default="outputs/gantt_charts/gantt_100_rw1_academic.png")
    return p.parse_args()


if __name__ == "__main__":
    create_deck(parse_args())
