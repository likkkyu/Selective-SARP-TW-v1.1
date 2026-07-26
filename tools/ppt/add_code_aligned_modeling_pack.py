#!/usr/bin/env python3
"""Append code-aligned mathematical modeling slide pack (3 slides) to report PPT.

This script validates key constants from current code (Config) and writes a revised PPT
with corrected modeling/objective/constraint/symbol content.
"""

import argparse
import math
import os
import shutil

from pptx import Presentation
from pptx.util import Inches, Pt
from pptx.dml.color import RGBColor

from problem_mcvrptw_v2 import Config


def add_textbox(slide, left, top, width, height, text, size=14, bold=False, color=(0, 0, 0), line_spacing=1.2):
    tb = slide.shapes.add_textbox(Inches(left), Inches(top), Inches(width), Inches(height))
    tf = tb.text_frame
    tf.clear()
    p = tf.paragraphs[0]
    run = p.add_run()
    run.text = text
    run.font.size = Pt(size)
    run.font.bold = bold
    run.font.color.rgb = RGBColor(*color)
    p.line_spacing = line_spacing
    return tb


def add_bullets(slide, left, top, width, height, lines, size=12, color=(35, 35, 35), bullet="• "):
    tb = slide.shapes.add_textbox(Inches(left), Inches(top), Inches(width), Inches(height))
    tf = tb.text_frame
    tf.clear()
    for i, line in enumerate(lines):
        p = tf.paragraphs[0] if i == 0 else tf.add_paragraph()
        p.text = f"{bullet}{line}"
        p.level = 0
        p.line_spacing = 1.2
        if p.runs:
            p.runs[0].font.size = Pt(size)
            p.runs[0].font.color.rgb = RGBColor(*color)
    return tb


def add_header(slide, title):
    hdr = slide.shapes.add_shape(1, Inches(0), Inches(0), Inches(13.333), Inches(0.55))
    hdr.fill.solid()
    hdr.fill.fore_color.rgb = RGBColor(220, 235, 247)
    hdr.line.fill.background()
    add_textbox(slide, 0.22, 0.07, 10.0, 0.40, title, size=23, bold=True, color=(31, 78, 121))


def add_footer(slide, text):
    add_textbox(slide, 0.42, 6.95, 12.3, 0.28, text, size=10, color=(90, 90, 90))


def make_slide_problem_objective(prs):
    slide = prs.slides.add_slide(prs.slide_layouts[6])
    add_header(slide, "数学建模与问题定义（代码修订版）")

    # block A
    b1 = slide.shapes.add_shape(1, Inches(0.35), Inches(0.72), Inches(12.62), Inches(1.55))
    b1.fill.solid(); b1.fill.fore_color.rgb = RGBColor(247, 249, 251); b1.line.color.rgb = RGBColor(210, 220, 230)
    add_textbox(slide, 0.5, 0.80, 4.0, 0.30, "1) 业务场景与建模范围", size=16, bold=True, color=(31, 78, 121))
    add_bullets(slide, 0.56, 1.08, 12.0, 1.08, [
        "平峰时段 10:00–16:00，静态批调度（day-ahead），订单为乘客+货物混合。",
        f"双舱车辆：乘客舱上限 Q^P={Config.PASSENGER_CAPACITY} 人，货物舱上限 Q^C={Config.CARGO_CAPACITY} 单位。",
        "目标是在硬约束可行域内最小化 RMB 总成本，并允许策略主动 reject。",
    ], size=12)

    # block B
    b2 = slide.shapes.add_shape(1, Inches(0.35), Inches(2.40), Inches(12.62), Inches(3.9))
    b2.fill.solid(); b2.fill.fore_color.rgb = RGBColor(255, 255, 255); b2.line.color.rgb = RGBColor(210, 220, 230)
    add_textbox(slide, 0.5, 2.48, 4.8, 0.30, "2) 评估目标函数（raw RMB）", size=16, bold=True, color=(31, 78, 121))

    formula = (
        "min J_raw = C_energy + C_p_delay + C_c_delay + C_vehicle + C_reject + C_unfulfilled + C_tripOT\n"
        "C_energy = p_e·Σ_t η(W_t)·d_t,   η(W)=0.18·(1 + W/10000),   p_e=1.0\n"
        f"C_p_delay = {Config.PASSENGER_DELAY_COST}·Σ_(i∈D^P) δ_i^del(min)\n"
        f"C_c_delay = Σ_(i∈D^C) g(δ_i^del),  g(x)=0.06·min(x,30)+{Config.CARGO_DELAY_COST_30_60:.2f}·min(max(x-30,0),30)+{Config.CARGO_DELAY_COST_60_PLUS:.2f}·max(x-60,0)\n"
        f"C_vehicle = {Config.VEHICLE_COST:.0f}·K_used,  C_reject = {Config.ALPHA_REJECT:.0f}·N_rej,  C_unfulfilled = {Config.ALPHA_UNFULFILLED:.0f}·N_unf,\n"
        f"C_tripOT = {Config.ALPHA_TRIP_OVERTIME:.0f}·Σ_k (T_k^trip - {Config.MAX_TRIP_TIME:.0f})^+"
    )
    add_textbox(slide, 0.58, 2.88, 12.1, 3.25, formula, size=12, line_spacing=1.24)

    add_footer(slide, "Corrected by code truth: problem_mcvrptw_v2.py::Config/get_costs")


def make_slide_constraints(prs):
    slide = prs.slides.add_slide(prs.slide_layouts[6])
    add_header(slide, "关键硬约束（代码级实现）")

    block = slide.shapes.add_shape(1, Inches(0.35), Inches(0.72), Inches(12.62), Inches(6.05))
    block.fill.solid(); block.fill.fore_color.rgb = RGBColor(247, 249, 251); block.line.color.rgb = RGBColor(210, 220, 230)

    add_textbox(slide, 0.5, 0.82, 7.5, 0.3, "Hard constraints are enforced by state mask (not pure MIP flow equations).", size=13, color=(70, 70, 70))

    left_lines = [
        "Precedence: delivery_i is feasible only if pickup_i has been served.",
        f"Dual capacity: q_t^P ≤ {Config.PASSENGER_CAPACITY}, q_t^C ≤ {Config.CARGO_CAPACITY}.",
        "Passenger pickup TW hard: a_{p_i} ≤ b_{p_i}.",
        f"Cargo pickup TW hard (current): {Config.HARD_CARGO_PICKUP_TIMEWINDOW}.",
        f"Passenger ride-time: ride_i ≤ {Config.PASSENGER_MAX_RIDE_TIME_MINUTES:.0f} min, excess_i ≤ {Config.PASSENGER_MAX_EXCESS_RIDE_TIME_MINUTES:.0f} min.",
    ]
    add_bullets(slide, 0.6, 1.24, 6.15, 2.75, left_lines, size=12)

    right_lines = [
        f"Trip bound: (a_j+τ_srv+t_{{j0}})-t_start^trip ≤ {Config.MAX_TRIP_TIME:.0f}h.",
        f"Operation end: a_j+τ_srv+t_{{j0}} ≤ {Config.OPERATION_END:.2f} (16:00).",
        "No-return-with-load: cannot return depot while carrying open orders.",
        "Fleet cap: K_used ≤ ceil(N/6).",
        "Reject action is enabled only with feasible candidate pickup; target is deterministic.",
    ]
    add_bullets(slide, 6.85, 1.24, 5.9, 2.75, right_lines, size=12)

    add_textbox(slide, 0.58, 4.20, 12.1, 2.35,
                "Implementation mapping:\n"
                "• state_mcvrptw_v2.py::get_mask handles precedence/capacity/TW/ride-time/trip/day/fleet/reject feasibility.\n"
                "• state_mcvrptw_v2.py::update applies reject state transition (pickup+delivery blocked).\n"
                "• attention_model uses mask to zero out invalid actions at each decode step.",
                size=12, color=(25, 25, 25), line_spacing=1.22)

    add_footer(slide, "Source: state_mcvrptw_v2.py::get_mask/update; nets/attention_model.py::_get_log_p")


def make_slide_symbol_table(prs):
    slide = prs.slides.add_slide(prs.slide_layouts[6])
    add_header(slide, "符号与参数表（当前代码口径）")

    rows = [
        ("Q^P", f"{Config.PASSENGER_CAPACITY}", "乘客舱容量（人）", "problem_mcvrptw_v2.py::Config"),
        ("Q^C", f"{Config.CARGO_CAPACITY}", "货物舱容量（单位）", "Config"),
        ("η(W)", "0.18·(1+W/10000)", "载重相关能耗系数（kWh/km）", "Config.ENERGY_*"),
        ("ω_p", f"{Config.PASSENGER_DELAY_COST}", "乘客 delivery 延误单价（元/min）", "Config.PASSENGER_DELAY_COST"),
        ("ω_c", "piecewise 0.06/0.18/0.36", "货物 delivery 迟到分段单价（元/min）", "Config.CARGO_DELAY_*"),
        ("ω_v", f"{Config.VEHICLE_COST:.0f}", "固定派车成本（元/车）", "Config.VEHICLE_COST"),
        ("ω_rej", f"{Config.ALPHA_REJECT:.0f}", "主动 reject 惩罚（元/单）", "Config.ALPHA_REJECT"),
        ("ω_unf", f"{Config.ALPHA_UNFULFILLED:.0f}", "未履约兜底惩罚（元/单）", "Config.ALPHA_UNFULFILLED"),
        ("ω_trip", f"{Config.ALPHA_TRIP_OVERTIME:.0f}", "单趟超时惩罚（元/h）", "Config.ALPHA_TRIP_OVERTIME"),
        ("τ_trip^max", f"{Config.MAX_TRIP_TIME:.0f} h", "单趟最大连续行程", "Config.MAX_TRIP_TIME"),
        ("Operation window", f"{Config.OPERATION_START:.0f}:00–{Config.OPERATION_END:.0f}:00", "运营时段硬约束", "Config.OPERATION_*"),
        ("K_max", "ceil(N/6)", "车队硬上限", "Config.DEFAULT_NUM_VEHICLE_RATIO"),
    ]

    n_rows = len(rows) + 1
    n_cols = 4
    t = slide.shapes.add_table(n_rows, n_cols, Inches(0.35), Inches(0.78), Inches(12.6), Inches(5.95)).table
    t.columns[0].width = Inches(1.8)
    t.columns[1].width = Inches(2.4)
    t.columns[2].width = Inches(4.4)
    t.columns[3].width = Inches(4.0)

    headers = ["符号", "当前值", "含义", "来源"]
    for j, h in enumerate(headers):
        cell = t.cell(0, j)
        cell.text = h
        run = cell.text_frame.paragraphs[0].runs[0]
        run.font.bold = True
        run.font.size = Pt(12)
        run.font.color.rgb = RGBColor(31, 78, 121)

    for i, (a, b, c, d) in enumerate(rows, start=1):
        t.cell(i, 0).text = a
        t.cell(i, 1).text = b
        t.cell(i, 2).text = c
        t.cell(i, 3).text = d
        for j in range(4):
            p = t.cell(i, j).text_frame.paragraphs[0]
            if p.runs:
                p.runs[0].font.size = Pt(11)

    note = (
        "Key corrections vs old slides: Q^P/Q^C=15/20 (not 20/100), η(W) denominator=10000 (not 1000), "
        "ω_rej=575 and includes ω_unf term in J_raw."
    )
    add_textbox(slide, 0.42, 6.84, 12.35, 0.34, note, size=10, color=(90, 90, 90))


def build(args):
    if not os.path.exists(args.input_ppt):
        raise FileNotFoundError(f"Input PPT not found: {args.input_ppt}")

    os.makedirs(os.path.dirname(args.output_ppt), exist_ok=True)
    shutil.copy2(args.input_ppt, args.output_ppt)

    prs = Presentation(args.output_ppt)
    make_slide_problem_objective(prs)
    make_slide_constraints(prs)
    make_slide_symbol_table(prs)

    prs.save(args.output_ppt)
    print(f"[done] wrote revised PPT: {args.output_ppt}")


def parse_args():
    p = argparse.ArgumentParser(description="Append corrected modeling slide pack")
    p.add_argument(
        "--input-ppt",
        default="/Users/bytedance/Downloads/2026年7月汇报_Selective_SARP_TW_阶段结果_v2_含参数表.pptx",
    )
    p.add_argument(
        "--output-ppt",
        default="/Users/bytedance/Downloads/2026年7月汇报_Selective_SARP_TW_阶段结果_v4_数学页修订版.pptx",
    )
    return p.parse_args()


if __name__ == "__main__":
    build(parse_args())
