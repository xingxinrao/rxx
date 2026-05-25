"""AI-HR Star 简历智慧分析平台 — Streamlit 主入口"""

from __future__ import annotations

import json
from typing import Any

import matplotlib.pyplot as plt
import streamlit as st

from src.ai_agent import run_ai_agent
from src.config import init_deepseek_client
from src.database import get_all_evaluations, get_evaluation_by_id, init_db, save_evaluation
from src.pdf_utils import extract_text_from_pdf, validate_pdf
from src.privacy import mask_privacy, sanitize_input
from src.visualization import close_figure, draw_radar, render_items

APP_TITLE = "AI-HR Star 简历智慧分析平台"
DEFAULT_JD = (
    "岗位：AI-HR Star。要求：211/985背景，熟悉Python/SQL，"
    "具备AI工具应用和数据分析能力。"
)


# ==========================================
# 页面与样式初始化
# ==========================================
st.set_page_config(page_title="腾讯 AI-HR 智慧分析平台", layout="wide", page_icon="🚢")
plt.rcParams["font.sans-serif"] = ["SimHei", "Microsoft YaHei", "Arial Unicode MS"]
plt.rcParams["axes.unicode_minus"] = False


@st.cache_resource(show_spinner=False)
def get_ai_client():
    return init_deepseek_client()


client = get_ai_client()
init_db()


def _init_state() -> None:
    st.session_state.setdefault("final_report", None)
    st.session_state.setdefault("last_filename", "")
    st.session_state.setdefault("_total_tokens", 0)


def _composite_score(report: dict[str, Any]) -> int:
    keys = ["skill_score", "project_score", "edu_score", "logic_score", "tencent_fit"]
    scores = [int(report.get(key, 60)) for key in keys]
    return round(sum(scores) / len(scores))


def _process_resume(uploaded_file, jd_text: str) -> dict | None:
    valid, err_msg = validate_pdf(uploaded_file)
    if not valid:
        st.error(err_msg)
        return None

    with st.spinner("正在提取 PDF 文本..."):
        try:
            raw_text = extract_text_from_pdf(uploaded_file)
        except RuntimeError as exc:
            st.error(str(exc))
            return None

    if not raw_text.strip():
        st.error("未能从 PDF 中提取到有效文本，请检查文件是否为可解析的简历。")
        return None

    with st.spinner("正在脱敏与清洗文本..."):
        safe_text = sanitize_input(mask_privacy(raw_text))

    with st.spinner("正在调用 AI 进行评估..."):
        result = run_ai_agent(safe_text, jd_text.strip(), client)

    if result is None:
        st.error("AI 分析失败，请检查网络连接或 API Key 配置后重试。")
        return None

    save_evaluation(uploaded_file.name, jd_text, result)
    st.session_state.final_report = result
    st.session_state.last_filename = uploaded_file.name
    return result


def _render_header() -> None:
    st.title(f"🚢 {APP_TITLE}")
    st.caption("基于 DeepSeek-V3 的简历初筛与胜任力评估")
    st.markdown("---")


def _render_history_panel() -> None:
    with st.expander("📚 最近分析记录", expanded=False):
        history = get_all_evaluations(limit=5)
        if not history:
            st.info("当前还没有历史记录，完成一次分析后会自动保存。")
            return

        options = {f"#{row['id']} · {row['resume_filename']} · {row['verdict']}": row['id'] for row in history}
        selected_label = st.selectbox("选择一条记录查看详情", list(options.keys()))
        record = get_evaluation_by_id(options[selected_label])
        if not record:
            st.warning("未能加载该记录详情。")
            return

        cols = st.columns(3)
        cols[0].metric("综合画像", record.get("portrait", "—"))
        cols[1].metric("评估结论", record.get("verdict", "—"))
        cols[2].metric("技能分", f"{record.get('skill_score', 0)} / 100")
        st.caption(f"文件：{record.get('resume_filename', '—')} | 时间：{record.get('created_at', '—')}")
        st.write("**关键优势**")
        render_items(record.get("highlights", []))
        st.write("**潜在风险**")
        render_items(record.get("risks", []))


# ==========================================
# 主 UI
# ==========================================
_init_state()
_render_header()

left_col, right_col = st.columns([1, 2], gap="large")

with left_col:
    st.subheader("⚙️ 评估配置")
    st.caption("建议填写完整 JD，以获得更稳定的分数解释。")
    jd_area = st.text_area(
        "岗位描述 (JD)",
        value=DEFAULT_JD,
        height=200,
        placeholder="请输入岗位职责、任职要求、优先条件等内容...",
    )

    uploaded_file = st.file_uploader("上传简历（PDF 格式）", type=["pdf"])

    file_name = uploaded_file.name if uploaded_file else "未上传"
    st.caption(f"当前文件：{file_name}")

    can_submit = bool(uploaded_file and jd_area.strip())
    analyze = st.button(
        "开始 AI 自动化初筛",
        type="primary",
        use_container_width=True,
        disabled=not can_submit,
    )

    if analyze:
        _process_resume(uploaded_file, jd_area)

    if st.session_state.last_filename:
        st.success(f"最近一次分析文件：{st.session_state.last_filename}")

with right_col:
    report = st.session_state.get("final_report")
    if report:
        top_left, top_right = st.columns([1, 1], gap="large")
        with top_left:
            st.subheader("📊 胜任力雷达图")
            fig = draw_radar(report)
            st.pyplot(fig, clear_figure=False)
            close_figure(fig)

        with top_right:
            st.subheader("🎯 核心评价")
            composite = _composite_score(report)
            st.metric("综合匹配得分", f"{composite} / 100", delta="稳定输出模式")
            st.markdown(f"**候选人画像：** {report.get('portrait', '—')}")
            st.info(f"**最终建议：** {report.get('verdict', '—')}")
            st.caption(f"简历文件：{st.session_state.get('last_filename', '—')}")

        st.markdown("---")
        detail_left, detail_right = st.columns(2, gap="large")
        with detail_left:
            st.success("**✅ 关键优势 (Competitive Advantages)**")
            render_items(report.get("highlights", []))
        with detail_right:
            st.warning("**⚠️ 潜在风险 (Potential Risks)**")
            render_items(report.get("risks", []))

        st.download_button(
            label="💾 导出结构化 JSON 数据",
            data=json.dumps(report, ensure_ascii=False, indent=2),
            file_name="hr_analysis_export.json",
            mime="application/json",
            use_container_width=True,
        )
    else:
        st.info("👈 请在左侧输入 JD 并上传简历。系统会自动生成多维度评估报告。")

    _render_history_panel()

st.markdown("---")
st.caption(
    "🔒 隐私保护：系统内置 Regex 脱敏引擎。"
    "本地解析非结构化数据后，仅向云端传输抹除个人隐私标识后的文本，符合公司信息安全审计标准。"
    f" | 本次会话 Token 消耗：{st.session_state.get('_total_tokens', 0):,}"
    if st.session_state.get("_total_tokens")
    else "🔒 隐私保护：系统内置 Regex 脱敏引擎。本地解析非结构化数据后，仅向云端传输抹除个人隐私标识后的文本，符合公司信息安全审计标准。"
)
