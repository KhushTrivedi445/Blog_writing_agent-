from pathlib import Path
from datetime import date
import re

import streamlit as st
import streamlit.components.v1 as components

from backend import app


# ============================================================
# PAGE CONFIG
# ============================================================

st.set_page_config(
    page_title="AI Blog Writer",
    page_icon="✍️",
    layout="wide",
    initial_sidebar_state="expanded",
)


# ============================================================
# SESSION STATE
# ============================================================

if "blog" not in st.session_state:
    st.session_state.blog = ""

if "plan" not in st.session_state:
    st.session_state.plan = None

if "image_plan" not in st.session_state:
    st.session_state.image_plan = None

if "generated_topic" not in st.session_state:
    st.session_state.generated_topic = ""

if "generated_date" not in st.session_state:
    st.session_state.generated_date = ""


# ============================================================
# HELPERS
# ============================================================

def safe_filename(text: str) -> str:
    text = re.sub(
        r"[^a-zA-Z0-9 _-]",
        "",
        text,
    ).strip()

    text = text.lower().replace(
        " ",
        "_",
    )

    return text or "ai_generated_blog"


def extract_title(markdown_text: str) -> str:

    if not markdown_text:
        return "AI Generated Blog"

    for line in markdown_text.splitlines():

        line = line.strip()

        if line.startswith("# "):
            return line[2:].strip()

    return "AI Generated Blog"


def resolve_image_path(image_path: str):

    if not image_path:
        return None

    path = Path(image_path)

    if path.is_absolute() and path.exists():
        return path

    if path.exists():
        return path

    project_path = (
        Path(__file__).resolve().parent / path
    )

    if project_path.exists():
        return project_path

    return None


def render_svg(image_path: str):

    resolved = resolve_image_path(
        image_path
    )

    if resolved is None:
        st.warning(
            "Diagram file was not found."
        )
        return

    try:

        svg_content = resolved.read_text(
            encoding="utf-8"
        )

        components.html(
            svg_content,
            height=530,
            scrolling=False,
        )

    except Exception as e:

        st.warning(
            f"Could not display diagram: {e}"
        )


# ============================================================
# SIDEBAR
# ============================================================

with st.sidebar:

    st.title(
        "✍️ AI Blog Writer"
    )

    topic = st.text_area(
        "Blog topic",
        placeholder="e.g. Machine Learning",
        height=120,
    )

    as_of_date = st.date_input(
        "As-of date",
        value=date.today(),
    )

    generate = st.button(
        "🚀 Generate Blog",
        width="stretch",
        type="primary",
    )

    st.divider()

    st.subheader(
        "📚 Previous Blogs"
    )

    blog_files = sorted(
        Path(".").glob("*.md"),
        key=lambda p: p.stat().st_mtime,
        reverse=True,
    )

    if blog_files:

        selected_blog = st.selectbox(
            "Select blog",
            blog_files,
            format_func=lambda p: p.stem.replace(
                "_",
                " ",
            ).title(),
        )

        open_blog = st.button(
            "📖 Open Previous Blog",
            width="stretch",
        )

        if open_blog:

            try:

                st.session_state.blog = (
                    selected_blog.read_text(
                        encoding="utf-8"
                    )
                )

                st.session_state.plan = None
                st.session_state.image_plan = None
                st.session_state.generated_topic = ""
                st.session_state.generated_date = ""

                st.rerun()

            except Exception as e:

                st.error(
                    f"Could not open blog: {e}"
                )

    else:

        st.caption(
            "Generated blogs will appear here."
        )


# ============================================================
# MAIN HEADER
# ============================================================

st.title(
    "✍️ Agentic AI Blog Writer"
)

st.caption(
    "LangGraph • Groq • Tavily • Streamlit"
)


# ============================================================
# GENERATE BLOG
# ============================================================

if generate:

    cleaned_topic = topic.strip()

    if not cleaned_topic:

        st.error(
            "Please enter a blog topic."
        )

    else:

        # Clear previous results
        st.session_state.blog = ""
        st.session_state.plan = None
        st.session_state.image_plan = None

        st.session_state.generated_topic = (
            cleaned_topic
        )

        st.session_state.generated_date = (
            str(as_of_date)
        )

        # ====================================================
        # LIVE AGENT PROCESS
        # ====================================================

        with st.status(
            "🚀 Running Agentic Blog Writer...",
            expanded=True,
        ) as process:

            try:

                result = {}

                research_count = 0

                total_research = 5

                # IMPORTANT:
                # Stream the graph ONCE.
                for update in app.stream(
                    {
                        "topic": cleaned_topic,
                        "needs_research": True,
                        "queries": [],
                        "evidence": [],
                        "sections": [],
                        "merged_md": "",
                        "final": "",
                        "image_plan": None,
                    },
                    stream_mode="updates",
                ):

                    if not isinstance(update, dict):
                        continue

                    for node_name, node_data in update.items():

                        # ------------------------------------
                        # PLAN
                        # ------------------------------------

                        if node_name == "orchestrator":

                            if isinstance(
                                node_data,
                                dict,
                            ):

                                if node_data.get("plan"):

                                    result["plan"] = (
                                        node_data["plan"]
                                    )

                                    plan = (
                                        node_data["plan"]
                                    )

                                    st.write(
                                        "✅ orchestrator completed"
                                    )

                                    st.write(
                                        f"🧩 Plan created: "
                                        f"{plan.blog_title}"
                                    )

                                    st.write(
                                        f"📚 Sections: "
                                        f"{len(plan.tasks)}"
                                    )

                        # ------------------------------------
                        # RESEARCH
                        # ------------------------------------

                        elif node_name == "research":

                            research_count += 1

                            st.write(
                                "✅ research completed "
                                f"({research_count}/{total_research})"
                            )

                        # ------------------------------------
                        # WRITING
                        # ------------------------------------

                        elif node_name == "worker":

                            st.write(
                                "✅ blog sections completed"
                            )

                        # ------------------------------------
                        # REDUCER
                        # ------------------------------------

                        elif node_name == "reducer":

                            st.write(
                                "✅ blog combined"
                            )

                            if isinstance(
                                node_data,
                                dict,
                            ):

                                if node_data.get(
                                    "final"
                                ):

                                    result["final"] = (
                                        node_data["final"]
                                    )

                                if node_data.get(
                                    "image_plan"
                                ):

                                    result["image_plan"] = (
                                        node_data[
                                            "image_plan"
                                        ]
                                    )

                # --------------------------------------------
                # Save final results
                # --------------------------------------------

                st.session_state.plan = result.get(
                    "plan"
                )

                st.session_state.blog = result.get(
                    "final",
                    "",
                )

                st.session_state.image_plan = result.get(
                    "image_plan"
                )

                process.update(
                    label="✅ Blog generated successfully!",
                    state="complete",
                )

            except Exception as e:

                process.update(
                    label="❌ Blog generation failed",
                    state="error",
                )

                st.error(
                    f"Generation failed: {e}"
                )


# ============================================================
# RESULTS
# ============================================================

plan = st.session_state.plan
blog = st.session_state.blog
image_plan = st.session_state.image_plan


# ============================================================
# TABS
# ============================================================

tab_plan, tab_blog = st.tabs(
    [
        "🧩 Plan",
        "📄 Blog",
    ]
)


# ============================================================
# PLAN TAB
# ============================================================

with tab_plan:

    st.subheader(
        "🧩 Blog Plan"
    )

    if plan:

        # -----------------------------------------------
        # Main plan information
        # -----------------------------------------------

        st.markdown(
            f"### {plan.blog_title}"
        )

        st.write(
            f"**Audience:** {plan.audience}"
        )

        st.write(
            f"**Tone:** {plan.tone}"
        )

        st.divider()

        # -----------------------------------------------
        # Every task
        # -----------------------------------------------

        for index, task in enumerate(
            plan.tasks,
            start=1,
        ):

            st.markdown(
                f"### {index}. {task.title}"
            )

            st.write(
                f"**Goal:** {task.goal}"
            )

            st.write(
                "**Key points:**"
            )

            for bullet in task.bullets:

                st.markdown(
                    f"- {bullet}"
                )

            if index < len(plan.tasks):

                st.divider()

    else:

        st.info(
            "Generate a blog to see the complete plan."
        )


# ============================================================
# BLOG TAB
# ============================================================

with tab_blog:

    st.subheader(
        "📄 Generated Blog"
    )

    if blog:

        title = extract_title(
            blog
        )

        # -----------------------------------------------
        # Metadata
        # -----------------------------------------------

        if st.session_state.generated_topic:

            st.caption(
                f"Topic: "
                f"{st.session_state.generated_topic}"
            )

            st.caption(
                f"As-of date: "
                f"{st.session_state.generated_date}"
            )

            st.divider()

        # -----------------------------------------------
        # MARKDOWN FIRST
        # -----------------------------------------------

        st.markdown(
            blog
        )

        # -----------------------------------------------
        # DOWNLOAD BUTTON
        # -----------------------------------------------

        st.divider()

        filename = (
            safe_filename(title)
            + ".md"
        )

        st.download_button(
            "⬇️ Download Markdown",
            data=blog,
            file_name=filename,
            mime="text/markdown",
            width="stretch",
        )

        # -----------------------------------------------
        # IMAGE AFTER MARKDOWN
        # -----------------------------------------------

        if image_plan:

            images = image_plan.get(
                "images",
                [],
            )

            if images:

                image_info = images[0]

                image_path = image_info.get(
                    "path",
                    "",
                )

                st.divider()

                st.subheader(
                    "🖼️ Blog Diagram"
                )

                render_svg(
                    image_path
                )

                caption = image_info.get(
                    "caption",
                    "Minimal conceptual diagram.",
                )

                st.caption(
                    caption
                )

    else:

        st.info(
            "Generate a blog to see the final article."
        )