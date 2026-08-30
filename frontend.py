import json
import re
from pathlib import Path
from datetime import date

import pandas as pd
import streamlit as st

from backend import app


# ============================================================
# CONFIG
# ============================================================

st.set_page_config(
    page_title="AI Blog Writer",
    page_icon="✍️",
    layout="wide"
)


# ============================================================
# HELPERS
# ============================================================

def safe_slug(title):

    title = re.sub(
        r"[^a-zA-Z0-9 _-]",
        "",
        title
    )

    return (
        title.strip()
        .lower()
        .replace(" ", "_")
        or "blog"
    )


def get_title(out):

    plan = out.get("plan")

    if hasattr(plan, "blog_title"):
        return plan.blog_title

    if isinstance(plan, dict):
        return plan.get(
            "blog_title",
            "blog"
        )

    return "blog"


def get_plan_dict(plan):

    if hasattr(plan, "model_dump"):
        return plan.model_dump()

    if isinstance(plan, dict):
        return plan

    return {}


def list_blogs():

    files = list(
        Path(".").glob("*.md")
    )

    return sorted(
        files,
        key=lambda x: x.stat().st_mtime,
        reverse=True
    )


# ============================================================
# SESSION STATE
# ============================================================

if "last_out" not in st.session_state:
    st.session_state["last_out"] = None


# ============================================================
# SIDEBAR
# ============================================================

st.sidebar.title("✍️ AI Blog Writer")

topic = st.sidebar.text_area(
    "Blog topic",
    placeholder="e.g. Self Attention in Transformers",
    height=120
)

as_of = st.sidebar.date_input(
    "As-of date",
    value=date.today()
)

generate = st.sidebar.button(
    "🚀 Generate Blog",
    type="primary",
    use_container_width=True
)


# ============================================================
# PAST BLOGS
# ============================================================

st.sidebar.divider()

st.sidebar.subheader(
    "📚 Previous Blogs"
)

blogs = list_blogs()

if blogs:

    selected = st.sidebar.selectbox(
        "Select blog",
        blogs,
        format_func=lambda x: x.stem
    )

    if st.sidebar.button(
        "📂 Load Blog",
        use_container_width=True
    ):

        content = selected.read_text(
            encoding="utf-8"
        )

        st.session_state["last_out"] = {
            "plan": None,
            "evidence": [],
            "final": content
        }

else:

    st.sidebar.caption(
        "No previous blogs found."
    )


# ============================================================
# HEADER
# ============================================================

st.title("✍️ Agentic AI Blog Writer")

st.caption(
    "LangGraph • Groq • Tavily • Streamlit"
)


# ============================================================
# GENERATE
# ============================================================

if generate:

    if not topic.strip():

        st.warning(
            "Please enter a blog topic."
        )

        st.stop()

    inputs = {
        "topic": topic.strip(),
        "needs_research": False,
        "queries": [],
        "evidence": [],
        "plan": None,
        "sections": [],
        "merged_md": "",
        "final": "",
        "image_plan": None,
    }

    progress = st.status(
        "🚀 Running Agentic Blog Writer...",
        expanded=True
    )

    try:

        final_output = None

        for update in app.stream(
            inputs,
            stream_mode="updates"
        ):

            for node_name, node_data in update.items():

                progress.write(
                    f"✅ `{node_name}` completed"
                )

                if (
                    isinstance(node_data, dict)
                    and "final" in node_data
                ):
                    final_output = node_data

        # Get complete state
        final_output = app.invoke(inputs)

        st.session_state[
            "last_out"
        ] = final_output

        progress.update(
            label="✅ Blog generated!",
            state="complete",
            expanded=False
        )

    except Exception as e:

        progress.update(
            label="❌ Generation failed",
            state="error"
        )

        st.error(
            f"Error: {e}"
        )


# ============================================================
# OUTPUT
# ============================================================

out = st.session_state.get(
    "last_out"
)


if out:

    tabs = st.tabs(
        [
            "🧩 Plan",
            "🔎 Research",
            "📝 Blog",
            "🖼️ Image",
        ]
    )

    # --------------------------------------------------------
    # PLAN
    # --------------------------------------------------------

    with tabs[0]:

        st.subheader(
            "Blog Plan"
        )

        plan = out.get("plan")

        if plan:

            plan_dict = get_plan_dict(
                plan
            )

            st.write(
                "**Title:**",
                plan_dict.get(
                    "blog_title"
                )
            )

            col1, col2 = st.columns(2)

            col1.write(
                "**Audience:** "
                + str(
                    plan_dict.get(
                        "audience"
                    )
                )
            )

            col2.write(
                "**Tone:** "
                + str(
                    plan_dict.get(
                        "tone"
                    )
                )
            )

            tasks = plan_dict.get(
                "tasks",
                []
            )

            if tasks:

                df = pd.DataFrame(
                    tasks
                )

                st.dataframe(
                    df,
                    use_container_width=True,
                    hide_index=True
                )

        else:

            st.info(
                "No plan available."
            )


    # --------------------------------------------------------
    # RESEARCH
    # --------------------------------------------------------

    with tabs[1]:

        st.subheader(
            "Research Sources"
        )

        evidence = out.get(
            "evidence",
            []
        )

        if not evidence:

            st.info(
                "No web research was required."
            )

        else:

            rows = []

            for item in evidence:

                if hasattr(
                    item,
                    "model_dump"
                ):
                    item = item.model_dump()

                rows.append(
                    {
                        "Title": item.get(
                            "title"
                        ),
                        "Source": item.get(
                            "url"
                        ),
                    }
                )

            st.dataframe(
                pd.DataFrame(rows),
                use_container_width=True,
                hide_index=True
            )


    # --------------------------------------------------------
    # BLOG
    # --------------------------------------------------------

    with tabs[2]:

        st.subheader(
            "Generated Blog"
        )

        final_md = out.get(
            "final",
            ""
        )

        if final_md:

            st.markdown(
                final_md
            )

            title = get_title(
                out
            )

            filename = (
                f"{safe_slug(title)}.md"
            )

            st.download_button(
                "⬇️ Download Markdown",
                data=final_md.encode(
                    "utf-8"
                ),
                file_name=filename,
                mime="text/markdown",
                use_container_width=True
            )

        else:

            st.warning(
                "No blog generated."
            )


    # --------------------------------------------------------
    # IMAGE
    # --------------------------------------------------------

    with tabs[3]:

        st.subheader(
            "🖼️ Image"
        )

        image_plan = out.get(
            "image_plan"
        )

        if not image_plan:

            st.info(
                "No image required."
            )

        else:

            if hasattr(
                image_plan,
                "model_dump"
            ):
                image_plan = (
                    image_plan.model_dump()
                )

            images = image_plan.get(
                "images",
                []
            )

            if not images:

                st.info(
                    "The editor decided that an image would not add enough value."
                )

            else:

                image = images[0]

                st.write(
                    "**Image:**",
                    image.get("alt")
                )

                st.caption(
                    image.get("caption")
                )

                st.info(
                    "Image generation is intentionally limited to ONE image to reduce free-tier usage."
                )

                with st.expander(
                    "Image prompt"
                ):
                    st.code(
                        image.get(
                            "prompt",
                            ""
                        )
                    )

else:

    st.info(
        "Enter a topic in the sidebar and click **Generate Blog**."
    )