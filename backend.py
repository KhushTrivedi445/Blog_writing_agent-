from __future__ import annotations

import json
import operator
import os
import re
import time
from pathlib import Path
from typing import Annotated, List, TypedDict

from dotenv import load_dotenv
from pydantic import BaseModel, Field

from langchain_core.messages import HumanMessage, SystemMessage
from langchain_groq import ChatGroq
from langchain_tavily import TavilySearch

from langgraph.graph import END, START, StateGraph
from langgraph.types import Send


# ============================================================
# 1. ENVIRONMENT
# ============================================================

load_dotenv()

if not os.getenv("GROQ_API_KEY"):
    raise ValueError("GROQ_API_KEY is missing. Add it to your .env file.")

if not os.getenv("TAVILY_API_KEY"):
    raise ValueError("TAVILY_API_KEY is missing. Add it to your .env file.")


# ============================================================
# 2. GROQ
# ============================================================

llm = ChatGroq(
    model="openai/gpt-oss-20b",
    temperature=0,
    max_tokens=700,
)


# ============================================================
# 3. TAVILY
# ============================================================

tavily = TavilySearch(
    max_results=1
)


# ============================================================
# 4. SCHEMAS
# ============================================================

class Task(BaseModel):
    id: int
    title: str
    goal: str
    bullets: List[str] = Field(
        min_length=3,
        max_length=3
    )


class Plan(BaseModel):
    blog_title: str
    audience: str
    tone: str
    tasks: List[Task]


class ResearchPayload(TypedDict):
    """
    Private payload used by each parallel research branch.
    These fields are NOT written into shared State.
    """
    task: Task
    topic: str


class State(TypedDict, total=False):

    topic: str

    needs_research: bool

    plan: Plan

    queries: Annotated[
        List[str],
        operator.add
    ]

    evidence: Annotated[
        List[dict],
        operator.add
    ]

    sections: Annotated[
        List[str],
        operator.add
    ]

    merged_md: str

    final: str

    image_plan: dict


# ============================================================
# 5. HELPERS
# ============================================================

def clean_json(text: str) -> str:
    text = text.strip()

    # Remove ```json fences
    text = re.sub(
        r"^```json\s*",
        "",
        text,
        flags=re.IGNORECASE
    )

    # Remove generic fences
    text = re.sub(
        r"^```\s*",
        "",
        text
    )

    text = re.sub(
        r"\s*```$",
        "",
        text
    )

    # Extract JSON object
    start = text.find("{")
    end = text.rfind("}")

    if start != -1 and end != -1:
        text = text[start:end + 1]

    return text.strip()


def groq_call(messages, retries: int = 2):
    """
    Small retry wrapper for transient Groq 429 errors.
    """
    last_error = None

    for attempt in range(retries):

        try:
            return llm.invoke(messages)

        except Exception as e:

            last_error = e
            error_text = str(e)

            if "429" not in error_text:
                raise

            # Small bounded wait
            wait_time = 4 * (attempt + 1)

            print(
                f"⏳ Groq rate limit. Waiting {wait_time}s..."
            )

            time.sleep(wait_time)

    raise RuntimeError(
        f"Groq rate limit persisted after retries: {last_error}"
    )


# ============================================================
# 6. ORCHESTRATOR
# ============================================================

def orchestrator(state: State):

    topic = state["topic"]

    print("\n🧠 Creating blog plan...")

    messages = [
        SystemMessage(
            content=(
                "Create a concise technical blog plan.\n"
                "Return ONLY valid JSON.\n"
                "No markdown.\n"
                "No explanation.\n\n"

                "Schema:\n"
                "{"
                "\"blog_title\":\"...\","
                "\"audience\":\"...\","
                "\"tone\":\"...\","
                "\"tasks\":["
                "{"
                "\"id\":1,"
                "\"title\":\"...\","
                "\"goal\":\"...\","
                "\"bullets\":[\"...\",\"...\",\"...\"]"
                "}"
                "]"
                "}\n\n"

                "Rules:\n"
                "- Exactly 5 tasks.\n"
                "- Exactly 3 bullets per task.\n"
                "- Keep everything concise.\n"
                "- Technical and practical.\n"
                "- Cover fundamentals, core mechanism, "
                "implementation, mistakes/trade-offs, conclusion."
            )
        ),
        HumanMessage(
            content=f"Topic: {topic}"
        )
    ]

    response = groq_call(
        messages,
        retries=2
    )

    try:

        data = json.loads(
            clean_json(response.content)
        )

        plan = Plan.model_validate(data)

    except Exception as e:

        print(
            f"⚠️ Plan parsing failed: {e}"
        )

        # Safe deterministic fallback.
        plan = Plan(
            blog_title=f"{topic}: From Theory to Production",
            audience="Developers and AI/ML learners",
            tone="Technical and practical",
            tasks=[
                Task(
                    id=1,
                    title="Fundamentals",
                    goal="Explain the basic concepts and why they matter.",
                    bullets=[
                        "Definition",
                        "Why it matters",
                        "Key concepts",
                    ]
                ),
                Task(
                    id=2,
                    title="Core Mechanism",
                    goal="Explain how the system works internally.",
                    bullets=[
                        "Main components",
                        "Workflow",
                        "Important mechanisms",
                    ]
                ),
                Task(
                    id=3,
                    title="Implementation",
                    goal="Show how the concepts are applied in practice.",
                    bullets=[
                        "Practical workflow",
                        "Implementation choices",
                        "Example",
                    ]
                ),
                Task(
                    id=4,
                    title="Mistakes & Trade-offs",
                    goal="Discuss common mistakes and important trade-offs.",
                    bullets=[
                        "Common mistakes",
                        "Limitations",
                        "Trade-offs",
                    ]
                ),
                Task(
                    id=5,
                    title="Conclusion",
                    goal="Summarize the key lessons and practical guidance.",
                    bullets=[
                        "Key takeaways",
                        "Best practices",
                        "Next steps",
                    ]
                ),
            ]
        )

    print(
        f"✅ Plan created: {plan.blog_title}"
    )

    print(
        f"📚 Sections: {len(plan.tasks)}"
    )

    return {
        "plan": plan
    }


# ============================================================
# 7. FANOUT
# ============================================================

def fanout(state: State):

    plan = state["plan"]
    topic = state["topic"]

    return [
        Send(
            "research",
            {
                "task": task,
                "topic": topic,
            }
        )
        for task in plan.tasks
    ]


# ============================================================
# 8. RESEARCH
# ============================================================

def research_node(payload: ResearchPayload):

    task = payload["task"]
    topic = payload["topic"]

    print(
        f"\n🔎 Researching: {task.title}"
    )

    query = f"{topic} {task.title}"

    evidence = []

    try:

        results = tavily.invoke(
            {
                "query": query
            }
        )

        if isinstance(results, dict):

            possible_results = (
                results.get("results")
                or results.get("data")
                or []
            )

        elif isinstance(results, list):

            possible_results = results

        else:

            possible_results = []

        for item in possible_results:

            if not isinstance(item, dict):
                continue

            title = str(
                item.get("title", "")
            )

            content = str(
                item.get("content", "")
            )

            url = str(
                item.get("url", "")
            )

            if content:

                evidence.append(
                    {
                        "task_id": task.id,
                        "task_title": task.title,
                        "title": title[:160],
                        "content": content[:1400],
                        "url": url[:300],
                    }
                )

            # Only one useful source
            if len(evidence) >= 1:
                break

    except Exception as e:

        print(
            f"⚠️ Tavily research failed for "
            f"{task.title}: {e}"
        )

    if not evidence:

        evidence.append(
            {
                "task_id": task.id,
                "task_title": task.title,
                "title": "No external source",
                "content": (
                    "No external research was available. "
                    "Use general technical knowledge carefully."
                ),
                "url": "",
            }
        )

    return {
        "evidence": evidence
    }


# ============================================================
# 9. WRITER
# ============================================================

def worker(state: State):

    topic = state["topic"]
    plan = state["plan"]

    evidence = state.get(
        "evidence",
        []
    )

    print("\n✍️ Writing blog sections...")

    # ========================================================
    # Group research by task
    # ========================================================

    research_by_task = {}

    for item in evidence:

        task_id = item.get("task_id")

        if task_id not in research_by_task:
            research_by_task[task_id] = []

        research_by_task[task_id].append(item)

    sections = []

    # ========================================================
    # Generate each section
    # ========================================================

    for task in plan.tasks:

        print(
            f"✍️ Writing: {task.title}"
        )

        # ----------------------------------------------------
        # Collect research for this task
        # ----------------------------------------------------

        task_evidence = research_by_task.get(
            task.id,
            []
        )

        research_text = ""

        for item in task_evidence:

            research_text += (
                f"Source: {item.get('title', '')}\n"
                f"Content: {item.get('content', '')}\n"
                f"URL: {item.get('url', '')}\n\n"
            )

        # Keep prompt small to reduce Groq token usage
        research_text = research_text[:1600]

        bullets = "\n".join(
            f"- {bullet}"
            for bullet in task.bullets
        )

        # ====================================================
        # SPECIAL CASE: CONCLUSION
        # ====================================================

        is_conclusion = (
            task.id == plan.tasks[-1].id
        )

        # ----------------------------------------------------
        # Normal sections
        # ----------------------------------------------------

        if not is_conclusion:

            messages = [
                SystemMessage(
    content=(
        "You are an expert technical blog writer.\n\n"

        "Write ONE detailed, polished Markdown section for a "
        "long-form technical blog.\n\n"

        "Writing requirements:\n"
        "- Write approximately 250-300 words.\n"
        "- Explain the topic clearly from basic to practical level.\n"
        "- Use short paragraphs of 2-4 sentences.\n"
        "- Explain important technical terms in simple language.\n"
        "- Include practical examples where relevant.\n"
        "- Cover ALL required points.\n"
        "- Add useful technical detail instead of filler.\n"
        "- Avoid repeating ideas from other sections.\n"
        "- Do not copy the research text directly.\n"
        "- Do not mention the research process.\n"
        "- Do not invent unsupported technical claims.\n\n"

        "Formatting requirements:\n"
        "- Start with ## and the exact section title.\n"
        "- Use ### subsections when they improve readability.\n"
        "- Use bullet lists when appropriate.\n"
        "- Use numbered steps for workflows.\n"
        "- Use **bold** for important concepts.\n"
        "- Use inline code for functions, variables, parameters, "
        "libraries, and technical identifiers.\n"
        "- Use code blocks when a real code example is useful.\n"
        "- Do not create an H1.\n"
        "- Return ONLY the Markdown section."
    )
),

                HumanMessage(
                    content=(
                        f"Topic: {topic}\n"
                        f"Blog title: {plan.blog_title}\n"
                        f"Audience: {plan.audience}\n\n"

                        f"Section title: {task.title}\n"
                        f"Section goal: {task.goal}\n\n"

                        f"Required points:\n"
                        f"{bullets}\n\n"

                        f"Research:\n"
                        f"{research_text}"
                    )
                )
            ]

            section = ""

            # ------------------------------------------------
            # At most 2 attempts for normal sections
            # ------------------------------------------------

            for attempt in range(2):

                try:

                    response = groq_call(
                        messages,
                        retries=2
                    )

                    content = response.content

                    # Standard Groq/LangChain response
                    if isinstance(content, str):

                        section = content.strip()

                    # Some models can return block-style content
                    elif isinstance(content, list):

                        parts = []

                        for block in content:

                            if isinstance(block, dict):

                                text = block.get(
                                    "text",
                                    ""
                                )

                                if text:
                                    parts.append(
                                        str(text)
                                    )

                        section = "\n".join(
                            parts
                        ).strip()

                    if section:
                        break

                    print(
                        f"⚠️ Empty response for "
                        f"{task.title}. Retrying..."
                    )

                    if attempt == 0:
                        time.sleep(1)

                except Exception as e:

                    print(
                        f"⚠️ Writing failed for "
                        f"{task.title}: {e}"
                    )

                    if attempt == 0:
                        time.sleep(2)

            # ------------------------------------------------
            # Safe fallback for normal sections
            # ------------------------------------------------

            if not section:

                section = (
                    f"## {task.title}\n\n"
                    f"{task.goal}\n\n"
                    "Key points:\n"
                    + "\n".join(
                        f"- {bullet}"
                        for bullet in task.bullets
                    )
                )

            sections.append(
                section.strip()
            )

            continue

        # ====================================================
        # CONCLUSION
        # ====================================================

        print(
            "🧠 Creating conclusion from generated sections..."
        )

        # Use already generated content.
        # This prevents the conclusion from depending only
        # on Tavily research and gives the model context
        # about the actual article.
        previous_content = "\n\n".join(
            sections
        )

        # Keep the conclusion prompt reasonably small.
        previous_content = previous_content[-5000:]

        conclusion_messages = [

            SystemMessage(
                content=(
                    "You are finishing a technical blog.\n\n"

                    "Write a concise conclusion based ONLY on "
                    "the article content provided.\n"

                    "Rules:\n"
                    "- Start with ## Conclusion & Next Steps.\n"
                    "- Summarize the main technical lessons.\n"
                    "- Mention practical next steps.\n"
                    "- Do not introduce unrelated claims.\n"
                    "- Do not repeat entire sections.\n"
                    "- Write approximately 80-110 words.\n"
                    "- Return ONLY the Markdown section."
                )
            ),

            HumanMessage(
                content=(
                    f"Blog title: {plan.blog_title}\n"
                    f"Topic: {topic}\n\n"

                    f"Article written so far:\n"
                    f"{previous_content}\n\n"

                    "Create the conclusion now."
                )
            )
        ]

        section = ""

        # ----------------------------------------------------
        # One retry only for conclusion
        # ----------------------------------------------------

        for attempt in range(2):

            try:

                response = groq_call(
                    conclusion_messages,
                    retries=1
                )

                content = response.content

                if isinstance(content, str):

                    section = content.strip()

                elif isinstance(content, list):

                    parts = []

                    for block in content:

                        if isinstance(block, dict):

                            text = block.get(
                                "text",
                                ""
                            )

                            if text:
                                parts.append(
                                    str(text)
                                )

                    section = "\n".join(
                        parts
                    ).strip()

                if section:
                    break

                print(
                    "⚠️ Empty conclusion response. "
                    "Retrying..."
                )

                if attempt == 0:
                    time.sleep(1)

            except Exception as e:

                print(
                    f"⚠️ Conclusion generation failed: {e}"
                )

                if attempt == 0:
                    time.sleep(1)

        # ====================================================
        # Deterministic conclusion fallback
        # ====================================================

        if not section:

            section = (
                "## Conclusion & Next Steps\n\n"
                f"{topic.title()} is best understood by "
                "combining its fundamental concepts with "
                "practical implementation and careful evaluation. "
                "The key lessons are to understand how the underlying "
                "methods work, follow a reliable development workflow, "
                "and consider limitations and trade-offs before "
                "deployment. A practical next step is to build a "
                "small project, evaluate its results, and gradually "
                "explore more advanced techniques."
            )

        sections.append(
            section.strip()
        )

    # ========================================================
    # Return all generated sections
    # ========================================================

    return {
        "sections": sections
    }
# ============================================================
# 10. LOCAL SVG IMAGE
# ============================================================

def create_image_plan(title: str, topic: str):

    safe_title = re.sub(
        r"[^a-zA-Z0-9 _-]",
        "",
        title
    ).strip()

    filename = (
        safe_title.lower()
        .replace(" ", "_")
        or "blog"
    )

    image_filename = (
        f"{filename}_diagram.svg"
    )

    safe_topic = re.sub(
        r"[^a-zA-Z0-9 ,.:_-]",
        "",
        topic
    )[:80]

    svg = f"""<svg
xmlns="http://www.w3.org/2000/svg"
width="1200"
height="500"
viewBox="0 0 1200 500">

<rect
width="1200"
height="500"
fill="#111827"/>

<text
x="600"
y="80"
text-anchor="middle"
font-family="Arial"
font-size="36"
font-weight="bold"
fill="white">
{safe_title[:55]}
</text>

<rect
x="100"
y="180"
width="220"
height="100"
rx="15"
fill="#1f2937"
stroke="#9ca3af"/>

<text
x="210"
y="240"
text-anchor="middle"
font-family="Arial"
font-size="24"
fill="white">
Input
</text>

<rect
x="490"
y="180"
width="220"
height="100"
rx="15"
fill="#1f2937"
stroke="#9ca3af"/>

<text
x="600"
y="240"
text-anchor="middle"
font-family="Arial"
font-size="24"
fill="white">
Processing
</text>

<rect
x="880"
y="180"
width="220"
height="100"
rx="15"
fill="#1f2937"
stroke="#9ca3af"/>

<text
x="990"
y="240"
text-anchor="middle"
font-family="Arial"
font-size="24"
fill="white">
Output
</text>

<line
x1="320"
y1="230"
x2="490"
y2="230"
stroke="white"
stroke-width="4"/>

<polygon
points="490,230 470,215 470,245"
fill="white"/>

<line
x1="710"
y1="230"
x2="880"
y2="230"
stroke="white"
stroke-width="4"/>

<polygon
points="880,230 860,215 860,245"
fill="white"/>

<text
x="600"
y="380"
text-anchor="middle"
font-family="Arial"
font-size="22"
fill="#d1d5db">
{safe_topic}
</text>

</svg>
"""

    Path(image_filename).write_text(
        svg,
        encoding="utf-8"
    )

    return {
        "images": [
            {
                "alt": f"Diagram for {title}",
                "caption": "Minimal conceptual diagram.",
                "prompt": (
                    f"Minimal technical diagram explaining {topic}."
                ),
                "path": image_filename,
            }
        ]
    }


# ============================================================
# 11. REDUCER
# ============================================================

def reducer(state: State):

    print(
        "\n🧩 Combining sections..."
    )

    title = state["plan"].blog_title

    sections = state.get(
        "sections",
        []
    )

    body = "\n\n".join(
        sections
    ).strip()

    final_md = (
        f"# {title}\n\n"
        f"{body}\n"
    )

    # Save Markdown
    filename = re.sub(
        r"[^a-zA-Z0-9 _-]",
        "",
        title
    )

    filename = (
        filename.strip()
        .lower()
        .replace(" ", "_")
        or "blog"
    )

    markdown_path = Path(
        filename + ".md"
    )

    markdown_path.write_text(
        final_md,
        encoding="utf-8"
    )

    print(
        f"💾 Saved: {markdown_path}"
    )

    # One local image
    image_plan = create_image_plan(
        title,
        state["topic"]
    )

    return {
        "merged_md": final_md,
        "final": final_md,
        "image_plan": image_plan,
    }


# ============================================================
# 12. GRAPH
# ============================================================

graph = StateGraph(State)

graph.add_node(
    "orchestrator",
    orchestrator
)

graph.add_node(
    "research",
    research_node
)

graph.add_node(
    "worker",
    worker
)

graph.add_node(
    "reducer",
    reducer
)

graph.add_edge(
    START,
    "orchestrator"
)

graph.add_conditional_edges(
    "orchestrator",
    fanout,
    ["research"]
)

# Important:
# All parallel research branches finish before
# the worker node continues.
graph.add_edge(
    "research",
    "worker"
)

graph.add_edge(
    "worker",
    "reducer"
)

graph.add_edge(
    "reducer",
    END
)

app = graph.compile()


# ============================================================
# 13. TERMINAL MODE
# ============================================================

if __name__ == "__main__":

    topic = input(
        "\nEnter blog topic: "
    ).strip()

    if not topic:

        print(
            "❌ Topic cannot be empty."
        )

        raise SystemExit

    print(
        "\n🚀 Starting Agentic Blog Writer..."
    )

    try:

        output = app.invoke(
            {
                "topic": topic,
                "needs_research": True,
                "queries": [],
                "evidence": [],
                "sections": [],
                "merged_md": "",
                "final": "",
                "image_plan": None,
            }
        )

        print(
            "\n"
            + "=" * 70
        )

        print(
            "🎉 BLOG GENERATED"
        )

        print(
            "=" * 70
        )

        print(
            output["final"]
        )

        print(
            "\n"
            + "=" * 70
        )

        print(
            "✅ Done!"
        )

    except Exception as e:

        print(
            f"\n❌ Error: {e}"
        )