from __future__ import annotations

import json
import os
import time
from pathlib import Path
from typing import TypedDict, List, Annotated
import operator

from dotenv import load_dotenv
from pydantic import BaseModel, Field

from langchain_groq import ChatGroq
from langchain_core.messages import SystemMessage, HumanMessage
from langchain_tavily import TavilySearch

from langgraph.graph import StateGraph, START, END
from langgraph.types import Send


# ============================================================
# 1. ENVIRONMENT
# ============================================================

load_dotenv()

GROQ_API_KEY = os.getenv("GROQ_API_KEY")
TAVILY_API_KEY = os.getenv("TAVILY_API_KEY")

if not GROQ_API_KEY:
    raise ValueError("GROQ_API_KEY missing from .env")

if not TAVILY_API_KEY:
    raise ValueError("TAVILY_API_KEY missing from .env")


# ============================================================
# 2. GROQ
# ============================================================

llm = ChatGroq(
    model="openai/gpt-oss-20b",
    temperature=0,
    max_tokens=200,
)


# ============================================================
# 3. TAVILY
# ============================================================

tavily = TavilySearch(
    max_results=2
)


# ============================================================
# 4. PLAN MODELS
# ============================================================

class Task(BaseModel):
    id: int
    title: str
    goal: str
    bullets: List[str]


class Plan(BaseModel):
    blog_title: str
    audience: str
    tone: str
    tasks: List[Task]


# ============================================================
# 5. STATE
# ============================================================

class State(TypedDict):
    topic: str

    plan: Plan

    research: Annotated[
        List[str],
        operator.add
    ]

    sections: Annotated[
        List[str],
        operator.add
    ]

    final: str


# ============================================================
# 6. SAFE GROQ CALL
# ============================================================

def groq_call(messages, retries=3):

    for attempt in range(retries):

        try:

            return llm.invoke(messages)

        except Exception as e:

            error = str(e)

            # Groq rate limit
            if "429" in error or "rate_limit" in error:

                wait_time = 8 + (attempt * 5)

                print(
                    f"\n⏳ Groq rate limit. "
                    f"Waiting {wait_time}s..."
                )

                time.sleep(wait_time)

            else:

                raise e

    raise RuntimeError(
        "Groq rate limit persisted after retries."
    )


# ============================================================
# 7. EXTRACT JSON
# ============================================================

def extract_json(text: str):

    text = text.strip()

    # Remove markdown code fence
    if text.startswith("```"):

        lines = text.splitlines()

        lines = [
            line
            for line in lines
            if not line.strip().startswith("```")
        ]

        text = "\n".join(lines).strip()

    # Find first {
    start = text.find("{")

    # Find last }
    end = text.rfind("}")

    if start == -1 or end == -1:

        raise ValueError(
            "Groq did not return valid JSON."
        )

    json_text = text[start:end + 1]

    return json.loads(json_text)


# ============================================================
# 8. ORCHESTRATOR
# ============================================================

def orchestrator(state: State):

    print("\n🧠 Creating blog plan...")

    messages = [

        SystemMessage(
            content=(
                "You are a technical blog planner.\n"
                "Return ONLY valid JSON.\n"
                "Do NOT use markdown.\n"
                "Do NOT use ```.\n\n"

                "JSON format:\n"
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
                "EXACTLY 5 tasks.\n"
                "Exactly 3 bullets per task.\n"
                "Keep all text short.\n"
                "Task 1 = fundamentals.\n"
                "Task 2 = core mechanism.\n"
                "Task 3 = implementation/examples.\n"
                "Task 4 = trade-offs/common mistakes.\n"
                "Task 5 = conclusion.\n"
            )
        ),

        HumanMessage(
            content=f"Topic: {state['topic']}"
        )
    ]

    response = groq_call(messages)

    try:

        data = extract_json(
            response.content
        )

        plan = Plan.model_validate(data)

    except Exception as e:

        print(
            f"⚠️ Plan parsing failed: {e}"
        )

        # Simple fallback
        plan = Plan(
            blog_title=state["topic"].title(),
            audience="software developers",
            tone="technical and practical",
            tasks=[
                Task(
                    id=1,
                    title="Fundamentals",
                    goal="Understand the basic concepts.",
                    bullets=[
                        "Define the core idea.",
                        "Explain why it is useful.",
                        "Show the basic workflow."
                    ]
                ),
                Task(
                    id=2,
                    title="Core Mechanism",
                    goal="Understand how the mechanism works.",
                    bullets=[
                        "Explain the main components.",
                        "Describe the processing steps.",
                        "Explain the important equations."
                    ]
                ),
                Task(
                    id=3,
                    title="Implementation",
                    goal="Understand how to implement it.",
                    bullets=[
                        "Show a minimal example.",
                        "Explain important parameters.",
                        "Mention practical implementation tips."
                    ]
                ),
                Task(
                    id=4,
                    title="Trade-offs and Mistakes",
                    goal="Avoid common implementation problems.",
                    bullets=[
                        "Explain major limitations.",
                        "Show common mistakes.",
                        "Explain performance considerations."
                    ]
                ),
                Task(
                    id=5,
                    title="Conclusion",
                    goal="Summarize the practical takeaways.",
                    bullets=[
                        "Summarize the main concepts.",
                        "List important takeaways.",
                        "Suggest what to learn next."
                    ]
                )
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
# 9. FANOUT → RESEARCH
# ============================================================

def fanout(state: State):

    return [

        Send(
            "research",
            {
                "task": task,
                "topic": state["topic"]
            }
        )

        for task in state["plan"].tasks
    ]


# ============================================================
# 10. RESEARCH
# ============================================================

def research_node(payload: dict):

    task = payload["task"]
    topic = payload["topic"]

    print(
        f"\n🔎 Researching: {task.title}"
    )

    query = (
        f"{topic} {task.title}"
    )

    try:

        results = tavily.invoke(
            {
                "query": query
            }
        )

        research_text = ""

        # Tavily list
        if isinstance(results, list):

            for result in results[:2]:

                if isinstance(result, dict):

                    title = result.get(
                        "title",
                        ""
                    )

                    content = result.get(
                        "content",
                        ""
                    )

                    url = result.get(
                        "url",
                        ""
                    )

                    research_text += (
                        f"{title}\n"
                        f"{content[:800]}\n"
                        f"{url}\n\n"
                    )

        # Tavily dictionary
        elif isinstance(results, dict):

            items = results.get(
                "results",
                []
            )

            for result in items[:2]:

                if isinstance(result, dict):

                    research_text += (
                        f"{result.get('title', '')}\n"
                        f"{result.get('content', '')[:800]}\n"
                        f"{result.get('url', '')}\n\n"
                    )

        else:

            research_text = str(results)

    except Exception as e:

        print(
            f"⚠️ Research failed: {e}"
        )

        research_text = (
            "No external research available."
        )

    # VERY IMPORTANT
    # Keep Tavily context small
    research_text = research_text[:1800]

    return {
        "research": [
            research_text
        ]
    }


# ============================================================
# 11. RESEARCH → WORKER
# ============================================================

def research_to_worker(state: State):

    sends = []

    tasks = state["plan"].tasks

    research_items = state["research"]

    for i, task in enumerate(tasks):

        research = ""

        if i < len(research_items):

            research = research_items[i]

        sends.append(

            Send(
                "worker",
                {
                    "task": task,
                    "topic": state["topic"],
                    "plan": state["plan"],
                    "research": research
                }
            )
        )

    return sends


# ============================================================
# 12. WORKER
# ============================================================

def worker(payload: dict):

    task = payload["task"]
    topic = payload["topic"]
    plan = payload["plan"]
    research = payload.get(
        "research",
        ""
    )

    print(
        f"✍️ Writing: {task.title}"
    )

    bullets = "\n".join(
        f"- {b}"
        for b in task.bullets
    )

    messages = [

        SystemMessage(
            content=(
                "You are a concise technical writer.\n\n"

                "Write ONE Markdown section.\n"

                "Rules:\n"
                "Start with ## section title.\n"
                "Cover all 3 bullets.\n"
                "Use the research when useful.\n"
                "Maximum 120 words.\n"
                "No H1.\n"
                "No introduction outside the section.\n"
                "No conclusion outside the section.\n"
                "Avoid repetition.\n"
                "Be technically accurate."
            )
        ),

        HumanMessage(
            content=(
                f"Topic: {topic}\n"
                f"Blog: {plan.blog_title}\n"
                f"Audience: {plan.audience}\n\n"

                f"Section: {task.title}\n"
                f"Goal: {task.goal}\n\n"

                f"Bullets:\n"
                f"{bullets}\n\n"

                f"Research:\n"
                f"{research[:1500]}"
            )
        )
    ]

    response = groq_call(
        messages
    )

    section = response.content.strip()

    return {
        "sections": [
            section
        ]
    }


# ============================================================
# 13. REDUCER
# ============================================================

def reducer(state: State):

    print(
        "\n🧩 Combining sections..."
    )

    title = state["plan"].blog_title

    sections = state["sections"]

    body = "\n\n".join(
        sections
    ).strip()

    final_md = (
        f"# {title}\n\n"
        f"{body}\n"
    )

    # Safe filename
    filename = "".join(

        c
        if c.isalnum()
        or c in (" ", "_", "-")
        else ""

        for c in title
    )

    filename = (
        filename
        .strip()
        .lower()
        .replace(" ", "_")
    )

    if not filename:

        filename = "blog"

    filename += ".md"

    Path(filename).write_text(
        final_md,
        encoding="utf-8"
    )

    print(
        f"💾 Saved: {filename}"
    )

    return {
        "final": final_md
    }


# ============================================================
# 14. GRAPH
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


graph.add_conditional_edges(
    "research",
    research_to_worker,
    ["worker"]
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
# 15. RUN
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
                "research": [],
                "sections": [],
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
            "\n"
            + output["final"]
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