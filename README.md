# ✍️ Agentic AI Blog Writer

An AI-powered multi-agent blog writing application that researches a user-provided topic, creates a structured blog plan, generates detailed sections, and combines them into a complete Markdown blog.

The project uses **LangGraph** to orchestrate the agent workflow, **Groq (`openai/gpt-oss-20b`)** for generation, **Tavily** for web research, and **Streamlit** for the user interface.

---

## 🚀 Features

- 🤖 Multi-agent blog generation workflow
- 🧠 LangGraph-based agent orchestration
- 📋 Automatic blog planning
- 🔎 Web research using Tavily
- ✍️ AI-generated technical blog sections
- 🔄 Parallel research for different blog sections
- 🧩 Reducer node for combining generated sections
- 📄 Markdown-based final output
- 📊 Structured plan display
- 📈 Live workflow progress in Streamlit
- 🖼️ Locally generated SVG diagram
- 💾 Download generated blogs as Markdown files
- 🔁 Retry and fallback handling for failed LLM responses
- ⚡ Token-aware prompts and research truncation
- 🔐 Environment-variable based API key management

---

## 🏗️ Architecture

```text
                         ┌─────────────────┐
                         │   User Topic    │
                         └────────┬────────┘
                                  │
                                  ▼
                         ┌─────────────────┐
                         │  Orchestrator   │
                         │      Agent      │
                         └────────┬────────┘
                                  │
                                  ▼
                         ┌─────────────────┐
                         │   Blog Plan     │
                         │  5 Sections     │
                         └────────┬────────┘
                                  │
                    ┌─────────────┼─────────────┐
                    │             │             │
                    ▼             ▼             ▼
              ┌──────────┐  ┌──────────┐  ┌──────────┐
              │ Research │  │ Research │  │ Research │
              │  Agent   │  │  Agent   │  │  Agent   │
              └────┬─────┘  └────┬─────┘  └────┬─────┘
                   │             │             │
                   └─────────────┼─────────────┘
                                 │
                                 ▼
                         ┌─────────────────┐
                         │  Writer Agents  │
                         └────────┬────────┘
                                  │
                                  ▼
                         ┌─────────────────┐
                         │     Reducer     │
                         │    / Combiner   │
                         └────────┬────────┘
                                  │
                                  ▼
                         ┌─────────────────┐
                         │  Final Markdown │
                         │      Blog       │
                         └────────┬────────┘
                                  │
                                  ▼
                         ┌─────────────────┐
                         │ Local SVG       │
                         │ Diagram        │
                         └─────────────────┘
```

---

## 🧠 How It Works

### 1. User Input

The user enters a topic in the Streamlit interface.

Example:

```text
Machine Learning
```

### 2. Orchestrator Agent

The Orchestrator analyzes the topic and creates a structured blog plan.

The plan contains:

- Blog title
- Section titles
- Section goals
- Important points to cover

This allows the system to break a large writing task into smaller tasks instead of asking one LLM call to generate the entire article.

### 3. Research Agents

Each planned section is sent to a research agent.

The research agents use **Tavily** to find relevant information from the web.

Research is performed separately for each section so that every writer receives focused information.

### 4. Writer Agents

The writer stage receives:

- Blog topic
- Blog title
- Section title
- Section goal
- Required points
- Relevant research

The Groq model then generates the Markdown section.

The prompt is designed to produce:

- Clear explanations
- Technical accuracy
- Practical examples
- Markdown formatting
- Short paragraphs
- Bullet points where useful
- Code examples when relevant

### 5. Reducer

The Reducer combines the generated sections into the final blog.

This produces one complete Markdown document instead of separate independent outputs.

### 6. SVG Diagram

A lightweight local SVG diagram is generated to visually represent the blog workflow.

No additional image-generation API is required for this diagram.

---

## 🛠️ Tech Stack

| Technology | Purpose |
|---|---|
| Python | Core programming language |
| LangGraph | Agent workflow and state management |
| LangChain | LLM and tool integration |
| Groq | Fast LLM inference |
| `openai/gpt-oss-20b` | Blog planning and generation |
| Tavily | Web research |
| Streamlit | Frontend and interactive UI |
| Pydantic | Structured data validation |
| python-dotenv | Environment variable management |
| SVG | Lightweight local diagram generation |

---

## 📂 Project Structure

```text
AI-Blog-Writing-Agent/
│
├── screenshots/
│   ├── 1_UI.jpeg
│   ├── 2_UI.jpeg
│   ├── 3_UI.jpeg
│   └── 4_UI.jpeg
│
├── backend.py
├── frontend.py
├── requirements.txt
├── .env.example
├── .gitignore
└── README.md
```

---

## ⚙️ Installation

### 1. Clone the repository

```bash
git clone https://github.com/snehapankhi05/AI-Blog-Writing-Agent.git
```

```bash
cd AI-Blog-Writing-Agent
```

### 2. Create a virtual environment

Windows:

```powershell
python -m venv venv
```

Activate it:

```powershell
.\venv\Scripts\Activate.ps1
```

Linux/macOS:

```bash
python3 -m venv venv
source venv/bin/activate
```

### 3. Install dependencies

```bash
pip install -r requirements.txt
```

---

## 🔑 Environment Variables

Create a `.env` file in the project root.

```env
GROQ_API_KEY=your_groq_api_key
TAVILY_API_KEY=your_tavily_api_key
```

The `.env` file should never be committed to GitHub.

The repository uses `.gitignore` to keep API credentials and the virtual environment out of version control.

---

## ▶️ Run the Application

Start the Streamlit application:

```bash
streamlit run frontend.py
```

The application will open in your browser.

Enter a topic and click:

```text
🚀 Generate Blog
```

The interface displays the workflow progress while the agents execute.

---

## 🖥️ Application Screenshots

### Application Interface

![Application Interface](./screenshots/1_UI.jpeg)

### Agentic Blog Generation

![Agentic Blog Generation](./screenshots/2_UI.jpeg)

### Generated Blog

![Generated Blog](./screenshots/3_UI.jpeg)

### Generated Diagram

![Generated Diagram](./screenshots/4_UI.jpeg)

---

## 📌 Example Workflow

Input:

```text
Machine Learning
```

The Orchestrator can create a plan such as:

```text
Mastering Machine Learning: A Practical Guide

1. Fundamentals of Machine Learning
2. Core Mechanism: Gradient Descent
3. Implementation Workflow
4. Common Mistakes & Trade-offs
5. Conclusion & Next Steps
```

The research agents then gather information for the individual sections.

The writer agents generate the sections.

Finally, the Reducer combines them into one Markdown article.

---

## 🎯 Why LangGraph?

A single LLM call could generate a blog, but it would provide less control over the workflow.

LangGraph was chosen because it provides:

- Explicit workflow control
- State management
- Multiple agent stages
- Conditional routing
- Parallel execution
- Reducer-based aggregation
- Better observability
- Easier extension of the workflow

The project demonstrates an actual agentic workflow rather than simply sending one prompt to an LLM.

---

## 🤖 Why a Multi-Agent Architecture?

The writing task is divided into specialized responsibilities:

```text
Orchestrator
     ↓
Planning
     ↓
Research
     ↓
Writing
     ↓
Reduction
```

Each stage has a specific purpose.

This makes the system easier to:

- Debug
- Extend
- Control
- Optimize
- Maintain

It also allows research and writing to be handled independently.

---

## 🔎 Research Strategy

Tavily is used to provide external web information to the writing agents.

Research context is intentionally limited before being passed to the LLM.

This helps:

- Reduce token usage
- Keep prompts focused
- Avoid unnecessarily large contexts
- Improve generation efficiency

The research stage is an internal part of the agent pipeline rather than the final user-facing output.

---

## 💰 Token Optimization

The project was designed with limited LLM token usage in mind.

Several decisions were made to control token consumption:

- Limited Tavily results
- Truncated research context
- Concise system prompts
- Controlled output length
- Section-based generation
- No unnecessary image-generation API
- Retry only when generation fails
- Local SVG generation instead of an external image model

This keeps the application lightweight and cost-efficient.

---

## 🛡️ Reliability & Error Handling

LLM-based systems can occasionally return empty responses or fail because of temporary API issues.

The project therefore includes:

- Retry handling
- Empty-response detection
- Fallback section generation
- API error handling
- State validation
- Controlled generation limits

If a writer agent fails to produce content, the system can fall back to the section goal and required points instead of completely failing the workflow.

---

## 🧩 Important LangGraph Design Decision

The research stage uses parallel execution.

Initially, sending shared state values directly to parallel branches caused a LangGraph concurrent state update problem.

The solution was to send each research branch a private payload containing only the information it needs:

```text
ResearchPayload
    ├── task
    └── topic
```

Research results are then aggregated into a reducer-backed state field.

This allows multiple research branches to complete without overwriting the same state value.

---

## 🖼️ Why Local SVG Instead of Image Generation?

The project does not require an additional image-generation API for its workflow diagram.

Instead, the diagram is generated locally using SVG.

Advantages:

- No additional API cost
- No image model dependency
- Fast generation
- Lightweight output
- Easy to customize
- Works well for technical diagrams

This keeps the application focused on the core agentic workflow.

---

## 📄 Generated Output

The final output is a Markdown blog containing:

- Blog title
- Structured sections
- Technical explanations
- Bullet points
- Examples where relevant
- Markdown formatting
- Workflow diagram

The generated Markdown can also be downloaded from the Streamlit interface.

---

## 🔮 Future Improvements

Possible future improvements include:

- Better source ranking
- Multiple search queries per section
- Citation support inside generated blogs
- Human-in-the-loop editing
- Blog tone selection
- SEO optimization
- Automatic keyword extraction
- Image generation
- Blog quality evaluation
- Fact-checking agent
- Plagiarism detection
- Multiple LLM provider support
- Persistent blog history
- Cloud deployment
- Streaming token-level generation

---

## 🎓 Key Learning Outcomes

Through this project, I worked with:

- Agentic AI architecture
- LangGraph state machines
- Multi-agent workflows
- Parallel agent execution
- Reducer-based state aggregation
- Tool calling
- Web research with Tavily
- LLM application development
- Prompt engineering
- Structured data validation
- Error handling and retries
- Token optimization
- Streamlit application development
- Git and GitHub project management

---

## 🎯 Project Objective

The main objective of this project was to build a practical **agentic AI system** capable of transforming a simple user topic into a researched and structured technical blog.

Instead of relying on a single LLM request, the system separates planning, research, writing, and final assembly into different stages managed by LangGraph.

---

## 👨‍💻 Author

**Khush Trivedi **

