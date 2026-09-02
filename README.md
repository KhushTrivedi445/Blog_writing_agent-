# ✍️ Agentic AI Blog Writer

An AI-powered multi-agent blog writing application that researches a user-provided topic, creates a structured blog plan, generates detailed sections, and combines them into a complete Markdown blog.

The project uses **LangGraph** to orchestrate the agent workflow, **Groq (`openai/gpt-oss-20b`)** for AI generation, **Tavily** for web research, and **Streamlit** for the interactive user interface.

---

## 🚀 Features

- 🤖 Multi-agent AI blog generation
- 🧠 LangGraph-based agent orchestration
- 📋 Automatic blog planning
- 🔎 Web research using Tavily
- ✍️ AI-generated blog sections
- 🔄 Parallel research for different sections
- 🧩 Reducer-based section combination
- 📄 Complete Markdown output
- 📊 Structured blog plan
- 📈 Live workflow progress
- 🖼️ Locally generated SVG diagram
- 💾 Download generated blogs as Markdown
- 🔁 Retry and fallback handling
- ⚡ Token-aware generation
- 🔐 Secure API key management using environment variables

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
                         │   SVG Diagram   │
                         └─────────────────┘
```

---

## 🧠 How It Works

### 1. User Input

The user enters a topic through the Streamlit interface.

Example:

```text
Machine Learning
```

The topic is passed to the LangGraph workflow.

---

### 2. Orchestrator Agent

The Orchestrator analyzes the requested topic and creates a structured blog plan.

The plan contains:

- Blog title
- Section titles
- Section goals
- Important points to cover

Breaking the task into smaller sections provides better control than generating the complete article using a single LLM call.

---

### 3. Research Agents

Each planned section is assigned to a research agent.

The research agents use **Tavily** to retrieve relevant information from the web.

Research is performed independently for different sections so that each writer receives focused context related to its assigned topic.

---

### 4. Writer Agents

The Writer agents generate individual Markdown sections using the research collected for each task.

Each writer receives:

- Blog topic
- Blog title
- Section title
- Section goal
- Required points
- Relevant research

The generated content is designed to be:

- Informative
- Clear
- Technically accurate
- Well structured
- Markdown formatted
- Easy to read

---

### 5. Reducer / Combiner

After the individual sections are generated, the Reducer combines them into a single final Markdown document.

This creates one complete blog instead of separate independent sections.

---

### 6. SVG Diagram

A lightweight SVG diagram is generated locally to represent the workflow.

This avoids requiring an additional image-generation API and keeps the application lightweight.

---

## 🛠️ Tech Stack

| Technology | Purpose |
|---|---|
| Python | Core programming language |
| LangGraph | Agent workflow and state management |
| LangChain | LLM and tool integration |
| Groq | LLM inference |
| `openai/gpt-oss-20b` | Blog planning and generation |
| Tavily | Web research |
| Streamlit | Interactive frontend |
| Pydantic | Data validation |
| python-dotenv | Environment variable management |
| SVG | Local diagram generation |

---

## 📂 Project Structure

```text
Blog_writing_agent-/
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
git clone https://github.com/KhushTrivedi445/Blog_writing_agent-.git
```

```bash
cd Blog_writing_agent-
```

---

### 2. Create a virtual environment

#### Windows

```powershell
python -m venv venv
```

Activate the environment:

```powershell
.\venv\Scripts\Activate.ps1
```

#### Linux / macOS

```bash
python3 -m venv venv
```

```bash
source venv/bin/activate
```

---

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

Do not commit the `.env` file to GitHub.

The API keys should be stored only in environment variables.

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

The application then runs the agentic workflow and generates the final blog.

---

---

## 📌 Example

### Input

```text
Machine Learning
```

### Generated Plan

```text
Mastering Machine Learning: A Practical Guide

1. Fundamentals of Machine Learning
2. Core Mechanism: Gradient Descent
3. Implementation Workflow
4. Common Mistakes & Trade-offs
5. Conclusion & Next Steps
```

The system then:

```text
Topic
  ↓
Planning
  ↓
Research
  ↓
Writing
  ↓
Combining
  ↓
Final Blog
```

---

## 🎯 Why LangGraph?

LangGraph was selected to manage the multi-step agent workflow and state transitions.

Using LangGraph makes it possible to:

- Manage shared workflow state
- Create multiple agent stages
- Execute research tasks in parallel
- Control the execution flow
- Aggregate results using reducers
- Handle complex agent workflows
- Extend the application with additional agents

Instead of using a single LLM call, the project demonstrates a structured agentic workflow.

---

## 🤖 Why a Multi-Agent Architecture?

The blog generation process is divided into specialized stages:

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

Each stage has a dedicated responsibility.

This makes the application:

- Easier to maintain
- Easier to debug
- More modular
- More controllable
- Easier to extend
- More suitable for complex tasks

---

## 🔎 Research Strategy

Tavily provides external web information to the research agents.

The retrieved information is filtered and limited before being passed to the writer agents.

This helps:

- Keep prompts focused
- Reduce unnecessary context
- Control token usage
- Improve generation efficiency
- Provide topic-specific research

Research is primarily an internal stage of the workflow and is used to improve the generated blog.

---

## 💰 Token Optimization

The project is designed to use LLM resources efficiently.

The following strategies are used:

- Limited research results
- Controlled research context
- Section-based generation
- Concise prompts
- Controlled output length
- Retry only when required
- Local SVG generation instead of an external image-generation API

These decisions help keep the application lightweight and cost-efficient.

---

## 🛡️ Reliability & Error Handling

LLM and external API calls can occasionally fail or return incomplete responses.

The application therefore includes mechanisms such as:

- Retry handling
- Empty-response detection
- Fallback content
- API error handling
- State validation
- Controlled generation limits

If a section cannot be generated successfully, fallback content can be used instead of terminating the entire workflow.

---

## 🧩 Parallel Research Design

The research stage uses parallel execution for different blog sections.

Each research branch receives only the information required for its specific task.

Conceptually:

```text
ResearchPayload
├── task
└── topic
```

The research results are then collected into a shared reducer-backed state.

This allows multiple research branches to execute without conflicting state updates.

---

## 🖼️ Why Local SVG?

The workflow diagram is generated locally using SVG instead of requiring another image-generation service.

Advantages include:

- No additional API cost
- No image model dependency
- Fast generation
- Lightweight output
- Easy customization
- Suitable for technical diagrams

This keeps the project focused on the core agentic AI workflow.

---

## 📄 Generated Output

The final output is a Markdown blog containing:

- Blog title
- Structured sections
- Technical explanations
- Important points
- Examples where appropriate
- Markdown formatting
- Workflow diagram

The generated Markdown can also be downloaded from the Streamlit interface.

---

## 🔮 Future Improvements

Potential improvements include:

- Better research source ranking
- Multiple search queries per section
- Automatic citations
- Fact-checking agent
- Human-in-the-loop editing
- SEO optimization
- Keyword extraction
- Blog tone selection
- Blog quality evaluation
- Plagiarism detection
- Multiple LLM provider support
- Persistent blog history
- Cloud deployment
- Advanced image generation

---

## 🎓 Key Learning Outcomes

This project provides practical experience with:

- Agentic AI architecture
- LangGraph workflows
- Multi-agent systems
- Parallel agent execution
- State management
- Reducer-based aggregation
- Tool integration
- Web research
- Prompt engineering
- LLM application development
- Structured data validation
- Error handling
- Retry mechanisms
- Token optimization
- Streamlit development
- Git and GitHub

---

## 🎯 Project Objective

The objective of this project is to build a practical agentic AI system that can transform a simple topic into a researched, structured, and readable technical blog.

Rather than relying on one large LLM request, the system separates the process into planning, research, writing, and final assembly stages using LangGraph.

This demonstrates how multiple specialized AI agents can collaborate to solve a larger task.

---

## 👨‍💻 Author

**Khush Trivedi**

GitHub:

https://github.com/KhushTrivedi445

Project Repository:

https://github.com/KhushTrivedi445/Blog_writing_agent-

---

## 📜 License

This project is created for educational, portfolio, and demonstration purposes.
````