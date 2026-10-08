# Generative AI with LangChain

A hands-on learning series for building with chat models, prompt templates, conversation history, structured outputs, tools, agents, document loaders, embeddings, and retrieval-augmented generation (RAG).

**Course author:** Jayneel Girnara

The repository is organized as a progressive set of Jupyter notebooks, followed by small example apps. Work through the notebooks in order; each lesson introduces a building block used by later RAG examples.

## Learning path

| Notebook | Topic |
| --- | --- |
| [00 — Course foundations](notebooks/00_course_foundations.ipynb) | Core concepts and course map |
| [01 — Chat models and message prompts](notebooks/01_langchain_with_openai.ipynb) | Compare provider integrations and role-based messages |
| [02 — Prompt templates and chains](notebooks/02_prompt_chains.ipynb) | Parameterized prompts and LCEL |
| [03 — Conversation history](notebooks/03_basic_memory_with_langchain.ipynb) | Stateless calls and manually managed history |
| [04 — Structured output](notebooks/04_structured_outputs.ipynb) | Pydantic schemas for model responses |
| [05 — Ollama](notebooks/05_ollama_app.ipynb) | Run a model through Ollama |
| [06 — Groq](notebooks/06_groq_langchain.ipynb) | Call a Groq-hosted model |
| [07 — Streaming](notebooks/07_streaming_response.ipynb) | Consume generated text in chunks |
| [08 — Agents and tools](notebooks/08_basic_agents.ipynb) | Let an agent select typed Python tools |
| [09 — Search agent](notebooks/09_google_search_agent.ipynb) | Add Google Serper search to an agent |
| [10 — Document loaders](notebooks/10_rag_load_document.ipynb) | Load text, PDF, CSV, web, and Wikipedia sources |
| [11 — Text splitting](notebooks/11_rag_splitters.ipynb) | Prepare documents as overlapping chunks |
| [12 — Embeddings and vector search](notebooks/12_vector_embeddings.ipynb) | Embed, index, and retrieve text |
| [13 — PDF RAG chain](notebooks/13_rag_based_pdf_qna_bot.ipynb) | Build a retrieval-grounded PDF Q&A chain |
| [14 — Agentic RAG](notebooks/14_agentic_rag.ipynb) | Give an agent a document retrieval tool |
| [15 — Pydantic validation](notebooks/15_pydantic_data_validation.ipynb) | Validate typed and nested data |

## Example applications

The `apps/` directory contains small examples that apply the notebook concepts:

- `1_qna_bot.py` — Streamlit chat UI with Google Gemini.
- `2_google_bot.py` — terminal agent with web search.
- `3_qna_bot_withgroq.py` — Streamlit chat UI with Groq and web search.
- `4_sql_agent.py` — Streamlit task assistant backed by SQLite.
- `5_rag_agent.py` — Streamlit PDF upload and retrieval-augmented chat.

These are educational examples, not production-ready services. Review each app's tools, data handling, provider costs, and safety limitations before using it with real users or sensitive information.

## Setup

Use Python 3.10 or newer (Python 3.12 is recommended). From the repository root:

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
cp .env.example .env
```

Add only the API keys needed for the examples you intend to run to `.env`. The `.env` file is ignored by Git. Never paste a real key into a notebook or commit it.
Direct dependencies are pinned in `requirements.txt` so learners install the same tested package versions.

| Environment variable | Used for |
| --- | --- |
| `OPENAI_API_KEY` | OpenAI chat and embedding examples |
| `GOOGLE_API_KEY` | Google Gemini example |
| `ANTHROPIC_API_KEY` | Anthropic example |
| `GROQ_API_KEY` | Groq chat examples |
| `SERPER_API_KEY` | Google Serper search examples |

Provider model names, availability, pricing, quotas, and API behavior can change. Check the relevant provider account and update the example configuration when needed. Ollama examples also require Ollama to be installed and running with the configured model available.

## Open the notebooks

Start Jupyter from the repository root so notebook paths to `data/` resolve:

```bash
jupyter lab notebooks/
```

Read each notebook's introduction and run cells from top to bottom. Some examples make external API or web requests; local text splitting and Pydantic examples do not require model API credentials.

Run the offline repository checks from the project root with:

```bash
python -m unittest discover -s tests -v
```

## Run an app

For example, launch the PDF RAG app from the repository root:

```bash
streamlit run apps/5_rag_agent.py
```

Other Streamlit examples can be launched by replacing the file path with the relevant script in `apps/`. `2_google_bot.py` is a terminal program and can be run with:

```bash
python apps/2_google_bot.py
```

The PDF RAG app creates `doc_files/` for uploaded files at runtime. Do not upload sensitive documents unless you understand that document text may be sent to the configured embedding and model providers.

## Project layout

```text
.
├── apps/       # Small Streamlit and terminal applications
├── data/       # Sample documents used by selected lessons
├── notebooks/  # Progressive, annotated Jupyter lessons
└── requirements.txt
```

## Notes

- Model output can be incorrect. Verify results against their source, especially in retrieval and agent examples.
- API-backed examples may send prompts and document excerpts to third-party providers and may incur charges.
- The medical-document examples are educational demonstrations, not medical advice or clinical tools.
- `doc_files/` is runtime upload storage and is intentionally excluded from version control.
- This project is licensed under the MIT License; see [LICENSE](LICENSE). Sample documents and third-party dependencies may have separate terms.
