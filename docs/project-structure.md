# Project Structure

```
student-admission-agents/
│
├── frontend/                           # Client-side application (React / Next.js UI)
│   ├── src/
│   │   ├── components/                 # Reusable UI components
│   │   │   ├── chat/                   # Chat UI (messages, typing indicator, etc.)
│   │   │   ├── profile/                # Student profile input forms
│   │   │   ├── review/                 # Human-in-the-loop confirmation UI
│   │   │   └── layout/                 # Layout components (header, sidebar, etc.)
│   │   ├── hooks/                      # Custom React hooks (stateful logic)
│   │   ├── services/                   # API client (HTTP / WebSocket wrappers)
│   │   ├── store/                      # Global state management (Zustand / Redux)
│   │   ├── types/                      # TypeScript interfaces and types
│   │   └── pages/ / app/               # Routing layer (Next.js pages or app router)
│   ├── public/                         # Static assets
│   ├── .env.example                    # Environment variables template
│   └── package.json                    # Frontend dependencies
│ 
├── backend/
│   ├── app/
│   │
│   │   ├── api/                        # API layer (FastAPI endpoints)
│   │   │   ├── routes/
│   │   │   │   ├── chat.py             # Chat endpoints (POST /chat, WS /chat)
│   │   │   │   ├── profile.py          # Student profile CRUD APIs
│   │   │   │   └── application.py      # Application actions (submit, confirm)
│   │   │   └── deps.py                 # Dependency injection (auth, DB session)
│   │
│   │   ├── agents/                     # Atomic AI agents (no orchestration here)
│   │   │   ├── base.py                 # BaseAgent abstraction (common interface)
│   │   │   ├── researcher.py           # Extract info from news / external sources
│   │   │   ├── planner.py              # Generate admission plan/checklist
│   │   │   ├── advisor.py              # Explain and guide the student
│   │   │   └── application_agent.py    # Execute application-related actions
│   │
│   │   ├── workflows/                  # Orchestration layer (multi-agent coordination)
│   │   │   ├── admission_workflow.py   # End-to-end admission pipeline
│   │   │   ├── chat_workflow.py        # Conversational agent flow (entry point)
│   │   │   └── state.py                # Shared state object passed across agents
│   │
│   │   ├── tools/                      # Tools that agents can invoke
│   │   │   ├── web_search_tool.py      # Search external information (news, web)
│   │   │   ├── rag_tool.py             # Retrieval-Augmented Generation (vector DB)
│   │   │   ├── form_submit_tool.py     # Submit application forms
│   │   │   └── browser/                # Browser automation layer
│   │   │       ├── actions.py          # Low-level browser actions (click, fill, etc.)
│   │   │       └── session.py          # Browser session management
│   │
│   │   ├── prompts/                    # Prompt templates (decoupled from code)
│   │   │   ├── planner.txt             # Prompt for planning tasks
│   │   │   ├── researcher.txt          # Prompt for information extraction
│   │   │   ├── advisor.txt             # Prompt for student guidance
│   │   │   └── application.txt         # Prompt for execution logic
│   │
│   │   ├── services/                   # Shared infrastructure services
│   │   │   ├── llm/                 
│   │   │   │   ├── client.py           # Low-level LLM API client (OpenAI, etc.)
│   │   │   │   ├── completion.py       # Higher-level completion wrapper
│   │   │   │   └── router.py           # Model selection / routing logic
│   │   │   ├── memory.py               # Conversation and agent memory handling
│   │   │   └── embedding.py            # Embedding generation logic
│   │
│   │   ├── ingestion/                  # Data ingestion pipeline
│   │   │   ├── crawlers/               # Crawl data from sources (e.g., vnexpress)
│   │   │   ├── parsers/                # Parse raw HTML into structured data
│   │   │   └── normalizers/            # Clean and normalize extracted data
│   │
│   │   ├── models/                     # Database models (ORM - SQLAlchemy)
│   │   │   ├── student.py              # Student entity
│   │   │   ├── session.py              # Chat/session tracking
│   │   │   └── application.py          # Admission application data
│   │
│   │   ├── schemas/                    # Pydantic schemas (API contracts)
│   │   │   ├── student.py              # Request/response for student
│   │   │   ├── chat.py                 # Chat message schemas
│   │   │   ├── application.py          # Application request/response
│   │   │   └── agent_state.py          # Shared state structure for workflows
│   │
│   │   ├── tasks/                      # Background jobs (Celery / async workers)
│   │   │   ├── admission_tasks.py      # Long-running admission workflows
│   │   │   └── scraping_tasks.py       # Background scraping jobs
│   │
│   │   ├── core/                       # Core system configuration
│   │   │   ├── config.py               # App configuration (env, settings)
│   │   │   ├── security.py             # Authentication / authorization logic
│   │   │   └── logging.py              # Structured logging setup
│   │
│   │   └── main.py                     # FastAPI application entry point
│
│   ├── alembic/                        # Database migrations
│   ├── tests/                          # Test suite
│   │   ├── unit/                       # Unit tests (isolated logic)
│   │   └── integration/                # Integration tests (API, DB, workflows)
│   ├── .env.example                    # Backend environment template
│   ├── Dockerfile                      # Backend container definition
│   └── requirements.txt                # Python dependencies
│
├── infra/                              # Infrastructure & deployment configs
│   ├── docker-compose.yml              # Main services (API, DB, Redis, worker)
│   ├── docker-compose.dev.yml          # Dev overrides
│   └── nginx/                          # Reverse proxy config
│
├── docs/                               # Project documentation
│   ├── architecture.md                 # System design & architecture explanation
│   └── api-spec.yaml                   # OpenAPI specification
│
├── scripts/                            # Utility scripts (dev, setup, migration)
├── README.md                           # Project overview & setup guide
└── .env                                # Environment variables (local only)
```

---
# Frontend Architecture


---
# Backend Architecture

The backend is built with **FastAPI** and follows a modular, layered architecture designed for a **multi-agent system**. Each layer has a clear responsibility to ensure scalability, maintainability, and separation of concerns.

## Description

### 1. API Layer (`app/api/`)

Handles all incoming client requests and exposes REST/WebSocket endpoints.

* `routes/`
  Defines API endpoints such as chat, student profile management, and application actions.
* `deps.py`
  Provides dependency injection (e.g., database session, authentication).

**Responsibility:**
Translate HTTP/WebSocket requests into internal service or workflow calls. This layer should remain thin and contain minimal business logic.


### 2. Agents Layer (`app/agents/`)

Contains atomic AI agents responsible for specific tasks.

* `base.py`
  Defines a common interface for all agents.
* `researcher.py`
  Extracts and retrieves information from external sources (e.g., news, web).
* `planner.py`
  Generates structured plans or checklists for admission.
* `advisor.py`
  Provides explanations and guidance to the student.
* `application_agent.py`
  Handles application-related execution logic.

**Responsibility:**
Encapsulate individual reasoning capabilities. Agents do not coordinate with each other directly.


### 3. Workflow Layer (`app/workflows/`)

Implements orchestration logic across multiple agents.

* `admission_workflow.py`
  End-to-end pipeline for processing admission tasks.
* `chat_workflow.py`
  Entry point for conversational interactions.
* `state.py`
  Shared state object passed between agents during execution.

**Responsibility:**
Control execution flow, manage state, and coordinate agent interactions.


### 4. Tools Layer (`app/tools/`)

Provides capabilities that agents can invoke to interact with external systems.

* `web_search_tool.py`
  Retrieves external information from the web.
* `rag_tool.py`
  Performs retrieval-augmented generation using a vector database.
* `form_submit_tool.py`
  Handles form submission for applications.
* `browser/`

  * `actions.py`: Low-level browser actions (click, fill, etc.)
  * `session.py`: Browser session management

**Responsibility:**
Abstract external operations into reusable tools that agents can call.


### 5. Prompts (`app/prompts/`)

Stores prompt templates used by agents.

* Separate prompt files for each agent role (planner, researcher, advisor, etc.)

**Responsibility:**
Decouple prompt engineering from code to allow easier iteration and tuning.


### 6. Services Layer (`app/services/`)

Contains shared infrastructure logic.

* `llm/`

  * `client.py`: Low-level LLM API communication
  * `completion.py`: Higher-level wrapper for completions
  * `router.py`: Model selection and routing logic
* `memory.py`
  Manages conversation and agent memory.
* `embedding.py`
  Handles embedding generation for RAG.

**Responsibility:**
Provide reusable services used across agents and workflows.


### 7. Ingestion Layer (`app/ingestion/`)

Handles data collection and preprocessing.

* `crawlers/`
  Collect raw data from sources (e.g., news websites).
* `parsers/`
  Convert raw HTML into structured data.
* `normalizers/`
  Clean and standardize extracted data.

**Responsibility:**
Transform external data into a usable format for the system.


### 8. Data Layer

#### Models (`app/models/`)

Defines database entities using an ORM.

* `student.py`
* `session.py`
* `application.py`

#### Schemas (`app/schemas/`)

Defines request/response contracts using Pydantic.

* API schemas (student, chat, application)
* `agent_state.py`: shared state structure for workflows

**Responsibility:**
Ensure consistent data representation between internal logic and external APIs.


### 9. Background Tasks (`app/tasks/`)

Handles asynchronous and long-running operations.

* `admission_tasks.py`
  Runs admission workflows in the background.
* `scraping_tasks.py`
  Handles background data ingestion.

**Responsibility:**
Offload heavy or time-consuming tasks from the main API thread.


### 10. Core Module (`app/core/`)

Provides foundational system utilities.

* `config.py`
  Application configuration management.
* `security.py`
  Authentication and authorization logic.
* `logging.py`
  Structured logging setup.

**Responsibility:**
Centralize cross-cutting concerns used across the backend.


### 11. Entry Point (`app/main.py`)

Initializes and runs the FastAPI application.

**Responsibility:**
Application bootstrap, route registration, and middleware setup.