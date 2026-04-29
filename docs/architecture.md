# System Architecture
The architecture relies on a central Orchestrator that delegates tasks to specialized sub-agents. The pipeline is: **Information → Planning → Explanation → Execution**.

```mermaid
flowchart TD
  User([Student / User]) --> Orchestrator[Orchestrator Agent<br/>Receives requests, plans, coordinates]

  Orchestrator --> Researcher[Researcher Agent<br/>Collects info, RAG/search, extract data]
  Orchestrator --> Planner[Planner Agent<br/>Analyzes profile, creates plan]
  Orchestrator --> Advisor[Advisor Agent<br/>Explains plan, gives guidance]
  Orchestrator --> Application[Application Agent<br/>Fills forms, submits application]

  subgraph Tool Layer
      WebScraper[Web Scraper<br/>Browser parser]
      MemoryRAG[Memory / RAG<br/>Store student context]
      LLMReasoning[LLM Reasoning<br/>GPT-4o / Claude]
      BrowserAuto[Browser Auto<br/>Playwright / Selenium]
  end

  Researcher --> WebScraper
  Researcher --> MemoryRAG
  Planner --> MemoryRAG
  Planner --> LLMReasoning
  Advisor --> LLMReasoning
  Application --> BrowserAuto
  Application --> LLMReasoning

  subgraph Infrastructure
      VDB[(Vector DB<br/>Chroma)]
      SQL[(Relational DB<br/>PostgreSQL)]
      Celery[[Task Queue<br/>Celery + Redis]]
      API{{FastAPI Backend}}
  end

  WebScraper -.-> API
  MemoryRAG -.-> VDB
  BrowserAuto -.-> Celery
  LLMReasoning -.-> API
  API -.-> SQL
```


## Execution Flow

```mermaid
flowchart TD

%% ===== CLIENT =====
A[Client / Frontend] -->|HTTP / WebSocket| B[API Layer]

%% ===== API LAYER =====
B --> C[Route Handler]
C --> D[Workflow Dispatcher]

%% ===== WORKFLOW LAYER =====
D --> E[Workflow Engine]

%% ===== AGENT FLOW =====
E --> F[Researcher Agent]
F --> G[Planner Agent]
G --> H[Advisor Agent]
H --> I[Application Agent]

%% ===== TOOL USAGE =====
F --> T1[Web Search Tool]
F --> T2[RAG Tool]

I --> T3[Form Submit Tool]
I --> T4[Browser Automation]

%% ===== SERVICES =====
F --> S1[LLM Service]
G --> S1
H --> S1
I --> S1

F --> S2[Memory Service]
G --> S2
H --> S2

%% ===== DATA LAYER =====
S2 --> DB[(Database)]
T2 --> VS[(Vector Store)]

%% ===== ASYNC TASKS =====
E --> Q[Task Queue]
Q --> W[Worker]
W --> I

%% ===== RESPONSE =====
I --> R[Result Aggregation]
R --> B
B --> A
```
##  Chat + HITL Flow

```mermaid
sequenceDiagram

participant User
participant FE as Frontend
participant API
participant WF as Workflow
participant Agents
participant Tools

User->>FE: Input (news link / question)
FE->>API: POST /chat

API->>WF: Start chat_workflow
WF->>Agents: Researcher

Agents->>Tools: Web Search / RAG
Tools-->>Agents: Data

WF->>Agents: Planner
WF->>Agents: Advisor

alt Requires submission
    WF->>Agents: Application Agent
    Agents->>Tools: Browser Automation
    Tools-->>Agents: Filled form
    
    API-->>FE: Request confirmation (HITL)
    FE->>User: Show review UI
    
    User->>FE: Confirm
    FE->>API: POST /confirm
    
    API->>WF: Resume workflow
    WF->>Tools: Submit form
end

WF-->>API: Final result
API-->>FE: Response
FE-->>User: Display result
```