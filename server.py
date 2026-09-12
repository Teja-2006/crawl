import os
import sys
import json
import asyncio
from datetime import datetime
from typing import List, Dict, Any, Optional
import httpx
import trafilatura
from ddgs import DDGS
import ollama
from fastapi import FastAPI, BackgroundTasks, HTTPException
from fastapi.responses import HTMLResponse, JSONResponse, StreamingResponse
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

# Configure Windows terminal encoding
if sys.platform == "win32":
    try:
        sys.stdout.reconfigure(encoding="utf-8")
        sys.stderr.reconfigure(encoding="utf-8")
    except Exception:
        pass

app = FastAPI(title="Hermes 3 Cloud API Research Dashboard")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
REPORTS_DIR = os.path.join(BASE_DIR, "research_reports")
os.makedirs(REPORTS_DIR, exist_ok=True)

MODEL_NAME = "hermes3"
OLLAMA_HOST = "http://127.0.0.1:11434"
ollama_client = ollama.Client(host=OLLAMA_HOST)

# Global State & Event Queue for SSE
class AgentState:
    def __init__(self):
        self.is_running = False
        self.current_topic = ""
        self.current_keywords: List[str] = []
        self.current_step = 0
        self.max_steps = 12
        self.events: List[Dict[str, Any]] = []
        self.listeners: List[asyncio.Queue] = []
        self.current_report_file: Optional[str] = None
        self.final_content: str = ""

state = AgentState()

async def broadcast(event_type: str, data: Any):
    event = {
        "id": len(state.events) + 1,
        "type": event_type,
        "timestamp": datetime.now().strftime("%H:%M:%S"),
        "data": data
    }
    state.events.append(event)
    for q in state.listeners:
        await q.put(event)

# -------------------------------------------------------------
# TOOLS IMPLEMENTED FOR ASYNC / DASHBOARD STREAMING
# -------------------------------------------------------------

async def tool_search_papers(query: str, source: str = "all", max_results: int = 5) -> str:
    source_filters = {
        "arxiv": "site:arxiv.org",
        "academic": "(site:arxiv.org OR site:openreview.net OR site:semanticscholar.org OR site:ieee.org OR site:acm.org OR site:springer.com)",
        "cloud_engineering": "(site:usenix.org OR site:aws.amazon.com OR site:cloud.google.com OR site:cloudflare.com OR site:datadoghq.com)",
        "all": "(site:arxiv.org OR site:openreview.net OR site:semanticscholar.org OR site:usenix.org OR site:ieee.org OR filetype:pdf)"
    }
    site_query = source_filters.get(source.lower(), source_filters["all"])
    full_query = f"{site_query} {query}".strip()

    await broadcast("search_start", {"query": query, "filter": source})

    def run_ddgs():
        res = []
        with DDGS() as ddgs:
            for item in ddgs.text(full_query, max_results=max_results):
                res.append({
                    "title": item.get("title", ""),
                    "url": item.get("href", ""),
                    "snippet": item.get("body", "")
                })
        if not res:
            with DDGS() as ddgs:
                fallback_query = f"{query} research paper anomaly detection cloud API"
                for item in ddgs.text(fallback_query, max_results=max_results):
                    res.append({
                        "title": item.get("title", ""),
                        "url": item.get("href", ""),
                        "snippet": item.get("body", "")
                    })
        return res

    loop = asyncio.get_running_loop()
    results = await loop.run_in_executor(None, run_ddgs)
    
    await broadcast("search_results", {"count": len(results), "results": results})
    return json.dumps(results if results else {"status": "no_results", "message": f"No papers found for {query}"})

async def tool_crawl_paper(url: str, max_chars: int = 8000) -> str:
    await broadcast("crawl_start", {"url": url})

    def run_crawl():
        downloaded = trafilatura.fetch_url(url)
        if not downloaded:
            headers = {
                "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/124.0.0.0 Safari/537.36",
                "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8"
            }
            with httpx.Client(timeout=25.0, follow_redirects=True, headers=headers) as c:
                resp = c.get(url)
                if resp.status_code == 200:
                    downloaded = resp.text
        
        if not downloaded:
            return None, "Failed to download paper HTML"

        extracted = trafilatura.extract(downloaded, url=url, include_links=True, include_tables=True, output_format="txt")
        return extracted, None

    loop = asyncio.get_running_loop()
    extracted, err = await loop.run_in_executor(None, run_crawl)

    if err or not extracted or len(extracted.strip()) < 50:
        await broadcast("crawl_error", {"url": url, "error": err or "Content too short / paywalled"})
        return json.dumps({"warning": "Content minimal or protected"})

    clean_content = extracted[:max_chars]
    await broadcast("crawl_done", {
        "url": url,
        "length": len(clean_content),
        "preview": clean_content[:350] + "..."
    })
    return json.dumps({"url": url, "length": len(clean_content), "content": clean_content})

async def tool_save_report(topic: str, content: str) -> str:
    safe_topic = "".join(c if c.isalnum() or c in " _-" else "_" for c in topic)[:60].strip()
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    filename = f"{safe_topic}_{timestamp}.md"
    file_path = os.path.join(REPORTS_DIR, filename)

    header = (
        f"# Academic Research Report: {topic}\n"
        f"**Date:** {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n"
        f"**Domain:** AI-Driven Anomaly Detection for Cloud-Based APIs & Microservices\n"
        f"**Model:** {MODEL_NAME} (Local Ollama via RTX 5070 Ti 16GB)\n"
        f"**Generated by:** Autonomous Local Hermes 3 Research Agent\n\n"
        f"---\n\n"
    )

    with open(file_path, "w", encoding="utf-8") as f:
        f.write(header + content)

    state.current_report_file = filename
    await broadcast("report_saved", {"filename": filename, "path": file_path})
    return f"Report saved to {filename}"

TOOLS_SCHEMA = [
    {
        "type": "function",
        "function": {
            "name": "search_research_papers",
            "description": "Searches for academic research papers and preprints on AI-driven cloud API anomaly detection.",
            "parameters": {
                "type": "object",
                "properties": {
                    "query": {"type": "string", "description": "Academic search query keywords."},
                    "source": {"type": "string", "enum": ["all", "arxiv", "academic", "cloud_engineering"], "description": "Repository filter."},
                    "max_results": {"type": "integer", "description": "Max results (default 5)."}
                },
                "required": ["query"]
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "crawl_paper_or_webpage",
            "description": "Crawls and extracts full paper abstracts, methodology, formulas, and benchmark results.",
            "parameters": {
                "type": "object",
                "properties": {
                    "url": {"type": "string", "description": "Full URL of academic paper or arXiv page."}
                },
                "required": ["url"]
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "save_research_report",
            "description": "Saves the completed, comprehensive academic literature review into a structured Markdown document.",
            "parameters": {
                "type": "object",
                "properties": {
                    "topic": {"type": "string", "description": "Research topic or title."},
                    "content": {"type": "string", "description": "The complete markdown report."}
                },
                "required": ["topic", "content"]
            }
        }
    }
]

SYSTEM_PROMPT = """You are an elite Computer Science & Cloud Security Researcher specialized in:
"AI-Driven Anomaly Detection for Cloud-Based APIs and Microservices".

RESEARCH & REPORTING PROTOCOL:
1. Formulate targeted search queries using `search_research_papers`. Focus on deep learning, GNNs for API dependency call graphs, LLM agents (e.g. CloudAnoAgent), autoencoders, Isolation Forests, and eBPF telemetry.
2. MANDATORY STEP: Call `crawl_paper_or_webpage` on at least 2 relevant paper URLs or arXiv pages discovered from your search. Read their actual abstracts, architectures, datasets, and benchmark results.
3. Synthesize your research into a structured, publication-grade Literature Review:
   - # Executive Summary & Problem Scope
   - ## 1. Core API Anomaly Types (Latency spikes, schema drift, token exhaustion, malicious injection, cascading failures)
   - ## 2. AI/ML Architectures (Autoencoders, Isolation Forest, GNNs, LLM Agents like CloudAnoAgent)
   - ## 3. Benchmark Datasets & Metrics (GAIA, Alibaba Microservices, F1-Score, FPR)
   - ## 4. Cloud Deployment (Envoy sidecars, eBPF telemetry, API Gateways)
   - ## 5. 📚 Verified Citations & Source References:
     You MUST list every paper/article found with exact clickable markdown links and key takeaway:
     - [Paper Title](Exact URL) - Authors/Year. Key Takeaway: Explanation of findings.
4. Call `save_research_report` to save the comprehensive review.
5. Provide a clear summary to the user outlining the top breakthroughs and citations.
"""

# -------------------------------------------------------------
# ASYNC AGENT WORKFLOW RUNNER
# -------------------------------------------------------------

async def run_agent_task(topic: str, keywords: Optional[List[str]] = None, max_steps: int = 12):
    state.is_running = True
    state.current_topic = topic
    state.current_keywords = keywords or []
    state.current_step = 0
    state.max_steps = max_steps
    state.events = []
    state.current_report_file = None
    state.final_content = ""

    await broadcast("agent_started", {
        "topic": topic,
        "keywords": state.current_keywords,
        "max_steps": max_steps
    })

    user_query = f"Please conduct in-depth academic research and literature survey on: '{topic}'."
    if keywords:
        user_query += f" Strongly focus your investigation and paper searches on the following keywords: {', '.join(keywords)}."

    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": user_query}
    ]

    report_saved = False
    step = 0

    try:
        while step < max_steps:
            step += 1
            state.current_step = step
            await broadcast("step_started", {"step": step, "max_steps": max_steps})

            # Call Ollama in thread pool to avoid blocking async loop
            loop = asyncio.get_running_loop()
            def chat_call():
                return ollama_client.chat(model=MODEL_NAME, messages=messages, tools=TOOLS_SCHEMA)

            response = await loop.run_in_executor(None, chat_call)
            msg = response.get("message") if isinstance(response, dict) else response.message
            messages.append(msg)

            tool_calls = msg.get("tool_calls") if isinstance(msg, dict) else getattr(msg, "tool_calls", None)

            if tool_calls:
                for tc in tool_calls:
                    if isinstance(tc, dict):
                        fn_name = tc.get("function", {}).get("name")
                        raw_args = tc.get("function", {}).get("arguments", {})
                    else:
                        fn_name = tc.function.name
                        raw_args = tc.function.arguments

                    if isinstance(raw_args, str):
                        try:
                            args = json.loads(raw_args)
                        except Exception:
                            args = {}
                    elif isinstance(raw_args, dict):
                        args = raw_args
                    else:
                        args = dict(raw_args)

                    await broadcast("tool_call", {"tool": fn_name, "arguments": args})

                    # Execute respective tool
                    if fn_name == "search_research_papers":
                        result = await tool_search_papers(**args)
                    elif fn_name == "crawl_paper_or_webpage":
                        result = await tool_crawl_paper(**args)
                    elif fn_name == "save_research_report":
                        report_saved = True
                        result = await tool_save_report(**args)
                    else:
                        result = f"Error: Tool '{fn_name}' not found."

                    messages.append({"role": "tool", "content": str(result)})
            else:
                content = msg.get("content") if isinstance(msg, dict) else getattr(msg, "content", "")
                if not content or not content.strip():
                    await broadcast("agent_thinking", {"message": "Hermes is synthesizing crawled findings..."})
                    messages.append({
                        "role": "user",
                        "content": "Please synthesize all your crawled research findings and save or present your final literature review with citations."
                    })
                    continue

                state.final_content = content
                if not report_saved and len(content.strip()) > 100:
                    await tool_save_report(topic, content)

                await broadcast("agent_completed", {
                    "summary": content,
                    "report_file": state.current_report_file
                })
                break

    except Exception as e:
        await broadcast("agent_error", {"error": str(e)})
    finally:
        state.is_running = False
        await broadcast("agent_idle", {})

# -------------------------------------------------------------
# REST API & DASHBOARD ENDPOINTS
# -------------------------------------------------------------

class ResearchRequest(BaseModel):
    topic: str
    keywords: Optional[List[str]] = None
    max_steps: Optional[int] = 12

@app.post("/api/start")
async def start_research(req: ResearchRequest, background_tasks: BackgroundTasks):
    if state.is_running:
        raise HTTPException(status_code=400, detail="A research task is already currently running.")
    
    topic = req.topic.strip() or "AI-Driven Anomaly Detection for Cloud-Based APIs"
    background_tasks.add_task(run_agent_task, topic, req.keywords, req.max_steps or 12)
    return {"status": "started", "topic": topic, "keywords": req.keywords or []}

@app.get("/api/status")
async def get_status():
    return {
        "is_running": state.is_running,
        "current_topic": state.current_topic,
        "current_keywords": state.current_keywords,
        "current_step": state.current_step,
        "max_steps": state.max_steps,
        "current_report_file": state.current_report_file,
        "events_count": len(state.events)
    }

@app.get("/api/stream")
async def sse_stream():
    """Server-Sent Events endpoint for real-time live activity streaming."""
    queue = asyncio.Queue()
    state.listeners.append(queue)

    async def event_generator():
        try:
            for ev in state.events:
                yield f"data: {json.dumps(ev)}\n\n"
            
            while True:
                ev = await queue.get()
                yield f"data: {json.dumps(ev)}\n\n"
        except asyncio.CancelledError:
            pass
        finally:
            if queue in state.listeners:
                state.listeners.remove(queue)

    return StreamingResponse(event_generator(), media_type="text/event-stream")

@app.get("/api/reports")
async def list_reports():
    reports = []
    if os.path.exists(REPORTS_DIR):
        for f in sorted(os.listdir(REPORTS_DIR), reverse=True):
            if f.endswith(".md"):
                fp = os.path.join(REPORTS_DIR, f)
                stat = os.stat(fp)
                reports.append({
                    "filename": f,
                    "size": stat.st_size,
                    "created_at": datetime.fromtimestamp(stat.st_mtime).strftime("%Y-%m-%d %H:%M:%S")
                })
    return reports

@app.get("/api/reports/{filename}")
async def get_report_content(filename: str):
    clean_name = os.path.basename(filename)
    fp = os.path.join(REPORTS_DIR, clean_name)
    if not os.path.exists(fp):
        raise HTTPException(status_code=404, detail="Report not found")
    with open(fp, "r", encoding="utf-8") as f:
        content = f.read()
    return {"filename": clean_name, "content": content}

# -------------------------------------------------------------
# DASHBOARD HTML INTERFACE WITH ACTION BUTTONS & CITATIONS
# -------------------------------------------------------------

DASHBOARD_HTML = """<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="UTF-8">
  <meta name="viewport" content="width=device-width, initial-scale=1.0">
  <title>AI Cloud API Anomaly Detection - Research Dashboard</title>
  <script src="https://cdn.tailwindcss.com"></script>
  <script src="https://cdn.jsdelivr.net/npm/marked/marked.min.js"></script>
  <link rel="stylesheet" href="https://cdnjs.cloudflare.com/ajax/libs/font-awesome/6.4.0/css/all.min.css">
  <style>
    body { background-color: #0b0f17; color: #c9d1d9; font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, sans-serif; }
    .custom-scroll::-webkit-scrollbar { width: 6px; }
    .custom-scroll::-webkit-scrollbar-track { background: #161b22; }
    .custom-scroll::-webkit-scrollbar-thumb { background: #30363d; border-radius: 3px; }
    .markdown-body h1 { font-size: 1.5rem; font-weight: bold; margin-bottom: 0.8rem; color: #58a6ff; border-bottom: 1px solid #30363d; padding-bottom: 0.4rem; }
    .markdown-body h2 { font-size: 1.25rem; font-weight: bold; margin-top: 1.2rem; margin-bottom: 0.5rem; color: #79c0ff; }
    .markdown-body h3 { font-size: 1.05rem; font-weight: 600; margin-top: 1rem; margin-bottom: 0.3rem; color: #a5d6ff; }
    .markdown-body p { margin-bottom: 0.8rem; line-height: 1.6; }
    .markdown-body ul, .markdown-body ol { margin-left: 1.4rem; margin-bottom: 0.8rem; list-style-type: disc; }
    .markdown-body a { color: #58a6ff; text-decoration: underline; }
    .markdown-body a:hover { color: #79c0ff; }
    .markdown-body code { background: #1f242c; padding: 0.15rem 0.4rem; border-radius: 4px; font-size: 0.85rem; color: #f0883e; }
    .markdown-body pre code { display: block; padding: 0.8rem; overflow-x: auto; background: #161b22; color: #c9d1d9; }
    .markdown-body blockquote { border-left: 4px solid #388bfd; padding-left: 1rem; color: #8b949e; margin-bottom: 0.8rem; }
    .tag-active { background-color: #1f6feb !important; color: white !important; border-color: #388bfd !important; }
  </style>
</head>
<body class="min-h-screen flex flex-col">

  <!-- TOP NAVIGATION -->
  <header class="bg-[#111620] border-b border-[#21262d] px-6 py-3 flex items-center justify-between sticky top-0 z-50 shadow-md">
    <div class="flex items-center space-x-3">
      <div class="bg-gradient-to-tr from-blue-600 to-indigo-600 text-white p-2.5 rounded-xl shadow-md shadow-blue-500/10">
        <i class="fa-solid fa-radar text-lg"></i>
      </div>
      <div>
        <h1 class="text-base font-bold text-white flex items-center gap-2">
          AI Cloud API Anomaly Detection Research Crawler
          <span class="text-[11px] bg-emerald-950 text-emerald-400 border border-emerald-800/80 px-2 py-0.5 rounded-full font-mono">RTX 5070 Ti (16GB)</span>
        </h1>
        <p class="text-xs text-gray-400">Autonomous Web Scout, Literature Survey & Citation Extractor</p>
      </div>
    </div>

    <!-- Live Status & Telemetry -->
    <div class="flex items-center space-x-3 text-xs font-mono">
      <div class="flex items-center space-x-2 bg-[#161b22] px-3 py-1.5 rounded-lg border border-[#30363d]">
        <span class="relative flex h-2.5 w-2.5">
          <span id="pulseRing" class="hidden absolute inline-flex h-full w-full rounded-full bg-emerald-400 opacity-75 animate-ping"></span>
          <span id="statusDot" class="relative inline-flex rounded-full h-2.5 w-2.5 bg-gray-500"></span>
        </span>
        <span id="statusText" class="text-gray-300 font-semibold tracking-wide">IDLE</span>
      </div>
      <div class="hidden sm:flex items-center space-x-1.5 bg-[#161b22] px-3 py-1.5 rounded-lg border border-[#30363d] text-yellow-400">
        <i class="fa-solid fa-microchip"></i>
        <span>Hermes 3 8B</span>
      </div>
      <div class="hidden md:flex items-center space-x-1.5 bg-[#161b22] px-3 py-1.5 rounded-lg border border-[#30363d] text-emerald-400">
        <i class="fa-solid fa-bolt"></i>
        <span>Local Ollama</span>
      </div>
    </div>
  </header>

  <!-- DASHBOARD WORKSPACE GRID -->
  <div class="flex-1 grid grid-cols-1 xl:grid-cols-12 gap-6 p-6 max-w-[1920px] w-full mx-auto">
    
    <!-- LEFT PANEL: ACTION CONTROLS, KEYWORDS & LIVE STREAM (7 cols) -->
    <div class="xl:col-span-7 flex flex-col space-y-6">
      
      <!-- ACTION BUTTONS & KEYWORD SCOUT TOOLBAR -->
      <div class="bg-[#111620] border border-[#21262d] rounded-2xl p-5 shadow-xl space-y-4">
        
        <!-- Header -->
        <div class="flex items-center justify-between">
          <div class="flex items-center space-x-2">
            <i class="fa-solid fa-bolt-lightning text-amber-400"></i>
            <h2 class="text-xs font-bold uppercase tracking-wider text-gray-200">One-Click Scout Action Buttons</h2>
          </div>
          <span class="text-[11px] text-gray-500 font-mono">100% Free • Direct Web & arXiv Crawl</span>
        </div>

        <!-- 5 PRE-CONFIGURED ACTION BUTTONS -->
        <div class="grid grid-cols-1 sm:grid-cols-2 lg:grid-cols-3 gap-2.5">
          
          <button onclick="runPresetAction('scout_arxiv')" 
            class="flex items-start p-3 bg-[#161b22] hover:bg-[#1c212a] border border-[#30363d] hover:border-blue-500/70 rounded-xl transition text-left group">
            <div class="p-2 bg-blue-500/10 text-blue-400 rounded-lg mr-2.5 group-hover:scale-105 transition">
              <i class="fa-solid fa-graduation-cap text-base"></i>
            </div>
            <div>
              <div class="text-xs font-semibold text-white group-hover:text-blue-400">Scout arXiv Preprints</div>
              <div class="text-[11px] text-gray-400 mt-0.5">Fresh 2025-2026 preprints & abstracts</div>
            </div>
          </button>

          <button onclick="runPresetAction('gnn_microservices')" 
            class="flex items-start p-3 bg-[#161b22] hover:bg-[#1c212a] border border-[#30363d] hover:border-purple-500/70 rounded-xl transition text-left group">
            <div class="p-2 bg-purple-500/10 text-purple-400 rounded-lg mr-2.5 group-hover:scale-105 transition">
              <i class="fa-solid fa-diagram-project text-base"></i>
            </div>
            <div>
              <div class="text-xs font-semibold text-white group-hover:text-purple-400">GNNs & API Call Graphs</div>
              <div class="text-[11px] text-gray-400 mt-0.5">Microservice topology & dependencies</div>
            </div>
          </button>

          <button onclick="runPresetAction('llm_sre_agents')" 
            class="flex items-start p-3 bg-[#161b22] hover:bg-[#1c212a] border border-[#30363d] hover:border-emerald-500/70 rounded-xl transition text-left group">
            <div class="p-2 bg-emerald-500/10 text-emerald-400 rounded-lg mr-2.5 group-hover:scale-105 transition">
              <i class="fa-solid fa-robot text-base"></i>
            </div>
            <div>
              <div class="text-xs font-semibold text-white group-hover:text-emerald-400">LLM SRE Agents</div>
              <div class="text-[11px] text-gray-400 mt-0.5">CloudAnoAgent & autonomous log agents</div>
            </div>
          </button>

          <button onclick="runPresetAction('ebpf_telemetry')" 
            class="flex items-start p-3 bg-[#161b22] hover:bg-[#1c212a] border border-[#30363d] hover:border-amber-500/70 rounded-xl transition text-left group">
            <div class="p-2 bg-amber-500/10 text-amber-400 rounded-lg mr-2.5 group-hover:scale-105 transition">
              <i class="fa-solid fa-network-wired text-base"></i>
            </div>
            <div>
              <div class="text-xs font-semibold text-white group-hover:text-amber-400">eBPF & Kernel Telemetry</div>
              <div class="text-[11px] text-gray-400 mt-0.5">Real-time low latency packet tracing</div>
            </div>
          </button>

          <button onclick="runPresetAction('api_security_abuse')" 
            class="flex items-start p-3 bg-[#161b22] hover:bg-[#1c212a] border border-[#30363d] hover:border-red-500/70 rounded-xl transition text-left group">
            <div class="p-2 bg-red-500/10 text-red-400 rounded-lg mr-2.5 group-hover:scale-105 transition">
              <i class="fa-solid fa-shield-halved text-base"></i>
            </div>
            <div>
              <div class="text-xs font-semibold text-white group-hover:text-red-400">API Abuse & Zero-Days</div>
              <div class="text-[11px] text-gray-400 mt-0.5">Credential stuffing, token theft & injection</div>
            </div>
          </button>

          <button onclick="runPresetAction('benchmarks_metrics')" 
            class="flex items-start p-3 bg-[#161b22] hover:bg-[#1c212a] border border-[#30363d] hover:border-cyan-500/70 rounded-xl transition text-left group">
            <div class="p-2 bg-cyan-500/10 text-cyan-400 rounded-lg mr-2.5 group-hover:scale-105 transition">
              <i class="fa-solid fa-chart-column text-base"></i>
            </div>
            <div>
              <div class="text-xs font-semibold text-white group-hover:text-cyan-400">Datasets & Benchmarks</div>
              <div class="text-[11px] text-gray-400 mt-0.5">Alibaba, GAIA & F1-score comparisons</div>
            </div>
          </button>

        </div>

        <!-- CUSTOM TOPIC & KEYWORD INPUT -->
        <div class="pt-2 border-t border-[#21262d] space-y-3">
          
          <div class="flex items-center space-x-2">
            <div class="relative flex-1">
              <i class="fa-solid fa-magnifying-glass absolute left-3.5 top-3 text-gray-500 text-xs"></i>
              <input id="topicInput" type="text" 
                class="w-full bg-[#0b0f17] border border-[#30363d] rounded-xl pl-9 pr-4 py-2.5 text-xs text-white focus:outline-none focus:border-blue-500 placeholder-gray-500 font-medium"
                placeholder="Enter custom research topic or paper title..."
                value="AI-Driven Anomaly Detection for Cloud-Based APIs">
            </div>
            <button id="startBtn" onclick="triggerCustomResearch()" 
              class="px-5 py-2.5 bg-gradient-to-r from-blue-600 to-indigo-600 hover:from-blue-500 hover:to-indigo-500 text-white text-xs font-bold rounded-xl shadow-lg shadow-blue-600/20 transition flex items-center gap-2">
              <i class="fa-solid fa-play"></i> Scout Web
            </button>
          </div>

          <!-- KEYWORDS SELECTOR PILLS -->
          <div class="space-y-1.5">
            <div class="flex items-center justify-between text-[11px]">
              <span class="text-gray-400 font-semibold flex items-center gap-1.5">
                <i class="fa-solid fa-tags text-indigo-400"></i> Active Keywords Filter:
              </span>
              <button onclick="clearKeywords()" class="text-gray-500 hover:text-gray-300 text-[10px]">Clear all</button>
            </div>

            <div class="flex flex-wrap gap-1.5" id="keywordPills">
              <!-- Clickable keyword chips -->
              <span onclick="toggleKeyword(this, 'CloudAnoAgent')" class="keyword-pill px-2.5 py-1 bg-[#161b22] border border-[#30363d] text-gray-300 hover:text-white rounded-lg text-xs cursor-pointer select-none transition">
                + CloudAnoAgent
              </span>
              <span onclick="toggleKeyword(this, 'Graph Neural Networks (GNN)')" class="keyword-pill px-2.5 py-1 bg-[#161b22] border border-[#30363d] text-gray-300 hover:text-white rounded-lg text-xs cursor-pointer select-none transition">
                + Graph Neural Networks
              </span>
              <span onclick="toggleKeyword(this, 'eBPF Telemetry')" class="keyword-pill px-2.5 py-1 bg-[#161b22] border border-[#30363d] text-gray-300 hover:text-white rounded-lg text-xs cursor-pointer select-none transition">
                + eBPF Telemetry
              </span>
              <span onclick="toggleKeyword(this, 'Autoencoders')" class="keyword-pill px-2.5 py-1 bg-[#161b22] border border-[#30363d] text-gray-300 hover:text-white rounded-lg text-xs cursor-pointer select-none transition">
                + Autoencoders
              </span>
              <span onclick="toggleKeyword(this, 'Isolation Forest')" class="keyword-pill px-2.5 py-1 bg-[#161b22] border border-[#30363d] text-gray-300 hover:text-white rounded-lg text-xs cursor-pointer select-none transition">
                + Isolation Forest
              </span>
              <span onclick="toggleKeyword(this, 'Envoy Proxy Sidecar')" class="keyword-pill px-2.5 py-1 bg-[#161b22] border border-[#30363d] text-gray-300 hover:text-white rounded-lg text-xs cursor-pointer select-none transition">
                + Envoy Proxy
              </span>
              <span onclick="toggleKeyword(this, 'Zero-Day API Attacks')" class="keyword-pill px-2.5 py-1 bg-[#161b22] border border-[#30363d] text-gray-300 hover:text-white rounded-lg text-xs cursor-pointer select-none transition">
                + Zero-Day Attacks
              </span>
              <span onclick="toggleKeyword(this, 'Alibaba Microservices Dataset')" class="keyword-pill px-2.5 py-1 bg-[#161b22] border border-[#30363d] text-gray-300 hover:text-white rounded-lg text-xs cursor-pointer select-none transition">
                + Alibaba Dataset
              </span>
            </div>

            <!-- Custom Keyword Add -->
            <div class="flex items-center space-x-2 pt-1">
              <input id="customKeywordInput" type="text" 
                class="w-full sm:w-72 bg-[#0b0f17] border border-[#30363d] rounded-lg px-3 py-1 text-xs text-white placeholder-gray-500 focus:outline-none focus:border-indigo-500"
                placeholder="Type custom keyword and press Enter..." 
                onkeydown="if(event.key==='Enter') addCustomKeyword()">
              <button onclick="addCustomKeyword()" class="px-3 py-1 bg-[#1c212a] hover:bg-[#252c38] text-indigo-300 border border-indigo-900/60 rounded-lg text-xs transition">
                Add Tag
              </button>
            </div>

          </div>

        </div>

      </div>

      <!-- REAL-TIME ACTIVITY STREAM CARD -->
      <div class="bg-[#111620] border border-[#21262d] rounded-2xl p-5 shadow-xl flex-1 flex flex-col min-h-[460px]">
        <div class="flex items-center justify-between pb-3 border-b border-[#21262d] mb-4">
          <div class="flex items-center space-x-2">
            <i class="fa-solid fa-terminal text-emerald-400"></i>
            <h2 class="text-xs font-bold uppercase tracking-wider text-gray-200">Live Crawler Activity Feed</h2>
          </div>
          <span id="eventCountBadge" class="text-xs font-mono bg-[#0b0f17] border border-[#30363d] text-gray-400 px-2.5 py-0.5 rounded-full">0 events</span>
        </div>

        <div id="activityFeed" class="flex-1 overflow-y-auto space-y-3 custom-scroll pr-1 font-sans text-sm">
          <div class="text-center py-20 text-gray-500">
            <i class="fa-solid fa-satellite-dish text-3xl mb-2 opacity-40"></i>
            <p class="text-xs">Select any action button or click 'Scout Web' to begin streaming live crawler events.</p>
          </div>
        </div>
      </div>

    </div>

    <!-- RIGHT PANEL: OUTCOME TABS (REPORT & CITATIONS) (5 cols) -->
    <div class="xl:col-span-5 flex flex-col space-y-6">
      
      <!-- SAVED REPORTS CAROUSEL -->
      <div class="bg-[#111620] border border-[#21262d] rounded-2xl p-4 shadow-xl">
        <div class="flex items-center justify-between mb-2.5">
          <h2 class="text-xs font-bold uppercase tracking-wider text-gray-300 flex items-center gap-2">
            <i class="fa-solid fa-folder-tree text-amber-400"></i> Generated Research Reports
          </h2>
          <button onclick="loadReportsList()" class="text-xs text-blue-400 hover:text-blue-300 flex items-center gap-1 font-medium">
            <i class="fa-solid fa-rotate"></i> Refresh
          </button>
        </div>

        <div id="reportsList" class="space-y-1.5 max-h-40 overflow-y-auto custom-scroll pr-1">
          <div class="text-xs text-gray-500 py-2">Loading reports...</div>
        </div>
      </div>

      <!-- MAIN REPORT & CITATIONS VIEWER -->
      <div class="bg-[#111620] border border-[#21262d] rounded-2xl p-5 shadow-xl flex-1 flex flex-col min-h-[550px]">
        
        <!-- Viewer Tabs Header -->
        <div class="flex items-center justify-between pb-3 border-b border-[#21262d] mb-4">
          <div class="flex items-center space-x-2">
            <button id="tabReportBtn" onclick="switchViewerTab('report')" 
              class="px-3 py-1.5 bg-[#161b22] text-blue-400 border border-blue-500/30 rounded-lg text-xs font-bold transition flex items-center gap-1.5">
              <i class="fa-solid fa-file-lines"></i> Synthesized Report
            </button>
            <button id="tabCitationsBtn" onclick="switchViewerTab('citations')" 
              class="px-3 py-1.5 bg-transparent text-gray-400 hover:text-white rounded-lg text-xs font-semibold transition flex items-center gap-1.5">
              <i class="fa-solid fa-book-bookmark text-amber-400"></i> Citations & Sources (<span id="citationsCount">0</span>)
            </button>
          </div>

          <button id="copyReportBtn" onclick="copyReportText()" 
            class="text-xs bg-[#161b22] hover:bg-[#21262d] border border-[#30363d] text-gray-300 px-3 py-1.5 rounded-lg transition flex items-center gap-1.5">
            <i class="fa-regular fa-copy"></i> Copy Markdown
          </button>
        </div>

        <!-- Title of current report -->
        <div id="activeReportName" class="text-xs font-mono text-gray-400 pb-2 truncate font-semibold">No report loaded</div>

        <!-- TAB 1: Markdown Report View -->
        <div id="reportTabContent" class="flex-1 overflow-y-auto custom-scroll pr-2 text-sm markdown-body">
          <div class="text-center py-28 text-gray-500">
            <i class="fa-solid fa-book-open-reader text-3xl mb-2 opacity-40"></i>
            <p class="text-xs">Run an action above or select a report to inspect the literature review.</p>
          </div>
        </div>

        <!-- TAB 2: Extracted Citations & Source Links -->
        <div id="citationsTabContent" class="hidden flex-1 overflow-y-auto custom-scroll pr-2 space-y-3">
          <div id="citationsList" class="space-y-2.5">
            <div class="text-center py-28 text-gray-500 text-xs">
              <i class="fa-solid fa-link text-3xl mb-2 opacity-40"></i>
              <p>No citations extracted yet. Run a research scout to discover papers.</p>
            </div>
          </div>
        </div>

      </div>

    </div>

  </div>

  <!-- JAVASCRIPT CLIENT LOGIC -->
  <script>
    let currentReportContent = "";
    let selectedKeywords = [];
    let extractedCitations = [];

    // Toggle Keyword Chips
    function toggleKeyword(el, tag) {
      const idx = selectedKeywords.indexOf(tag);
      if (idx > -1) {
        selectedKeywords.splice(idx, 1);
        el.classList.remove('tag-active');
        el.innerText = '+ ' + tag;
      } else {
        selectedKeywords.push(tag);
        el.classList.add('tag-active');
        el.innerText = '✓ ' + tag;
      }
    }

    function addCustomKeyword() {
      const input = document.getElementById('customKeywordInput');
      const val = input.value.trim();
      if (!val) return;
      
      if (!selectedKeywords.includes(val)) {
        selectedKeywords.push(val);
        const pillContainer = document.getElementById('keywordPills');
        const span = document.createElement('span');
        span.className = 'keyword-pill px-2.5 py-1 rounded-lg text-xs cursor-pointer select-none transition tag-active';
        span.innerText = '✓ ' + val;
        span.onclick = function() { toggleKeyword(this, val); };
        pillContainer.appendChild(span);
      }
      input.value = '';
    }

    function clearKeywords() {
      selectedKeywords = [];
      document.querySelectorAll('.keyword-pill').forEach(el => {
        el.classList.remove('tag-active');
        const text = el.innerText.replace('✓ ', '').replace('+ ', '');
        el.innerText = '+ ' + text;
      });
    }

    // Preset Action Button Handlers
    function runPresetAction(action) {
      clearKeywords();
      let topic = "AI-Driven Anomaly Detection for Cloud-Based APIs";

      if (action === 'scout_arxiv') {
        topic = "SOTA preprints on AI-driven anomaly detection for cloud APIs and microservices";
        selectedKeywords = ["arXiv preprints", "Deep Learning", "Cloud APIs"];
      } else if (action === 'gnn_microservices') {
        topic = "Graph Neural Networks and Graph Attention Networks for microservice API call graph anomaly detection";
        selectedKeywords = ["Graph Neural Networks (GNN)", "Microservices Call Graph", "Topology Anomaly"];
      } else if (action === 'llm_sre_agents') {
        topic = "Autonomous LLM agents and neuro-symbolic models for cloud SRE and API anomaly detection";
        selectedKeywords = ["CloudAnoAgent", "LLM SRE Agents", "Neuro-symbolic verification"];
      } else if (action === 'ebpf_telemetry') {
        topic = "eBPF kernel telemetry and deep learning for real-time cloud API attack and anomaly detection";
        selectedKeywords = ["eBPF Telemetry", "Envoy Proxy Sidecar", "Low Latency Gateway"];
      } else if (action === 'api_security_abuse') {
        topic = "AI-driven detection of API credential abuse, token hijacking, and zero-day attacks in cloud environments";
        selectedKeywords = ["Zero-Day API Attacks", "Isolation Forest", "Credential Stuffing"];
      } else if (action === 'benchmarks_metrics') {
        topic = "Evaluation benchmark datasets and F1-score performance in cloud API anomaly detection";
        selectedKeywords = ["Alibaba Microservices Dataset", "GAIA Benchmark", "F1-Score Metrics"];
      }

      document.getElementById('topicInput').value = topic;
      triggerResearch(topic, selectedKeywords);
    }

    function triggerCustomResearch() {
      const topic = document.getElementById('topicInput').value.trim() || "AI-Driven Anomaly Detection for Cloud-Based APIs";
      triggerResearch(topic, selectedKeywords);
    }

    async function triggerResearch(topic, keywords) {
      const btn = document.getElementById('startBtn');
      btn.disabled = true;
      btn.innerHTML = '<i class="fa-solid fa-spinner fa-spin"></i> Scouting...';

      try {
        const resp = await fetch('/api/start', {
          method: 'POST',
          headers: { 'Content-Type': 'application/json' },
          body: JSON.stringify({ topic: topic, keywords: keywords, max_steps: 12 })
        });
        if (!resp.ok) {
          const err = await resp.json();
          alert(err.detail || 'Failed to start research task');
        } else {
          document.getElementById('activityFeed').innerHTML = '';
        }
      } catch (err) {
        alert('Network error: ' + err.message);
      } finally {
        btn.disabled = false;
        btn.innerHTML = '<i class="fa-solid fa-play"></i> Scout Web';
      }
    }

    // Switch between Report and Citations Tabs
    function switchViewerTab(tab) {
      const repBtn = document.getElementById('tabReportBtn');
      const citBtn = document.getElementById('tabCitationsBtn');
      const repContent = document.getElementById('reportTabContent');
      const citContent = document.getElementById('citationsTabContent');

      if (tab === 'report') {
        repBtn.className = "px-3 py-1.5 bg-[#161b22] text-blue-400 border border-blue-500/30 rounded-lg text-xs font-bold transition flex items-center gap-1.5";
        citBtn.className = "px-3 py-1.5 bg-transparent text-gray-400 hover:text-white rounded-lg text-xs font-semibold transition flex items-center gap-1.5";
        repContent.classList.remove('hidden');
        citContent.classList.add('hidden');
      } else {
        citBtn.className = "px-3 py-1.5 bg-[#161b22] text-amber-400 border border-amber-500/30 rounded-lg text-xs font-bold transition flex items-center gap-1.5";
        repBtn.className = "px-3 py-1.5 bg-transparent text-gray-400 hover:text-white rounded-lg text-xs font-semibold transition flex items-center gap-1.5";
        citContent.classList.remove('hidden');
        repContent.classList.add('hidden');
      }
    }

    // Extract Markdown links & citations from report text
    function parseCitations(markdownText) {
      extractedCitations = [];
      // Regex for markdown links [Title](URL)
      const linkRegex = /\[([^\]]+)\]\((https?:\/\/[^\s\)]+)\)/g;
      let match;
      const seenUrls = new Set();

      while ((match = linkRegex.exec(markdownText)) !== null) {
        const title = match[1].trim();
        const url = match[2].trim();
        if (!seenUrls.has(url)) {
          seenUrls.add(url);
          let badge = "Academic Paper";
          let badgeColor = "bg-blue-950 text-blue-400 border-blue-800";
          if (url.includes("arxiv.org")) {
            badge = "arXiv Preprint";
            badgeColor = "bg-red-950 text-red-400 border-red-800";
          } else if (url.includes("ieee.org")) {
            badge = "IEEE Xplore";
            badgeColor = "bg-cyan-950 text-cyan-400 border-cyan-800";
          } else if (url.includes("acm.org")) {
            badge = "ACM Digital Library";
            badgeColor = "bg-emerald-950 text-emerald-400 border-emerald-800";
          } else if (url.includes("springer.com")) {
            badge = "Springer Nature";
            badgeColor = "bg-amber-950 text-amber-400 border-amber-800";
          }

          extractedCitations.push({ title, url, badge, badgeColor });
        }
      }

      document.getElementById('citationsCount').innerText = extractedCitations.length;

      const container = document.getElementById('citationsList');
      if (extractedCitations.length === 0) {
        container.innerHTML = `
          <div class="text-center py-20 text-gray-500 text-xs">
            <p>No direct citation links identified in this document.</p>
          </div>
        `;
        return;
      }

      container.innerHTML = extractedCitations.map((c, i) => `
        <div class="p-3 bg-[#0d1117] border border-[#21262d] hover:border-blue-500/40 rounded-xl transition flex items-start justify-between group">
          <div class="space-y-1 mr-3 flex-1">
            <div class="flex items-center gap-2">
              <span class="text-[10px] px-2 py-0.5 rounded border font-mono ${c.badgeColor}">${c.badge}</span>
              <span class="text-gray-500 text-[10px]">#${i+1}</span>
            </div>
            <h4 class="text-xs font-semibold text-gray-200 group-hover:text-blue-300 transition">${c.title}</h4>
            <p class="text-[11px] text-gray-500 truncate font-mono">${c.url}</p>
          </div>
          <a href="${c.url}" target="_blank" 
            class="px-2.5 py-1.5 bg-[#161b22] hover:bg-blue-600 text-gray-300 hover:text-white rounded-lg text-xs transition flex items-center gap-1 shrink-0 font-medium">
            <span>Open</span> <i class="fa-solid fa-arrow-up-right-from-square text-[10px]"></i>
          </a>
        </div>
      `).join('');
    }

    async function loadReportsList() {
      try {
        const res = await fetch('/api/reports');
        const list = await res.json();
        const container = document.getElementById('reportsList');
        if (!list || list.length === 0) {
          container.innerHTML = '<div class="text-xs text-gray-500 py-2">No reports generated yet.</div>';
          return;
        }

        container.innerHTML = list.map(item => `
          <div onclick="viewReport('${item.filename}')" 
            class="p-2.5 bg-[#0b0f17] hover:bg-[#161b22] border border-[#21262d] hover:border-blue-500/50 rounded-xl cursor-pointer transition flex items-center justify-between group">
            <div class="truncate mr-2">
              <p class="text-xs font-semibold text-blue-300 group-hover:text-blue-400 truncate">${item.filename}</p>
              <p class="text-[10px] text-gray-500 font-mono">${item.created_at} • ${(item.size / 1024).toFixed(1)} KB</p>
            </div>
            <i class="fa-solid fa-chevron-right text-xs text-gray-600 group-hover:text-blue-400 transition"></i>
          </div>
        `).join('');

        if (!currentReportContent && list.length > 0) {
          viewReport(list[0].filename);
        }
      } catch (e) {
        console.error("Failed to load reports:", e);
      }
    }

    async function viewReport(filename) {
      try {
        document.getElementById('activeReportName').innerText = "Viewing: " + filename;
        const res = await fetch(`/api/reports/${encodeURIComponent(filename)}`);
        const data = await res.json();
        currentReportContent = data.content;
        document.getElementById('reportTabContent').innerHTML = marked.parse(data.content);
        parseCitations(data.content);
      } catch (e) {
        document.getElementById('reportTabContent').innerText = "Error loading report: " + e.message;
      }
    }

    function copyReportText() {
      if (!currentReportContent) return;
      navigator.clipboard.writeText(currentReportContent);
      const btn = document.getElementById('copyReportBtn');
      btn.innerHTML = '<i class="fa-solid fa-check text-emerald-400"></i> Copied!';
      setTimeout(() => {
        btn.innerHTML = '<i class="fa-regular fa-copy"></i> Copy Markdown';
      }, 2000);
    }

    // Connect to Server-Sent Events (SSE)
    function setupSSE() {
      const evtSource = new EventSource('/api/stream');
      let eventCount = 0;

      evtSource.onmessage = function(e) {
        const event = JSON.parse(e.data);
        eventCount++;
        document.getElementById('eventCountBadge').innerText = eventCount + ' events';
        renderFeedEvent(event);
      };

      evtSource.onerror = function() {
        console.warn("SSE connection interrupted. Reconnecting in 3s...");
      };
    }

    function renderFeedEvent(ev) {
      const feed = document.getElementById('activityFeed');
      const type = ev.type;
      const data = ev.data;
      const time = ev.timestamp;

      const statusDot = document.getElementById('statusDot');
      const statusText = document.getElementById('statusText');
      const pulseRing = document.getElementById('pulseRing');

      if (type === "agent_started" || type === "step_started" || type === "search_start" || type === "crawl_start") {
        statusDot.className = "relative inline-flex rounded-full h-2.5 w-2.5 bg-emerald-400";
        statusText.innerText = "SCOUTING";
        statusText.className = "text-emerald-400 font-bold tracking-wide";
        pulseRing.classList.remove('hidden');
      } else if (type === "agent_completed" || type === "agent_idle") {
        statusDot.className = "relative inline-flex rounded-full h-2.5 w-2.5 bg-gray-500";
        statusText.innerText = "IDLE";
        statusText.className = "text-gray-300 font-semibold tracking-wide";
        pulseRing.classList.add('hidden');
        loadReportsList();
      }

      let cardHtml = "";

      if (type === "agent_started") {
        const kwHtml = data.keywords && data.keywords.length > 0 
          ? `<div class="flex flex-wrap gap-1 mt-1.5">${data.keywords.map(k => `<span class="bg-blue-900/40 text-blue-300 text-[10px] px-2 py-0.5 rounded border border-blue-700/50 font-mono">#${k}</span>`).join('')}</div>`
          : '';

        cardHtml = `
          <div class="bg-blue-950/20 border border-blue-800/40 rounded-xl p-3.5 shadow-sm">
            <div class="flex items-center justify-between text-xs text-blue-400 font-bold mb-1">
              <span><i class="fa-solid fa-play mr-1"></i> Agent Started Research Scout</span>
              <span class="text-gray-500 font-mono">${time}</span>
            </div>
            <p class="text-xs text-gray-200 font-semibold">${data.topic}</p>
            ${kwHtml}
          </div>
        `;
      } else if (type === "step_started") {
        cardHtml = `
          <div class="flex items-center space-x-2 text-[11px] text-gray-500 py-1 font-mono">
            <div class="h-px bg-[#21262d] flex-1"></div>
            <span>STEP ${data.step} / ${data.max_steps}</span>
            <div class="h-px bg-[#21262d] flex-1"></div>
          </div>
        `;
      } else if (type === "search_start") {
        cardHtml = `
          <div class="bg-[#0b0f17] border border-cyan-800/40 rounded-xl p-3 shadow-sm">
            <div class="flex items-center justify-between text-xs text-cyan-400 font-semibold mb-1">
              <span><i class="fa-solid fa-magnifying-glass mr-1"></i> Scouting Web & Academic Sources (${data.filter})</span>
              <span class="text-gray-500 font-mono">${time}</span>
            </div>
            <p class="text-xs text-cyan-200 font-mono">Query: "${data.query}"</p>
          </div>
        `;
      } else if (type === "search_results") {
        const items = data.results.slice(0, 4).map(r => `
          <li class="truncate flex items-center justify-between group">
            <a href="${r.url}" target="_blank" class="text-blue-400 hover:text-blue-300 truncate text-xs">• ${r.title}</a>
            <i class="fa-solid fa-arrow-up-right-from-square text-[9px] text-gray-500 ml-1.5 opacity-0 group-hover:opacity-100 transition"></i>
          </li>
        `).join('');
        cardHtml = `
          <div class="bg-[#0b0f17] border border-[#21262d] rounded-xl p-3 text-xs shadow-sm">
            <div class="flex items-center justify-between text-emerald-400 font-semibold mb-2">
              <span><i class="fa-solid fa-list-check mr-1"></i> Discovered ${data.count} Research Papers</span>
              <span class="text-gray-500 font-mono">${time}</span>
            </div>
            <ul class="space-y-1 text-gray-300">
              ${items}
            </ul>
          </div>
        `;
      } else if (type === "crawl_start") {
        cardHtml = `
          <div class="bg-[#0b0f17] border border-amber-800/40 rounded-xl p-3 shadow-sm">
            <div class="flex items-center justify-between text-xs text-amber-400 font-semibold mb-1">
              <span><i class="fa-solid fa-spider mr-1"></i> Crawling Paper / URL</span>
              <span class="text-gray-500 font-mono">${time}</span>
            </div>
            <p class="text-xs text-gray-300 truncate font-mono">${data.url}</p>
          </div>
        `;
      } else if (type === "crawl_done") {
        cardHtml = `
          <div class="bg-emerald-950/20 border border-emerald-800/40 rounded-xl p-3 text-xs">
            <div class="flex items-center justify-between text-emerald-400 font-semibold mb-1">
              <span><i class="fa-solid fa-check mr-1"></i> Extracted ${data.length} characters of clean text</span>
              <span class="text-gray-500 font-mono">${time}</span>
            </div>
            <p class="text-gray-400 italic text-[11px] line-clamp-2">${data.preview}</p>
          </div>
        `;
      } else if (type === "tool_call") {
        cardHtml = `
          <div class="bg-[#0b0f17] border border-purple-800/40 rounded-xl p-2.5 text-xs">
            <div class="flex items-center justify-between text-purple-400 font-semibold">
              <span><i class="fa-solid fa-bolt mr-1"></i> Hermes Tool Execution: <span class="text-white font-mono">${data.tool}</span></span>
              <span class="text-gray-500 font-mono">${time}</span>
            </div>
          </div>
        `;
      } else if (type === "report_saved") {
        cardHtml = `
          <div class="bg-yellow-950/20 border border-yellow-700/50 rounded-xl p-3 text-xs">
            <div class="flex items-center justify-between text-yellow-400 font-bold mb-1">
              <span><i class="fa-solid fa-floppy-disk mr-1"></i> Report Generated & Persisted</span>
              <span class="text-gray-500 font-mono">${time}</span>
            </div>
            <p class="text-gray-300 font-mono text-[11px]">${data.filename}</p>
          </div>
        `;
      } else if (type === "agent_completed") {
        cardHtml = `
          <div class="bg-emerald-950/30 border border-emerald-500/50 rounded-xl p-4 text-xs shadow-lg">
            <div class="flex items-center justify-between text-emerald-400 font-bold mb-2">
              <span class="text-sm"><i class="fa-solid fa-circle-check mr-1"></i> Research & Literature Review Completed!</span>
              <span class="text-gray-400 font-mono">${time}</span>
            </div>
            <div class="text-gray-200 mt-2">${marked.parse(data.summary)}</div>
          </div>
        `;
        if (data.report_file) {
          viewReport(data.report_file);
        }
      }

      if (cardHtml) {
        feed.insertAdjacentHTML('beforeend', cardHtml);
        feed.scrollTop = feed.scrollHeight;
      }
    }

    window.addEventListener('DOMContentLoaded', () => {
      loadReportsList();
      setupSSE();
    });
  </script>
</body>
</html>
"""

@app.get("/", response_class=HTMLResponse)
async def serve_dashboard():
    return HTMLResponse(content=DASHBOARD_HTML)

if __name__ == "__main__":
    import uvicorn
    print("=" * 70)
    print("🚀 Hermes 3 Cloud API Research Dashboard")
    print("📍 Dashboard URL: http://localhost:8000")
    print("=" * 70)
    uvicorn.run(app, host="127.0.0.1", port=8000, log_level="info")
