import os
import sys
import json
import httpx
import trafilatura
from datetime import datetime
from ddgs import DDGS
from rich.console import Console
from rich.panel import Panel
from rich.markdown import Markdown
from rich.table import Table
import ollama

# Configure Windows terminal to support UTF-8 cleanly
if sys.platform == "win32":
    try:
        sys.stdout.reconfigure(encoding="utf-8")
        sys.stderr.reconfigure(encoding="utf-8")
    except Exception:
        pass

console = Console(force_terminal=True, legacy_windows=False)

MODEL_NAME = "hermes3"
OLLAMA_HOST = "http://127.0.0.1:11434"
client = ollama.Client(host=OLLAMA_HOST)

DEFAULT_TOPIC = "AI-Driven Anomaly Detection for Cloud-Based APIs"
REPORTS_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "research_reports")
os.makedirs(REPORTS_DIR, exist_ok=True)

# -------------------------------------------------------------
# 1. SPECIALIZED RESEARCH TOOLS FOR PAPERS & TOPICS
# -------------------------------------------------------------

def search_research_papers(query: str, source: str = "all", max_results: int = 5) -> str:
    """
    Search for academic research papers, preprints, and technical whitepapers.
    Sources:
      - 'arxiv': Preprints on arxiv.org
      - 'academic': arXiv, IEEE, ACM, Springer, Semantic Scholar, OpenReview
      - 'cloud_engineering': USENIX, AWS, Google Cloud, Cloudflare, Datadog engineering
      - 'all': Broad academic and technical conference search
    """
    source_filters = {
        "arxiv": "site:arxiv.org",
        "academic": "(site:arxiv.org OR site:openreview.net OR site:semanticscholar.org OR site:ieee.org OR site:acm.org OR site:springer.com)",
        "cloud_engineering": "(site:usenix.org OR site:aws.amazon.com OR site:cloud.google.com OR site:cloudflare.com OR site:datadoghq.com)",
        "all": "(site:arxiv.org OR site:openreview.net OR site:semanticscholar.org OR site:usenix.org OR site:ieee.org OR filetype:pdf)"
    }

    site_query = source_filters.get(source.lower(), source_filters["all"])
    full_query = f"{site_query} {query}".strip()

    console.print(f"[bold cyan]🔍 [Paper Search][/bold cyan] Filter: [magenta]{source}[/magenta] | Query: [italic]{query}[/italic]")
    try:
        results = []
        with DDGS() as ddgs:
            for item in ddgs.text(full_query, max_results=max_results):
                results.append({
                    "title": item.get("title", ""),
                    "url": item.get("href", ""),
                    "snippet": item.get("body", "")
                })
        
        # Fallback if strict filter yields no results
        if not results:
            console.print("[dim yellow]Strict filter returned 0 results; running broader search...[/dim yellow]")
            with DDGS() as ddgs:
                fallback_query = f"{query} research paper anomaly detection cloud API"
                for item in ddgs.text(fallback_query, max_results=max_results):
                    results.append({
                        "title": item.get("title", ""),
                        "url": item.get("href", ""),
                        "snippet": item.get("body", "")
                    })

        if not results:
            return json.dumps({"status": "no_results", "message": f"No academic papers found for query: '{query}'."})

        table = Table(title="Research Papers & Publications Discovered", show_header=True, header_style="bold magenta")
        table.add_column("Paper / Article Title", style="cyan", no_wrap=False, max_width=45)
        table.add_column("URL / Repository", style="green", no_wrap=False, max_width=50)
        for r in results:
            table.add_row(r["title"], r["url"])
        console.print(table)

        return json.dumps(results, indent=2)
    except Exception as e:
        console.print(f"[red]Paper search error:[/red] {e}")
        return json.dumps({"error": f"Failed to search papers: {str(e)}"})


def crawl_paper_or_webpage(url: str, max_chars: int = 8000) -> str:
    """
    Crawl an academic paper page, arXiv abstract/HTML, conference proceeding,
    or cloud engineering whitepaper, extracting clean text, methodology, and findings.
    """
    console.print(f"[bold green]🕷️ [Crawl][/bold green] Fetching paper content: [underline]{url}[/underline]")
    try:
        downloaded = trafilatura.fetch_url(url)
        if not downloaded:
            headers = {
                "User-Agent": (
                    "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
                    "AppleWebKit/537.36 (KHTML, like Gecko) "
                    "Chrome/124.0.0.0 Safari/537.36"
                ),
                "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8"
            }
            with httpx.Client(timeout=25.0, follow_redirects=True, headers=headers) as http_client:
                resp = http_client.get(url)
                if resp.status_code == 200:
                    downloaded = resp.text
                else:
                    return json.dumps({"error": f"HTTP error {resp.status_code} when requesting {url}"})

        # Trafilatura extracts clean paper abstract, introduction, methodology, and evaluation
        extracted_text = trafilatura.extract(
            downloaded,
            url=url,
            include_links=True,
            include_images=False,
            include_tables=True,
            output_format="txt"
        )

        if not extracted_text or len(extracted_text.strip()) < 50:
            return json.dumps({
                "status": "warning",
                "message": "Content was minimal or protected by paywall/JavaScript."
            })

        cleaned_text = extracted_text[:max_chars]
        console.print(f"[dim green]✓ Extracted {len(cleaned_text)} characters of academic content & analysis.[/dim green]")

        return json.dumps({
            "url": url,
            "content_length": len(cleaned_text),
            "content": cleaned_text
        })
    except Exception as e:
        console.print(f"[red]Paper crawl error:[/red] {e}")
        return json.dumps({"error": f"Failed to crawl paper from {url}: {str(e)}"})


def save_research_report(topic: str, content: str) -> str:
    """
    Save the completed academic literature review / research report into a Markdown file.
    """
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

    console.print(Panel(
        f"[bold gold1]💾 Academic Report Successfully Saved![/bold gold1]\n"
        f"[green]{file_path}[/green]",
        title="Literature Review Saved",
        border_style="gold1"
    ))
    return f"Report successfully saved to {file_path}"


# -------------------------------------------------------------
# 2. HERMES 3 TOOL SCHEMAS & REGISTRATION
# -------------------------------------------------------------

TOOLS = [
    {
        "type": "function",
        "function": {
            "name": "search_research_papers",
            "description": "Searches for academic research papers, preprints (arXiv), and conference papers on AI-driven cloud API anomaly detection.",
            "parameters": {
                "type": "object",
                "properties": {
                    "query": {
                        "type": "string",
                        "description": "Specific academic search query (e.g. 'Cloud API anomaly detection deep learning', 'Graph neural networks microservices API call graph', 'unsupervised log anomaly detection')."
                    },
                    "source": {
                        "type": "string",
                        "enum": ["all", "arxiv", "academic", "cloud_engineering"],
                        "description": "Repository filter: 'arxiv' for preprints, 'academic' for peer-reviewed papers, 'cloud_engineering' for industry whitepapers, 'all' for broad search."
                    },
                    "max_results": {
                        "type": "integer",
                        "description": "Number of paper results to retrieve (default 5)."
                    }
                },
                "required": ["query"]
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "crawl_paper_or_webpage",
            "description": "Crawls and extracts full paper abstracts, methodology, formulas, algorithms, and benchmark results from an academic paper URL or arXiv link.",
            "parameters": {
                "type": "object",
                "properties": {
                    "url": {
                        "type": "string",
                        "description": "The full HTTP/HTTPS URL of the academic paper or arXiv page."
                    }
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
                    "topic": {
                        "type": "string",
                        "description": "Research topic or paper title."
                    },
                    "content": {
                        "type": "string",
                        "description": "The complete, detailed markdown literature review covering taxonomy, AI/ML models, datasets, metrics, and citations."
                    }
                },
                "required": ["topic", "content"]
            }
        }
    }
]

TOOL_REGISTRY = {
    "search_research_papers": search_research_papers,
    "crawl_paper_or_webpage": crawl_paper_or_webpage,
    "save_research_report": save_research_report
}

# -------------------------------------------------------------
# 3. SPECIALIZED SYSTEM PROMPT FOR CLOUD API ANOMALY DETECTION
# -------------------------------------------------------------

SYSTEM_PROMPT = """You are an elite Computer Science & Cloud Security Researcher specialized in:
"AI-Driven Anomaly Detection for Cloud-Based APIs and Microservices".

Your mission:
Investigate current State-of-the-Art (SOTA) research papers, preprints, and cloud engineering architectures for detecting anomalies in cloud APIs.

RESEARCH PROTOCOL:
1. Formulate precise academic search queries using `search_research_papers`. Focus on:
   - Deep learning for API traffic anomaly detection (Autoencoders, LSTMs, Transformers, TCNs).
   - Graph Neural Networks (GNNs/GATs) modeling distributed microservice API call graphs.
   - Foundation models & LLM-based autonomous log/API monitoring (e.g. neuro-symbolic agents, LogGPT, CloudAnoAgent).
   - Unsupervised methods (Isolation Forest, One-Class SVM, Mahalanobis distance) for high-throughput zero-day API abuse.
   - Low-latency edge/gateway integration (eBPF telemetry, Envoy sidecars, Kubernetes ingress controllers).

2. MANDATORY STEP: Call `crawl_paper_or_webpage` on at least 2 relevant paper URLs or arXiv pages discovered from your search. Read their actual abstracts, architectures, datasets, and benchmark results.

3. Synthesize your research into an exhaustive, publication-grade Literature Review with clear Markdown headings:
   - # Title & Executive Overview
   - ## 1. Problem Formulation: API Anomalies in Cloud Environments (Latency degradation, payload tampering, schema violations, DDoS/credential stuffing, cascading microservice failures)
   - ## 2. Taxonomy of AI/ML Methodologies (Unsupervised statistical ML, Deep Autoencoders, Sequence Transformers, Graph Neural Networks, LLM Agents)
   - ## 3. State-of-the-Art Papers & Architectures (Detailed analysis of crawled papers, their proposed neural network designs, and core innovations)
   - ## 4. Benchmark Datasets & Evaluation Metrics (Precision/Recall, F1-score, False Positive Rate, latency overhead in ms, datasets like GAIA, Alibaba microservices, TrainTicket, BGL)
   - ## 5. Practical Cloud Deployment Architecture (API Gateways, Envoy proxy, eBPF in Kubernetes, real-time inference vs offline training)
   - ## 6. Open Challenges & Future Directions (Concept drift, encrypted payload inspection, cold-start APIs)
   - ## 7. Bibliography & Paper Links (Exact paper titles, authors, venues/arXiv IDs, and URLs)

4. Call `save_research_report` to save the comprehensive review.
5. Provide a high-level summary to the user outlining the most critical breakthroughs discovered.
"""

# -------------------------------------------------------------
# 4. AGENT EXECUTION LOOP
# -------------------------------------------------------------

def run_agent(topic: str, max_iterations: int = 12):
    console.print(Panel.fit(
        f"[bold cyan]AI Cloud API Anomaly Detection Researcher[/bold cyan]\n"
        f"Subject: [bold white]{topic}[/bold white]\n"
        f"Engine: [green]Ollama ({MODEL_NAME})[/green] | [yellow]RTX 5070 Ti (16GB VRAM)[/yellow]",
        border_style="cyan"
    ))

    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": f"Please conduct in-depth academic research and literature survey on: '{topic}'"}
    ]

    report_saved = False
    iteration = 0
    while iteration < max_iterations:
        iteration += 1
        console.print(f"\n[dim bold]--- Step {iteration}/{max_iterations} ---[/dim bold]")

        try:
            response = client.chat(
                model=MODEL_NAME,
                messages=messages,
                tools=TOOLS
            )
        except ollama.ResponseError as e:
            if "model" in str(e).lower() and "not found" in str(e).lower():
                console.print(f"[bold red]Error:[/bold red] Model '{MODEL_NAME}' is not found in Ollama.")
                console.print(f"[yellow]Please run: ollama pull {MODEL_NAME}[/yellow]")
                return
            console.print(f"[red]Ollama error:[/red] {e}")
            return
        except Exception as e:
            console.print(f"[red]Error communicating with Ollama:[/red] {e}")
            return

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

                console.print(f"[bold blue]⚡ Tool Call:[/bold blue] [white]{fn_name}[/white]")

                if fn_name in TOOL_REGISTRY:
                    try:
                        if fn_name == "save_research_report":
                            report_saved = True
                        result = TOOL_REGISTRY[fn_name](**args)
                    except Exception as err:
                        result = f"Error executing tool {fn_name}: {err}"
                else:
                    result = f"Error: Tool '{fn_name}' not recognized."

                messages.append({
                    "role": "tool",
                    "content": str(result)
                })
        else:
            content = msg.get("content") if isinstance(msg, dict) else getattr(msg, "content", "")
            if not content or not content.strip():
                console.print("[dim yellow]Agent is planning. Prompting to begin research...[/dim yellow]")
                messages.append({
                    "role": "user",
                    "content": "Please proceed with your research by calling `search_research_papers` to find authoritative papers."
                })
                continue

            if not report_saved and len(content.strip()) > 100:
                save_research_report(topic, content)

            console.print("\n[bold green]✔ Academic Research Review Complete![/bold green]\n")
            console.print(Markdown(content))
            break


def main():
    console.print(Panel(
        "[bold cyan]AI-Driven Cloud API Anomaly Detection Research Crawler[/bold cyan]\n"
        "Specialized Academic Literature & Preprint Search Engine\n"
        "100% Free • Powered by Hermes 3 on RTX 5070 Ti",
        border_style="bright_blue"
    ))
    
    if len(sys.argv) > 1:
        topic = " ".join(sys.argv[1:])
    else:
        console.print(f"[dim]Default topic:[/dim] [bold white]{DEFAULT_TOPIC}[/bold white]")
        user_input = console.input("\n[bold yellow]Enter research topic (Press Enter for default):[/bold yellow] ").strip()
        topic = user_input if user_input else DEFAULT_TOPIC

    run_agent(topic)


if __name__ == "__main__":
    main()
