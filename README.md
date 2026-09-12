# 🧠 AI Cloud API Anomaly Detection Research Crawler

A **100% free, private, and local** academic research crawler and literature surveyor specialized in **AI-Driven Anomaly Detection for Cloud-Based APIs and Microservices**, powered by **Nous Research's Hermes 3** on your local **NVIDIA RTX 5070 Ti (16 GB VRAM)** via **Ollama**.

---

## 🎯 Target Research Domain
- **Subject**: AI-Driven Anomaly Detection for Cloud-Based APIs & Microservices
- **Focus Areas**:
  - Unsupervised & Deep Learning models (Autoencoders, Isolation Forests, One-Class SVM, LSTMs, Transformers).
  - Graph Neural Networks (GNNs / GATs) modeling microservice API call topologies & distributed traces.
  - Large Language Model (LLM) agents & Neuro-symbolic detectors (e.g., *CloudAnoAgent*, *LogGPT*).
  - High-throughput low-latency telemetry (eBPF, Envoy proxy sidecars, Kubernetes ingress).
  - Detection taxonomy: latency degradation, schema deviations, credential stuffing, DDoS, and cascading failures.

---

## 🛠️ Specialized Academic Tools

1. **`search_research_papers`**:
   - Searches across arXiv preprints, IEEE Xplore, ACM Digital Library, Springer Nature, USENIX, and Semantic Scholar.
   - Allows source filtering (`arxiv`, `academic`, `cloud_engineering`, or `all`).
2. **`crawl_paper_or_webpage`**:
   - Fetches arXiv papers, HTML paper editions (`arxiv.org/html/...`), IEEE/ACM abstracts, and technical whitepapers, extracting clean text, formulas, and findings.
3. **`save_research_report`**:
   - Compiles and writes structured literature reviews with citations into `research_reports/`.

---

## 🌐 Web-Based Dashboard

A real-time web dashboard is available to track crawler progress, live activity, and read generated reports.

### Start the Web Dashboard:
```powershell
cd C:\Code_Rex\research_crawler
python server.py
```
Open your browser at:
👉 **[http://localhost:8000](http://localhost:8000)**

### Dashboard Features:
- **Live Activity Stream**: Real-time event feed displaying paper queries, arXiv URLs, crawler extracts, and Hermes 3 tool calls.
- **One-Click Topic Launcher**: Start research on Cloud API Anomaly Detection or custom subtopics directly from the browser.
- **Outcome & Reports Viewer**: Browse, preview, render, and copy generated Markdown literature reviews side-by-side.
- **System Telemetry**: Displays GPU utilization status, local model, and Ollama connection state.

---

## 🚀 CLI Execution (Alternative)

If you prefer running via terminal:
```powershell
python research_agent.py
```
*(Press Enter when prompted to immediately start researching "AI-Driven Anomaly Detection for Cloud-Based APIs")*

### Run on Specific Sub-Topics or Architectural Questions:
```powershell
# Graph Neural Networks for API call graphs
python research_agent.py "Graph Neural Networks for microservice API anomaly detection"

# LLM Agents in Cloud Site Reliability & Anomaly Detection
python research_agent.py "LLM agents for cloud site reliability engineering and API anomaly detection"

# eBPF and Real-Time API Threat Detection
python research_agent.py "eBPF kernel telemetry deep learning API anomaly detection"
```

---

## 📂 Output Reports
All synthesized literature reviews, paper links, and comparative analyses are automatically stored in:
`C:\Code_Rex\research_crawler\research_reports/`
