# Prompt Injection & Jailbreak Guardrail Proxy

A FastAPI screening proxy that evaluates incoming prompts—and screens downstream LLM outputs—before responses reach users. It combines deterministic security heuristics, a TF-IDF + Logistic Regression baseline, and a fine-tuned `distilbert-base-uncased` classifier to score prompt-injection and jailbreak risk with structured JSONL decision logging.

## Architecture

```mermaid
flowchart LR
    Client["Client / Caller"] -->|"POST /chat"| API["FastAPI Guardrail Proxy (app/main.py)"]
    API -->|"score_prompt()"| Det["GuardrailDetector (app/detector.py)"]

    subgraph Ensemble["Risk Scoring Ensemble"]
        Det --> H["Heuristics (app/heuristics.py)"]
        Det --> TFIDF["TF-IDF + LogReg (models/tfidf_logreg_baseline.joblib)"]
        Det --> BERT["DistilBERT Classifier (models/transformer/)"]
    end

    API -->|"decision: allowed / flagged / blocked"| Policy{"Score vs Thresholds"}
    Policy -->|"score >= block_threshold"| Block["Blocked Reply (No Upstream Call)"]
    Policy -->|"score < block_threshold"| LLM["Mock LLM or Optional Gemini API"]
    LLM --> OutGuard["Output Secret & Leak Sanitizer"]
    OutGuard -->|"sanitized reply"| Client
    API -->|"SHA-256 prompt_hash, score, signals, latency_ms"| Logs["logs/decisions.jsonl"]
Threat Model
What It Helps Stop / Detect
Direct instruction overrides: Phrases attempting to ignore, bypass, or replace system/developer instructions (including leetspeak and common typos).

System-prompt extraction & DAN/roleplay framing: Prompts trying to dump hidden instructions or shift into unrestricted/jailbreak personas.

Encoded & indirect injection cues: Base64/long-encoded payloads with injection keywords and delimiter/instruction patterns inside text to summarize.

Basic output leaks: Response scanning and redaction for secret-like tokens, email addresses, and "my system prompt is" leaks before returning a reply.

What It Does Not Stop
Advanced multi-turn or adaptive jailbreaks: Sophisticated persona/crescendo attacks or novel phrasing outside training and heuristic rules can still bypass detection.

Deep non-English / code-mixed semantics: While basic transliteration patterns were added during red-teaming, broader multilingual coverage is still limited.

Full RAG / tool-call privilege enforcement: Prompt screening alone does not replace per-tool authorization checks, least-privilege IAM/API scopes, or full document sanitization in retrieval pipelines.

Cryptographic/privacy guarantees of hashes: logs/decisions.jsonl stores a SHA-256 hash rather than raw prompt text, which avoids plain-text prompt storage but is not a substitute for full PII governance or high-entropy key derivation.

Setup & Run Instructions
Bash
cd ~/projects/prompt-injection-guardrail
python -m venv venv
source venv/bin/activate
pip install --upgrade pip
pip install -r requirements.txt

# 1. Prepare datasets and splits
python scripts/prepare_data.py

# 2. Train baseline and transformer models
python scripts/train_baseline.py
python scripts/train_transformer.py

# 3. Run offline evaluation and red-team suite
PYTHONPATH=. python scripts/evaluate.py
PYTHONPATH=. python scripts/redteam.py --label "AFTER" --out metrics/redteam_after.json
cp metrics/redteam_after.json metrics/redteam.json

# 4. Configure environment and start the proxy
cp .env.example .env
uvicorn app.main:app --host 127.0.0.1 --port 8000 --reload
API Examples
Health Check (GET /health)
Bash
curl -s [http://127.0.0.1:8000/health](http://127.0.0.1:8000/health) | python -m json.tool
Stats (GET /stats)
Bash
curl -s [http://127.0.0.1:8000/stats](http://127.0.0.1:8000/stats) | python -m json.tool
Benign Prompt (POST /chat)
Bash
curl -s -X POST [http://127.0.0.1:8000/chat](http://127.0.0.1:8000/chat) \
  -H "Content-Type: application/json" \
  -d '{"prompt": "How do I reverse a list in Python?"}' | python -m json.tool
Attack Prompt (POST /chat)
Bash
