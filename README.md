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