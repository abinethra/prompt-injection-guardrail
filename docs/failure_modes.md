# Red-Team Failure Modes & Next-Stage Mitigations

## 1. Leetspeak & Typo Obfuscation (`ign0re`, `syst3m pr0mpt`, `ignroe`)
- **Why it happens:** Word-level regexes and un-normalized tokens miss character substitutions (`0` -> `o`, `1` -> `i`, `3` -> `e`, `4` -> `a`) or transposed letters, and small training sets may under-represent noisy spellings.
- **How to fix next:** Add a canonical text normalization pre-pass (leetspeak folding + Unicode NFKC normalization + repeated/transposed character handling) before heuristic and classifier scoring, while keeping the raw prompt for audit hashing.

## 2. Multilingual & Transliterated Attacks (Tamil, Hindi, French, Code-Mixed)
- **Why it happens:** `distilbert-base-uncased` and English-heavy heuristic patterns have weak coverage on non-English scripts and Romanized code-mixed phrasing (e.g., Tanglish/Hinglish/French commands).
- **How to fix next:** Add multilingual/transliterated attack examples to the training set, use a multilingual encoder or language-aware intent classifier, and pair input checks with strict system-prompt instruction hierarchy.

## 3. Roleplay, Persona & Authority Framing (`DAN`, fictional villain, maintenance/audit mode)
- **Why it happens:** Attackers wrap malicious intent inside benign-looking framing ("for a movie", "security audit", "maintenance mode") so surface phrases look like storytelling or QA.
- **How to fix next:** Detect framing + imperative action pairs (persona shift combined with rule-override or secret-disclosure verbs) and enforce a **tool-call & action allowlist** so roleplay text cannot trigger privileged actions or bypass policy.

## 4. Base64 & Encoded Payloads
- **Why it happens:** Attackers encode payloads in Base64, hex, or URL encoding so English keyword/ML detectors see opaque strings rather than `"ignore previous instructions"`.
- **How to fix next:** Recursively decode Base64/URL/hex segments up to a safe depth, scan decoded strings with the detector, and block or flag encoded payloads directed at the model unless explicitly expected by the application.

## 5. Indirect Prompt Injection Inside "Documents to Summarize"
- **Why it happens:** Untrusted external text (articles, tickets, notes, retrieved RAG chunks) is concatenated into a prompt alongside instructions, blurring data vs. control boundaries.
- **How to fix next:** Apply **retrieval/document sanitization** (strip/flag instruction-like lines inside untrusted data blocks), isolate untrusted documents in structured delimiters/roles, add **tool-call validation** (require explicit user intent + parameter schema checks before side effects), and run an **output guard** to catch exfiltration attempts (URLs/emails/secrets) in responses.

## 6. Benign Security/Policy Mentions (`benign_tricky`)
- **Why it happens:** Legitimate questions about `"ignore"`, `"system prompt"`, or `"password policy"` share vocabulary with attacks, causing false positives if lexical signals are weighted too heavily.
- **How to fix next:** Tune ensemble weights and thresholds on validation/red-team sets, reward benign technical context (e.g., `pandas`, `regex`, `git`, `password policy` Q&A), and route borderline scores to `flagged` with output guarding rather than hard-blocking harmless queries.