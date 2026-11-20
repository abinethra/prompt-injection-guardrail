from __future__ import annotations

from pathlib import Path
from typing import List, Tuple

import pandas as pd
from datasets import load_dataset
from sklearn.model_selection import train_test_split

RANDOM_SEED = 42
DATA_DIR = Path(__file__).resolve().parent.parent / "data"


def generate_benign_prompts() -> List[Tuple[str, int, str]]:
    """Generate at least 200 benign prompts across categories, including tricky-but-harmless ones."""
    samples: List[Tuple[str, int, str]] = []

    coding_prompts = [
        "Write a Python function to reverse a singly linked list in place.",
        "How do I index a pandas DataFrame by multiple columns and filter rows?",
        "Explain the time complexity difference between merge sort and quicksort.",
        "Write a bash one-liner to find the top 10 largest files in a directory.",
        "How can I prevent SQL injection when using parameterized queries in Python?",
        "Give an example of a FastAPI endpoint with Pydantic request validation.",
        "Why does a Python dictionary lookup run in O(1) average time?",
        "How do I rebase a feature branch onto main safely in Git?",
        "Write a SQL query to calculate a 7-day rolling average of daily signups.",
        "What is the difference between a thread, a process, and an asyncio coroutine?",
        "How do I parse a JSON file and handle missing keys gracefully in Python?",
        "Explain how TLS handshake establishes a shared session key.",
        "Write a regex pattern to validate an IPv4 address in Python.",
        "How do I inspect listening ports and processes on Linux?",
        "What causes a race condition and how do mutexes prevent it?",
        "Write a unit test using pytest that checks for a ValueError.",
        "How do I optimize a slow Dockerfile by ordering layers properly?",
        "Explain the difference between symmetric and asymmetric encryption.",
        "Write a Python decorator that measures function execution time.",
        "How do I convert a Unix timestamp to ISO-8601 format in Python?",
        "What is CSRF and how do SameSite cookies help mitigate it?",
        "Write a function that merges two sorted lists without built-in sort.",
        "How do I check memory usage of a Python object using sys.getsizeof?",
        "Explain how HTTP status codes 401, 403, 404, and 429 differ.",
        "Write a C++ snippet using std::unordered_map to count word frequencies.",
        "How do I set up pre-commit hooks for ruff or flake8?",
        "What is a deadlock and what are the four Coffman conditions?",
        "Write a Python generator that yields Fibonacci numbers lazily.",
        "How do I configure CORS middleware in a FastAPI application?",
        "Explain how JWT signature verification works on the server side.",
        "Write a script to batch-rename .jpeg files to .jpg in a folder.",
        "How do I profile a slow Python function with cProfile?",
        "What is the difference between authentication and authorization?",
        "Write a binary search function and explain its boundary conditions.",
        "How do I read environment variables safely with fallback defaults in Python?",
        "Explain how content security policy (CSP) headers reduce XSS risk.",
        "Write a function to flatten a nested list of arbitrary depth in Python.",
        "How do I check systemd service logs using journalctl?",
        "What is the purpose of a salt in password hashing algorithms like Argon2?",
        "Write a SQL join to list customers who have placed zero orders.",
        "How do I mock an external HTTP call in pytest?",
        "Explain how TCP congestion control handles packet loss.",
        "Write a Python script to validate CSV headers before ingestion.",
        "How do I export a scikit-learn pipeline using joblib?",
        "What is path traversal and how do you sanitize file paths?",
        "Write a recursive function to compute factorial with memoization.",
        "How do I inspect environment variables inside a running container?",
        "Explain the difference between hashing, encoding, and encryption.",
        "Write a Python function to compute precision, recall, and F1 from counts.",
        "How do I configure Uvicorn workers for production traffic?",
    ]

    support_prompts = [
        "Hi, I ordered a keyboard yesterday and haven't received a tracking number yet.",
        "Can you help me update the billing address on my monthly subscription?",
        "My invoice for October shows a duplicate charge—how can I get a refund?",
        "What are your customer support hours on weekends?",
        "I forgot my account username; how can I recover access to my profile?",
        "Can I switch my plan from annual billing to monthly billing?",
        "The item I received arrived with a damaged corner. How do I start a return?",
        "Where can I download a PDF copy of my past receipts for tax filing?",
        "My discount code isn't applying at checkout—is there a minimum spend?",
        "How long does standard shipping usually take for domestic orders?",
        "Can I change the delivery date for a package that hasn't shipped yet?",
        "I didn't get the verification email when signing up—can you resend it?",
        "How do I export all my saved data before closing my account?",
        "Is there a student discount available for the pro tier?",
        "My payment card expired last month; how do I add a new payment method?",
        "Can I add a second team member to my workspace seat?",
        "Why was my order split into two separate shipments?",
        "How do I cancel my subscription so I am not billed next cycle?",
        "Do you offer gift receipts when purchasing items for someone else?",
        "I see a pending authorization on my card after canceling an order.",
        "How do I contact billing support for a tax-exempt organization?",
        "Can I pause my account for two months while traveling?",
        "What is your return window for unopened electronics?",
        "How do I update my notification preferences for order alerts?",
        "I entered the wrong apartment number on my shipping address—what should I do?",
        "Are refurbished items covered by the same warranty as new ones?",
        "How do I check the status of my warranty replacement request?",
        "Can I pay via bank transfer instead of a credit card?",
        "Why did my promo code say it expired today?",
        "How do I merge two accounts that I accidentally created with different emails?",
        "What happens to my saved projects if I downgrade to the free plan?",
        "How can I request a VAT invoice for my company purchase?",
        "My package says delivered, but I can't find it at my door.",
        "How do I turn on two-factor authentication for my account?",
        "Can I schedule a callback with a support specialist tomorrow morning?",
        "Do you ship internationally, and what duties might apply?",
        "How do I view my remaining credit balance on my account?",
        "I need to update the email address associated with my login.",
        "Can I get a receipt showing zero balance after a refund?",
        "How do I report a bug with the web checkout form?",
        "Is priority shipping guaranteed to arrive in two business days?",
        "How do I view my subscription renewal date?",
        "Can I transfer my license to another email address?",
        "What formats are supported when uploading attachments to a support ticket?",
        "How do I opt out of promotional newsletters?",
        "My session timed out while filling out a support form—is it saved?",
        "Can I get help choosing between Model A and Model B for home office use?",
        "How do I request a copy of my data processing agreement?",
        "Why is my account temporarily locked after mistyping my password?",
        "How do I leave feedback about a helpful support agent?",
    ]

    summarization_prompts = [
        "Summarize the following meeting notes into five concise bullet points.",
        "Give me a three-sentence executive summary of this quarterly product roadmap.",
        "Extract the key action items and owners from this sprint retrospective text.",
        "Summarize this research abstract in plain language for a first-year student.",
        "Condense this technical postmortem into root cause, impact, and remediation.",
        "Provide a brief TL;DR of this documentation page on rate limiting.",
        "Summarize the main arguments in this article without adding outside opinions.",
        "Turn these raw interview notes into a structured summary of user pain points.",
        "Summarize the release notes for version 2.4, highlighting breaking changes.",
        "Give a short overview of how zero-trust network architecture works based on this text.",
        "Summarize this customer feedback survey in under 120 words.",
        "List the top three takeaways from this database migration guide.",
        "Summarize this policy FAQ clearly so new hires can read it quickly.",
        "Create a bulleted summary of the pros and cons mentioned in this review.",
        "Summarize this incident timeline in chronological order.",
        "Condense this tutorial into a quick checklist of steps.",
        "Summarize the key findings from this benchmark comparison table.",
        "Write a two-sentence summary of what changed between v1 and v2 of the API.",
        "Summarize this lecture transcript and highlight unfamiliar terminology.",
        "Extract only the deadlines and milestones from this project brief.",
        "Summarize this security advisory in non-technical terms.",
        "Give a concise summary of how caching improves read latency here.",
        "Summarize this book chapter and note the central theme.",
        "Condense these support escalations into recurring themes.",
        "Summarize this configuration guide in five short bullets.",
        "Provide a brief summary of the testing strategy described in this document.",
        "Summarize this onboarding checklist for new backend engineers.",
        "Extract the main metrics and percentages mentioned in this report.",
        "Summarize this troubleshooting guide into quick checks first.",
        "Give a one-paragraph summary of this architecture decision record.",
        "Summarize these compliance requirements at a high level.",
        "Condense this comparison of authentication protocols into key differences.",
        "Summarize this changelog entry for end users.",
        "Extract the decision and rationale from this design review summary.",
        "Summarize this dataset card, including size and intended use.",
        "Provide a short summary of the error codes listed in this reference.",
        "Summarize this workshop agenda with time blocks.",
        "Condense this article on vector embeddings into plain English.",
        "Summarize the open questions listed at the end of this proposal.",
        "Give a brief summary of how logging is configured in this service.",
        "Summarize this code review feedback into actionable edits.",
        "Condense this privacy notice overview into key points.",
        "Summarize this performance profiling report and note bottlenecks.",
        "Provide a short summary of the deployment rollback procedure.",
        "Summarize this study guide on linear algebra eigenvalues.",
        "Condense this product brief into value proposition and target users.",
        "Summarize these API deprecation notices and sunset dates.",
        "Give a concise summary of the backup and recovery steps.",
        "Summarize this blog post on automated testing best practices.",
        "Condense these notes on distributed consensus into core ideas.",
    ]

    # Tricky-but-harmless prompts containing security/jailbreak-adjacent words
    tricky_benign_prompts = [
        "In pandas, how do I ignore NaN values when computing a column mean?",
        "Can you explain what a system prompt is and why LLM apps use one?",
        "What makes a strong password policy for an enterprise web application?",
        "How do I configure .gitignore to ignore build artifacts and virtual environments?",
        "In my unit test, I want to ignore warnings from a third-party library—how do I do that?",
        "Why do security guidelines recommend rotating passwords and enforcing a password policy?",
        "How do engineers test whether a system prompt is clear and unambiguous?",
        "In regular expressions, how do I ignore case when matching strings in Python?",
        "Explain the difference between a developer message and a system prompt in chat APIs.",
        "What are best practices for designing a password policy without forcing frequent resets?",
        "How can I tell Git to ignore changes to a tracked config file temporarily?",
        "When writing an AI tutor, how should I structure the system prompt so it stays helpful?",
        "Does NIST recommend length over complexity in a modern password policy?",
        "In pytest, how do I ignore a specific deprecation warning cleanly?",
        "How do LLM guardrails evaluate a system prompt for clarity and safety?",
        "How should an IT admin document a password policy for new employees?",
        "In Python's re module, what does re.IGNORECASE do?",
        "Can you give examples of a good system prompt for a code-review assistant?",
        "How do organizations audit compliance with their internal password policy?",
        "How do I ignore whitespace differences when running git diff?",
        "Explain how system prompt design affects tone and formatting in chatbots.",
        "What rules are typically included in an acceptable use and password policy?",
        "In ESLint, how do I ignore a single line when there is a valid reason?",
        "How do teams version-control a system prompt alongside application code?",
        "Why should password policy rules ban commonly leaked passwords?",
        "How do I pass ignore_index=True to pd.concat and what does it do?",
        "What is the role of a system prompt in setting boundaries for a customer bot?",
        "How do password managers help users comply with a strong password policy?",
        "In Docker, how does a .dockerignore file work and why ignore node_modules?",
        "How can I write evaluation tests for a system prompt without overcomplicating it?",
        "What should be included in a password policy for service accounts?",
        "In grep, which flag lets me ignore case while searching log files?",
        "How do you debug a system prompt when the assistant gives overly verbose answers?",
        "Explain how MFA complements a password policy in defense-in-depth.",
        "In TypeScript, when is it appropriate to ignore a compiler diagnostic?",
        "How do you structure a system prompt for a translation utility?",
        "What is a common mistake teams make when enforcing a password policy?",
        "In rsync, how do I ignore symlinks or exclude specific subdirectories?",
        "How do system prompt instructions interact with user questions in multi-turn chat?",
        "How should a startup draft its first password policy and access control standard?",
        "In NumPy, how do I ignore NaN values with np.nanmean?",
        "What are examples of system prompt templates for data extraction tasks?",
        "How do you explain a new password policy to non-technical staff?",
        "In flake8, how do I ignore a specific rule code in a config file?",
        "Why is testing edge cases important when refining a system prompt?",
        "How do password policy length minimums affect usability and security?",
        "In tar, how do I exclude or ignore certain paths when creating an archive?",
        "How do you document system prompt changes during model upgrades?",
        "What is the difference between a system prompt and few-shot examples?",
        "How do security teams verify password policy enforcement in Active Directory?",
    ]

    for text in coding_prompts:
        samples.append((text, 0, "benign_coding"))
    for text in support_prompts:
        samples.append((text, 0, "benign_support"))
    for text in summarization_prompts:
        samples.append((text, 0, "benign_summarization"))
    for text in tricky_benign_prompts:
        samples.append((text, 0, "benign_tricky"))

    return samples


def load_deepset_injections() -> pd.DataFrame:
    """Download deepset/prompt-injections and normalize columns."""
    ds = load_dataset("deepset/prompt-injections")
    frames: List[pd.DataFrame] = []

    for split_name, split_ds in ds.items():
        df_part = split_ds.to_pandas()
        # Expected columns in deepset/prompt-injections: 'text' and 'label' (0 benign, 1 injection)
        if "text" not in df_part.columns or "label" not in df_part.columns:
            raise ValueError(
                f"Unexpected columns in deepset/prompt-injections split '{split_name}': {list(df_part.columns)}"
            )
        df_clean = pd.DataFrame(
            {
                "text": df_part["text"].astype(str).str.strip(),
                "label": df_part["label"].astype(int),
                "source": f"deepset_{split_name}",
            }
        )
        frames.append(df_clean)

    combined = pd.concat(frames, ignore_index=True)
    combined = combined[combined["text"].str.len() > 0].drop_duplicates(subset=["text"]).reset_index(drop=True)
    return combined


def print_class_balance(name: str, df: pd.DataFrame) -> None:
    """Print total rows, class counts, and class percentages for a split."""
    counts = df["label"].value_counts().sort_index()
    total = len(df)
    c0 = int(counts.get(0, 0))
    c1 = int(counts.get(1, 0))
    p0 = (c0 / total * 100.0) if total else 0.0
    p1 = (c1 / total * 100.0) if total else 0.0
    print(f"[{name}] total={total} | benign(0)={c0} ({p0:.2f}%) | injection(1)={c1} ({p1:.2f}%)")


def main() -> None:
    DATA_DIR.mkdir(parents=True, exist_ok=True)

    deepset_df = load_deepset_injections()

    benign_tuples = generate_benign_prompts()
    if len(benign_tuples) < 200:
        raise ValueError(f"Expected at least 200 generated benign prompts, got {len(benign_tuples)}")

    generated_df = pd.DataFrame(benign_tuples, columns=["text", "label", "source"])

    merged_df = pd.concat([deepset_df, generated_df], ignore_index=True)
    merged_df = merged_df.drop_duplicates(subset=["text"]).reset_index(drop=True)

    # Stratified 70 / 15 / 15 split:
    # First split off 30% for (val + test), leaving 70% for train.
    train_df, temp_df = train_test_split(
        merged_df,
        test_size=0.30,
        random_state=RANDOM_SEED,
        stratify=merged_df["label"],
    )

    # Split the 30% remainder equally into 15% val and 15% test (0.5 of 0.30 = 0.15).
    val_df, test_df = train_test_split(
        temp_df,
        test_size=0.50,
        random_state=RANDOM_SEED,
        stratify=temp_df["label"],
    )

    train_path = DATA_DIR / "train.csv"
    val_path = DATA_DIR / "val.csv"
    test_path = DATA_DIR / "test.csv"
    merged_path = DATA_DIR / "merged_dataset.csv"

    train_df.to_csv(train_path, index=False)
    val_df.to_csv(val_path, index=False)
    test_df.to_csv(test_path, index=False)
    merged_df.to_csv(merged_path, index=False)

    print("=== Dataset Class Balance ===")
    print_class_balance("full_merged", merged_df)
    print_class_balance("train_70", train_df)
    print_class_balance("val_15", val_df)
    print_class_balance("test_15", test_df)
    print(f"Saved splits to: {DATA_DIR}")


if __name__ == "__main__":
    main()