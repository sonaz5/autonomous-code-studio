import os
import json
import time
from datetime import datetime
from langchain_google_genai import ChatGoogleGenerativeAI
from deepeval.metrics import GEval
from deepeval.test_case import LLMTestCase, SingleTurnParams
from deepeval.models import DeepEvalBaseLLM

# --- Configuration -----------------------------------------------------------
# Deliberately two different models: the model that WRITES the code review
# should not be the same model that GRADES it. A model scoring its own output
# tends to rate it more favorably than an independent evaluator would --
# this is "self-preference bias" and it quietly inflates eval scores until
# you can no longer trust them. Different checkpoint, different training run,
# more honest signal.
#
# shutdown date, and Google has revised its own recommended-replacement
# model without a changelog entry. Treat these defaults as best-known-good
# override here without touching the rest of the file:
#   export GENERATOR_MODEL=gemini-x-flash
#   export JUDGE_MODEL=gemini-x-flash-lite
GENERATOR_MODEL = os.environ.get("GENERATOR_MODEL", "gemini-3.6-flash")       # writes the code review
JUDGE_MODEL = os.environ.get("JUDGE_MODEL", "gemini-3.1-flash-lite")          # grades it against SENIOR_CRITERIA
MAX_RETRIES = 3
RETRY_DELAY = 60


def verify_model(model_name: str, role: str) -> None:
    """Fail fast and clearly if a model name is dead, instead of burying a
    404 three layers deep inside a retry loop."""
    try:
        ChatGoogleGenerativeAI(model=model_name).invoke("ping")
    except Exception as exc:
        raise SystemExit(
            f"\n[Startup check failed] {role} model '{model_name}' did not respond: {exc}\n"
            f"Run `python list_models.py` to see which models your API key can "
            f"currently call, then set the {role.upper()}_MODEL env var to one of them.\n"
        )


def extract_text(content: list | str) -> str:
    """Extracts plain text from a LangChain message's .content, which can be
    either a plain string or a list of content-part dicts depending on the
    model/provider. Shared by both the generator and the judge so there's
    only one place that needs to know about this quirk."""
    if isinstance(content, list):
        return "".join(part.get("text", "") for part in content if isinstance(part, dict) and "text" in part)
    return str(content)


class GeminiJudge(DeepEvalBaseLLM):
    """Wraps a Gemini model so DeepEval can use it as an evaluation judge.

    This is instantiated with JUDGE_MODEL, never GENERATOR_MODEL -- see the
    note above CONFIGURATION for why that separation matters.
    """

    def __init__(self, model_name: str = JUDGE_MODEL):
        self.model_name = model_name
        self.model = ChatGoogleGenerativeAI(model=model_name)

    def load_model(self):
        return self.model

    def generate(self, prompt: str) -> str:
        response = self.model.invoke(prompt)
        return extract_text(response.content)

    async def a_generate(self, prompt: str) -> str:
        response = await self.model.ainvoke(prompt)
        return extract_text(response.content)

    def get_model_name(self) -> str:
        return self.model_name


SENIOR_CRITERIA = """
Evaluate the provided Code Review. A 'Senior Engineer' quality review MUST address:
1. Big-O Complexity: Time and space complexity analysis.
2. Security: Vulnerability detection (eval, injection, etc.).
3. Readability & Standards: Clean structure and PEP 8 compliance.
4. Robustness: Edge cases and exception handling.
"""


def save_log_to_json(score: float, reason: str, is_successful: bool) -> None:
    """Appends evaluation results to evaluation_logs.json."""
    log_entry = {
        "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "generator_model": GENERATOR_MODEL,
        "judge_model": JUDGE_MODEL,
        "score": score,
        "is_successful": is_successful,
        "reason": reason,
    }

    log_filename = "evaluation_logs.json"
    logs = []

    if os.path.exists(log_filename):
        with open(log_filename, "r", encoding="utf-8") as file:
            try:
                logs = json.load(file)
            except json.JSONDecodeError:
                pass

    logs.append(log_entry)

    with open(log_filename, "w", encoding="utf-8") as file:
        json.dump(logs, file, indent=4, ensure_ascii=False)

    print(f"\n[Logging] Results successfully saved to '{log_filename}'")


def run_evaluation() -> None:
    if "GEMINI_API_KEY" not in os.environ:
        print("Error: GEMINI_API_KEY environment variable is missing.")
        return

    print("=" * 60)
    print(f"Agent Evaluation Harness")
    print(f"Generator: {GENERATOR_MODEL}  |  Judge: {JUDGE_MODEL}")
    print("=" * 60)

    print("Checking both models respond before asking for code...")
    verify_model(GENERATOR_MODEL, "generator")
    verify_model(JUDGE_MODEL, "judge")
    print("Both models OK.\n")

    print("Please paste your multi-line code below.")
    print("Type 'DONE' on a new line and press Enter when you are finished:\n")

    code_lines = []
    while True:
        try:
            line = input()
            if line.strip().upper() == "DONE":
                break
            code_lines.append(line)
        except EOFError:
            break

    user_pasted_code = "\n".join(code_lines)

    if not user_pasted_code.strip():
        print("No code provided. Exiting.")
        return

    print(f"\nCaptured Code Block ({len(code_lines)} lines):\n{'-'*40}\n{user_pasted_code}\n{'-'*40}")

    # Separate model instance purely for generating the review. This is a
    # plain ChatGoogleGenerativeAI, not a GeminiJudge -- it never acts as
    # the judge, so it has no business being wrapped in the judge interface.
    generator = ChatGoogleGenerativeAI(model=GENERATOR_MODEL)

    prompt_for_review = f"""Perform a 'Senior Engineer' level Code Review on the following code.
You MUST explicitly include the following sections in your review:
1. Big-O Complexity: Explicitly state BOTH the Time Complexity AND Space Complexity.
2. Security: Identify any vulnerabilities (like eval, injection).
3. Readability & Clean Code: Note PEP 8 issues and suggest improvements.
4. Robustness: Point out missing error handling or edge cases.

Code to review:
{user_pasted_code}
"""

    # 1. Review Generation Phase
    print("\nThe Agent is reviewing the code, please wait...")
    review_text = ""

    for attempt in range(MAX_RETRIES):
        try:
            response = generator.invoke(prompt_for_review)
            review_text = extract_text(response.content)
            print(f"\n--- Code Review Output ---\n{review_text}\n--------------------------\n")
            break

        except Exception as exc:
            err = str(exc)
            if "429" in err or "RESOURCE_EXHAUSTED" in err:
                print(f"\n[Quota Reached] API limit hit. Waiting {RETRY_DELAY} seconds to retry... (Attempt {attempt + 1}/{MAX_RETRIES})")
                time.sleep(RETRY_DELAY)
                if attempt == MAX_RETRIES - 1:
                    print("Error: Max retries reached. Please try again later when your quota fully resets.")
                    return
            else:
                print(f"API Error: {exc}")
                return

    # 2. DeepEval Scoring Phase -- judged by a different model than the one
    # that wrote the review (see GENERATOR_MODEL / JUDGE_MODEL above).
    judge = GeminiJudge()

    metric = GEval(
        name="Senior Engineer Taste",
        criteria=SENIOR_CRITERIA,
        evaluation_params=[SingleTurnParams.INPUT, SingleTurnParams.ACTUAL_OUTPUT],
        model=judge,
        strict_mode=True,
    )

    test_case = LLMTestCase(input=user_pasted_code, actual_output=review_text)

    print("Executing DeepEval GEval scoring...")
    for attempt in range(MAX_RETRIES):
        try:
            metric.measure(test_case)
            score = metric.score
            reason = metric.reason
            success = metric.is_successful()
            break

        except Exception as exc:
            err = str(exc)
            if "429" in err or "RESOURCE_EXHAUSTED" in err:
                print(f"\n[Quota Reached] DeepEval hit API limit. Waiting {RETRY_DELAY} seconds to retry... (Attempt {attempt + 1}/{MAX_RETRIES})")
                time.sleep(RETRY_DELAY)
                if attempt == MAX_RETRIES - 1:
                    print("Error: Max retries reached for DeepEval scoring. Please try again later.")
                    return
            else:
                print(f"Scoring Error: {exc}")
                return

    print(f"\nScore: {score}")
    print(f"Status: {'Passed' if success else 'Failed'}")
    print(f"Reason: {reason}")

    save_log_to_json(score=score, reason=reason, is_successful=success)


if __name__ == "__main__":
    run_evaluation()