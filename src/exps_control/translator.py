"""
Code-to-NL translator using GPT 5.2 via OpenRouter.

Uses the consistency-optimized ICL prompt to translate code into
natural language reasoning that is stylistically indistinguishable
from original NL reasoning.
"""

import os
from pathlib import Path
from typing import Optional

import httpx
from dotenv import load_dotenv

load_dotenv()

# Load the ICL prompt from the consistency optimization
ICL_PROMPT_PATH = Path(__file__).parent.parent / "exps_performance" / "icl_consistency_FINAL.md"

DEFAULT_ICL_PROMPT = """
# ULTRA-UNPOLISHED ICL PROMPT: Code → "me kinda mumbling what I think it does"

You'll get a chunk of code. Write what it does the way I'd explain it to a friend after staring at it for a minute and only *mostly* getting it. The goal is to sound like a real person being a bit sloppy, not like a neat textbook summary.

## Style rules (more important than being correct/complete)
- **Casual + unsure:** add **3–6** little hedges like "I think…", "kinda", "basically", "pretty sure", "from what I can tell", "idk", "maybe".
- **Imperfect flow:** short sentences. Fragments. A tiny run-on is fine. Include **one** quick self-correction (e.g., "wait—no, actually…"). Optional small aside in parentheses.
- **Don't tie it up nicely:** avoid a clean conclusion or "this is optimal" vibe. Leave a little fuzziness. It should feel like you're recalling it, not presenting it.
- **Selective:** only **1–2 main takeaways**. Do **not** walk through every loop/branch. No step-by-step. No mini-derivations.
- **Use 2–4 anchors from the code** by name (like `dp[i][j]`, `parent`, `find(x)`, `edges.sort(...)`, `memo`, `while`, etc.). Mention them naturally, not as a list.
- **Avoid precision:** don't include exact numeric results, worked examples, or "if dims are X then Y" type calculations. No specific computed totals.

## What to avoid (big giveaways)
- **No famous technique names** (no "Dijkstra", "BFS", "DP", "greedy", "topological", etc.).
- No lecture voice: don't say "This code implements…", don't do "First/Second/Finally", no bullet points.
- No formal definitions ("X is used to…" / "in order to…" / "therefore…").
- Don't over-justify. Don't sound confident.

## Length / formatting
- **90–170 words** total.
- **1–2 short paragraphs max.**
- Readable, but slightly messy is good.

## Output format requirement
- Write the explanation text only.
- End with **exactly one plain closing line** that starts with: **"So in the end it returns …"** (finish the sentence).
""".strip()


def load_icl_prompt() -> str:
    """Load the ICL prompt from file or use default."""
    if ICL_PROMPT_PATH.exists():
        return ICL_PROMPT_PATH.read_text().strip()
    return DEFAULT_ICL_PROMPT


class CodeToNLTranslator:
    """Translates code to natural language using GPT 5.2."""

    def __init__(
        self,
        model: str = "openai/gpt-5.2",
        api_key: Optional[str] = None,
        base_url: str = "https://openrouter.ai/api/v1",
    ):
        self.model = model
        self.api_key = api_key or os.getenv("OPENROUTER_API_KEY")
        self.base_url = base_url
        self.icl_prompt = load_icl_prompt()

        if not self.api_key:
            raise ValueError("OPENROUTER_API_KEY not found in environment")

    def translate(self, code: str, timeout: float = 60.0) -> str:
        """Translate code to natural language reasoning.

        Args:
            code: The code to translate
            timeout: Request timeout in seconds

        Returns:
            Natural language explanation of the code
        """
        if not code or not code.strip():
            return ""

        messages = [
            {"role": "system", "content": self.icl_prompt},
            {"role": "user", "content": f"```python\n{code}\n```"},
        ]

        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json",
        }

        payload = {
            "model": self.model,
            "messages": messages,
            "max_tokens": 500,
            "temperature": 0.7,
        }

        with httpx.Client(timeout=timeout) as client:
            response = client.post(
                f"{self.base_url}/chat/completions",
                headers=headers,
                json=payload,
            )
            response.raise_for_status()
            data = response.json()

        return data["choices"][0]["message"]["content"].strip()

    def translate_batch(
        self,
        codes: list[str],
        progress_callback: Optional[callable] = None,
    ) -> list[str]:
        """Translate multiple code snippets.

        Args:
            codes: List of code snippets to translate
            progress_callback: Optional callback(i, total) for progress updates

        Returns:
            List of translated natural language explanations
        """
        translations = []
        for i, code in enumerate(codes):
            if progress_callback:
                progress_callback(i, len(codes))
            try:
                translation = self.translate(code)
            except Exception as e:
                print(f"  Warning: Translation failed for sample {i}: {e}")
                translation = ""
            translations.append(translation)
        return translations


if __name__ == "__main__":
    # Quick test
    translator = CodeToNLTranslator()
    test_code = """
def solution():
    arr = [3, 1, 4, 1, 5, 9, 2, 6]
    n = len(arr)
    for i in range(n):
        for j in range(0, n-i-1):
            if arr[j] > arr[j+1]:
                arr[j], arr[j+1] = arr[j+1], arr[j]
    return arr
"""
    print("Translating test code...")
    result = translator.translate(test_code)
    print(f"Translation:\n{result}")
