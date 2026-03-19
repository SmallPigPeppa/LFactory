from __future__ import annotations

from pathlib import Path
from time import perf_counter
from typing import Any

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from grounding import build_grounding_context


MODEL_CANDIDATES = [
    Path("./output/phi4-mini-intune-merged-v1-f16"),
    Path("./output/phi4-mini-intune-merged-v1"),
    Path("./output/phi4-mini-intune-merged"),
]
DEFAULT_SYSTEM_PROMPT = (
    "You are a senior Microsoft Intune and Windows endpoint support engineer. "
    "Answer like an experienced L1/L2 enterprise support technician, not a generic chatbot. "
    "Be specific to Intune, Entra ID, Windows Update, Conditional Access, Autopilot, "
    "BitLocker, IME, and Windows endpoint diagnostics. "
    "Do not add generic filler, motivational language, or vague advice. "
    "If an error code is present, explain what it usually means in this Microsoft endpoint context. "
    "Prefer concrete checks such as Event Viewer paths, registry paths, services, dsregcmd output, "
    "IME logs, Intune portal pages, Windows Update logs, and PowerShell commands. "
    "Return plain text only in exactly this skeleton and stop after the last line: "
    "Issue Summary: ... "
    "Likely Causes: - ... - ... "
    "Checks To Run: - ... - ... "
    "Remediation: - ... - ... "
    "Escalate If: - ... "
    "Use at most 2 bullets per section and keep each bullet short. "
    "Keep the full answer compact enough to fit in about 140 to 200 generated tokens. "
    "Avoid markdown headings, code fences, long explanations, repeated wording, and invented registry values. "
    "If you are unsure, say what to verify instead of inventing facts."
)


def pick_model_path() -> Path:
    for candidate in MODEL_CANDIDATES:
        if candidate.exists():
            return candidate
    raise FileNotFoundError(
        "No merged Phi model found. Expected one of: "
        + ", ".join(str(path) for path in MODEL_CANDIDATES)
    )


class PhiRuntime:
    def __init__(
        self,
        model_path: str | Path | None = None,
        system_prompt: str = DEFAULT_SYSTEM_PROMPT,
    ) -> None:
        self.model_path = Path(model_path) if model_path else pick_model_path()
        self.system_prompt = system_prompt
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.dtype = torch.bfloat16 if self.device == "cuda" else torch.float16

        self.tokenizer = AutoTokenizer.from_pretrained(
            str(self.model_path),
            trust_remote_code=False,
        )
        self.end_token_id = self.tokenizer.convert_tokens_to_ids("<|end|>")
        self.model = AutoModelForCausalLM.from_pretrained(
            str(self.model_path),
            dtype=self.dtype,
            device_map=self.device,
            trust_remote_code=False,
        )
        self.model.eval()

    def device_name(self) -> str:
        if torch.cuda.is_available():
            return torch.cuda.get_device_name(0)
        return "cpu"

    def build_prompt(
        self,
        prompt: str,
        system_prompt: str | None = None,
        grounding: dict[str, Any] | None = None,
    ) -> str:
        grounding = grounding or build_grounding_context(prompt)
        user_content = prompt
        if grounding["context_text"]:
            user_content = (
                "Important grounding rules:\n"
                "- Treat the grounding context below as factual and higher priority than model memory.\n"
                "- Do not contradict any official error reference in the grounding context.\n"
                "- If an official error reference is present, mention that meaning directly in Issue Summary or Likely Causes.\n"
                "- If the grounding context is limited, say what to verify instead of inventing details.\n\n"
                "Grounding context:\n"
                f"{grounding['context_text']}\n\n"
                f"User question:\n{prompt}"
            )

        messages = [
            {"role": "system", "content": system_prompt or self.system_prompt},
            {"role": "user", "content": user_content},
        ]
        return self.tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
        )

    def generate(
        self,
        prompt: str,
        *,
        system_prompt: str | None = None,
        max_new_tokens: int = 320,
        temperature: float = 0.2,
        top_p: float = 0.9,
        repetition_penalty: float = 1.05,
        do_sample: bool | None = None,
    ) -> dict[str, Any]:
        grounding = build_grounding_context(prompt)
        prompt_text = self.build_prompt(
            prompt,
            system_prompt=system_prompt,
            grounding=grounding,
        )
        inputs = self.tokenizer(prompt_text, return_tensors="pt").to(self.device)
        start = perf_counter()
        sample = temperature > 0 if do_sample is None else do_sample
        generation_kwargs = {
            "max_new_tokens": max_new_tokens,
            "do_sample": sample,
            "repetition_penalty": repetition_penalty,
            "pad_token_id": self.tokenizer.eos_token_id,
            "eos_token_id": [self.tokenizer.eos_token_id, self.end_token_id],
        }
        if sample:
            generation_kwargs["temperature"] = temperature
            generation_kwargs["top_p"] = top_p

        with torch.no_grad():
            output = self.model.generate(
                **inputs,
                **generation_kwargs,
            )

        elapsed = perf_counter() - start
        new_tokens = output[0][inputs["input_ids"].shape[1] :]
        reply = self.tokenizer.decode(new_tokens, skip_special_tokens=True).strip()
        reply = reply.split("<|end|>", 1)[0].strip()
        for marker in [
            "Use at most 2 bullets per section and keep each bullet short.",
            "Keep the full answer compact enough to fit in about 140 to 200 generated tokens.",
        ]:
            reply = reply.split(marker, 1)[0].rstrip()
        generated_tokens = int(new_tokens.shape[0])

        return {
            "reply": reply,
            "model_path": str(self.model_path),
            "device": self.device,
            "device_name": self.device_name(),
            "input_tokens": int(inputs["input_ids"].shape[1]),
            "generated_tokens": generated_tokens,
            "latency_seconds": round(elapsed, 3),
            "tokens_per_second": round(generated_tokens / elapsed, 2) if elapsed else None,
            "grounding_error_hits": len(grounding["error_rows"]),
            "grounding_knowledge_hits": len(grounding["knowledge_rows"]),
            "grounding_context": grounding["context_text"],
        }
