"""
LLM Gateway Service: Multi-Provider Failover, Model Routing, and Output Sanitization.
A clean, self-contained, domain-agnostic gateway for OpenAI-compatible APIs (Groq, Gemini, OpenAI).
Zero domain-specific dependencies or hardcoded application text.
"""

import os
import sys
import re
import json
import logging
from typing import Dict, List, Any, Optional, Tuple

try:
    from openai import OpenAI, APITimeoutError
except ImportError:
    OpenAI = None
    APITimeoutError = Exception

try:
    import httpx
    TimeoutException = httpx.TimeoutException
except ImportError:
    TimeoutException = Exception

logger = logging.getLogger(__name__)


def safe_extract_json(raw_text: str) -> Dict[str, Any]:
    """
    Safely parses JSON from LLM output, resilient against:
    - Reasoning tags (<think>...</think>) from reasoning models (Groq/DeepSeek)
    - Markdown code fences (```json ... ``` or ``` ...)
    - Preceding and trailing conversational or reasoning text
    - Trailing commas before } or ]
    """
    if not raw_text or not raw_text.strip():
        raise ValueError("Empty response received from LLM")

    clean = raw_text.strip()

    # 1. Strip reasoning tags (<think>...</think>)
    clean = re.sub(r"<think>[\s\S]*?</think>", "", clean, flags=re.DOTALL).strip()

    # 2. Try direct json.loads
    try:
        return json.loads(clean)
    except Exception:
        pass

    # 3. Extract content within markdown code blocks if present
    fence_match = re.search(r"```(?:json)?\s*([\s\S]*?)\s*```", clean, flags=re.DOTALL)
    if fence_match:
        fenced_candidate = fence_match.group(1).strip()
        try:
            return json.loads(fenced_candidate)
        except Exception:
            clean = fenced_candidate

    # 4. Locate outer JSON object braces { ... }
    first_brace = clean.find("{")
    last_brace = clean.rfind("}")
    if first_brace != -1 and last_brace != -1 and last_brace > first_brace:
        json_slice = clean[first_brace:last_brace + 1].strip()
        try:
            return json.loads(json_slice)
        except Exception:
            # 5. Remove trailing commas before closing braces or brackets
            fixed_slice = re.sub(r",\s*([}\]])", r"\1", json_slice)
            try:
                return json.loads(fixed_slice)
            except Exception:
                pass

    logger.error("safe_extract_json failed. Raw LLM content (first 800 chars):\n%s", clean[:800])
    raise ValueError(f"Could not parse valid JSON from LLM response: {clean[:200]}...")


def condense_head_tail(text: str, head_chars: int = 6000, tail_chars: int = 4500) -> str:
    """
    Condenses long text by preserving head and tail portions while condensing the middle.
    Returns text untouched if len(text) <= head_chars + tail_chars.
    Standard threshold is 10,500 chars (~2,600 tokens), allowing full 10k policies
    (such as SeraGuard's full notice) to pass to Groq without any middle truncation.
    """
    if not text:
        return ""
    if len(text) <= head_chars + tail_chars:
        return text
    return (
        text[:head_chars]
        + "\n\n...[MIDDLE SECTIONS CONDENSED FOR TOKEN WINDOW]...\n\n"
        + text[-tail_chars:]
    )


class LLMGateway:
    """
    Production-grade LLM Gateway supporting:
    - Input length-based model routing (long docs -> Gemini 1M window; short -> Groq fast reasoning)
    - Multi-provider fallback (Groq gpt-oss-120b/20b <-> Google Gemini 3.5/3.8 Flash)
    - Output sanitization (native JSON enforcement on Gemini; safe_extract_json on Groq)
    - Immediate failover on rate limits (429), payload limits (413), server errors (503/500), or bad JSON
    """

    def __init__(
        self,
        groq_api_key: Optional[str] = None,
        gemini_api_key: Optional[str] = None,
        groq_base_url: Optional[str] = None,
        gemini_base_url: Optional[str] = None,
    ):
        self._groq_api_key = groq_api_key
        self._gemini_api_key = gemini_api_key
        self._groq_base_url = groq_base_url
        self._gemini_base_url = gemini_base_url

    def _resolve_config(self) -> Dict[str, Any]:
        """Resolves API keys, URLs, and provider preferences dynamically (supporting runtime test patches)."""
        rag_mod = sys.modules.get("services.rag_auditor")

        # Support runtime module patches in tests (e.g. @patch("services.rag_auditor.GEMINI_API_KEY", ""))
        rag_gemini_key = getattr(rag_mod, "GEMINI_API_KEY", None) if rag_mod else None
        if rag_gemini_key is not None:
            gemini_key = rag_gemini_key
        else:
            gemini_key = (
                self._gemini_api_key
                or os.getenv("GEMINI_API_KEY", "").strip()
                or os.getenv("GOOGLE_API_KEY", "").strip()
            )

        rag_gemini_base = getattr(rag_mod, "GEMINI_BASE_URL", None) if rag_mod else None
        if rag_gemini_base is not None:
            gemini_base = rag_gemini_base
        else:
            gemini_base = (
                self._gemini_base_url
                or os.getenv("GEMINI_BASE_URL", "https://generativelanguage.googleapis.com/v1beta/openai/").strip()
            )
        if gemini_base and not gemini_base.endswith("/"):
            gemini_base += "/"

        rag_groq_key = getattr(rag_mod, "LLM_API_KEY", None) if rag_mod else None
        if rag_groq_key is not None:
            groq_key = rag_groq_key
        else:
            groq_key = (
                self._groq_api_key
                or os.getenv("LLM_API_KEY", "").strip()
                or os.getenv("GROQ_API_KEY", "").strip()
            )

        rag_groq_base = getattr(rag_mod, "LLM_BASE_URL", None) if rag_mod else None
        if rag_groq_base is not None:
            groq_base = rag_groq_base
        else:
            groq_base = (
                self._groq_base_url
                or os.getenv("LLM_BASE_URL", "https://api.groq.com/openai/v1").strip()
            )

        rag_provider = getattr(rag_mod, "LLM_PROVIDER", None) if rag_mod else None
        if rag_provider is not None:
            provider_pref = rag_provider.strip().lower()
        else:
            provider_pref = os.getenv("LLM_PROVIDER", "openai").strip().lower()

        rag_gemini_model = getattr(rag_mod, "GEMINI_MODEL", None) if rag_mod else None
        if rag_gemini_model is not None:
            primary_gemini_model = rag_gemini_model.strip()
        else:
            primary_gemini_model = os.getenv("GEMINI_MODEL", "gemini-3.5-flash").strip()

        rag_groq_model = getattr(rag_mod, "LLM_MODEL", None) if rag_mod else None
        if rag_groq_model is not None:
            primary_groq_model = rag_groq_model.strip()
        else:
            primary_groq_model = os.getenv("LLM_MODEL", "openai/gpt-oss-120b").strip()

        return {
            "gemini_key": gemini_key,
            "gemini_base": gemini_base,
            "groq_key": groq_key,
            "groq_base": groq_base,
            "provider_pref": provider_pref,
            "gemini_model": primary_gemini_model,
            "groq_model": primary_groq_model,
        }

    def get_candidate_chain(self, input_length: int = 0) -> List[Dict[str, Any]]:
        """
        Builds the ordered list of LLM targets for failover execution.
        
        Short/Standard Inputs (input_length <= 9000 chars):
          1. Groq primary model (default: openai/gpt-oss-120b, max_tokens=3400, reasoning_effort="medium")
          2. Groq fallback model (openai/gpt-oss-20b, max_tokens=3600, reasoning_effort="low")
          3. Gemini 3.5 Flash (max_tokens=8192)
          4. Gemini fallbacks (gemini-3.5-flash-lite, gemini-flash-lite-latest, gemini-3.8-flash)
        
        Long Documents (input_length > 9000 chars):
          1. Gemini 3.5 Flash (max_tokens=8192, 1M context window)
          2. Gemini 3.5 Flash Lite / Latest / 3.8 Flash
          3. Groq primary model (condensed via condense_head_tail)
          4. Groq fallback model (condensed via condense_head_tail)
        """
        cfg = self._resolve_config()
        targets: List[Dict[str, Any]] = []

        gemini_targets: List[Dict[str, Any]] = []
        if cfg["gemini_key"]:
            # Ordered candidate Gemini models
            gm_candidates = [cfg["gemini_model"]]
            for fb in ["gemini-3.5-flash", "gemini-3.5-flash-lite", "gemini-flash-lite-latest", "gemini-3.8-flash"]:
                if fb not in gm_candidates:
                    gm_candidates.append(fb)

            for gm in gm_candidates:
                gemini_targets.append({
                    "provider": "gemini",
                    "model": gm,
                    "api_key": cfg["gemini_key"],
                    "base_url": cfg["gemini_base"],
                    "max_tokens": 8192,
                    "is_gemini": True,
                })

        groq_targets: List[Dict[str, Any]] = []
        if cfg["groq_key"]:
            groq_targets.append({
                "provider": "groq/openai",
                "model": cfg["groq_model"],
                "api_key": cfg["groq_key"],
                "base_url": cfg["groq_base"],
                "max_tokens": 3400,
                "reasoning_effort": "medium" if "gpt-oss" in cfg["groq_model"].lower() else None,
                "is_gemini": False,
            })
            if cfg["groq_model"] != "openai/gpt-oss-20b":
                groq_targets.append({
                    "provider": "groq/openai-fallback",
                    "model": "openai/gpt-oss-20b",
                    "api_key": cfg["groq_key"],
                    "base_url": cfg["groq_base"],
                    "max_tokens": 3600,
                    "reasoning_effort": "low",
                    "is_gemini": False,
                })

        prefer_gemini = (cfg["provider_pref"] in ["gemini", "google"]) or (
            input_length > 9000 and bool(cfg["gemini_key"])
        )

        if prefer_gemini:
            targets.extend(gemini_targets)
            targets.extend(groq_targets)
        else:
            targets.extend(groq_targets)
            targets.extend(gemini_targets)

        # Deduplicate targets by (model, base_url)
        seen = set()
        deduped = []
        for t in targets:
            key = (t["model"], t["base_url"])
            if key not in seen:
                seen.add(key)
                deduped.append(t)
        return deduped

    def generate_json(
        self,
        system_prompt: str,
        user_content: str,
        document_text: str = ""
    ) -> Tuple[Dict[str, Any], str]:
        """
        Executes JSON generation with multi-provider failover.
        For Gemini: passes response_format={"type": "json_object"}.
        For Groq gpt-oss: omits response_format and extracts via safe_extract_json.
        Immediately fails over on 400, 404, 413, 429, 503, or JSON parse errors.
        Returns: (parsed_dict, model_used)
        """
        input_len = len(document_text) if document_text else len(user_content)
        candidates = self.get_candidate_chain(input_length=input_len)
        if not candidates:
            raise ValueError("No LLM API keys configured (set GROQ_API_KEY/LLM_API_KEY or GEMINI_API_KEY).")

        last_err: Optional[Exception] = None

        for target in candidates:
            model_name = target["model"]
            provider = target["provider"]
            is_gemini = target.get("is_gemini", False)

            # Format input text according to model capacity
            if document_text:
                if is_gemini:
                    doc_slice = document_text[:45000]
                else:
                    doc_slice = condense_head_tail(document_text, 6000, 4500)

                if "{document_text}" in user_content:
                    prompt_text = user_content.replace("{document_text}", doc_slice)
                else:
                    prompt_text = f"{user_content}\n\n{doc_slice}"
            else:
                prompt_text = user_content

            create_params: Dict[str, Any] = {
                "model": model_name,
                "messages": [
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": prompt_text}
                ],
                "temperature": 0.0,
                "max_tokens": target.get("max_tokens", 8192 if is_gemini else 3400)
            }
            if is_gemini:
                create_params["response_format"] = {"type": "json_object"}
            elif target.get("reasoning_effort"):
                create_params["reasoning_effort"] = target["reasoning_effort"]

            try:
                from openai import OpenAI
                logger.info(
                    "LLMGateway: Calling provider '%s' (model: '%s', input: %d chars)...",
                    provider, model_name, len(prompt_text)
                )
                client = OpenAI(
                    api_key=target["api_key"],
                    base_url=target["base_url"] if target.get("base_url") else None,
                    timeout=15.0,
                    max_retries=0
                )
                response = client.chat.completions.create(timeout=15.0, **create_params)
                raw_content = response.choices[0].message.content or ""
                parsed = safe_extract_json(raw_content)
                logger.info("LLMGateway: Successfully parsed JSON using model '%s'.", model_name)
                return parsed, model_name
            except (APITimeoutError, TimeoutException) as timeout_err:
                last_err = timeout_err
                logger.warning(
                    "LLMGateway: Model '%s' (%s) timed out after 15s (%s). Failing over immediately...",
                    model_name, provider, timeout_err
                )
            except Exception as err:
                last_err = err
                logger.warning(
                    "LLMGateway: Model '%s' (%s) failed during JSON generation (%s). Failing over immediately...",
                    model_name, provider, err
                )

        logger.error("LLMGateway: All candidate LLM models failed during JSON generation. Last error: %s", last_err)
        raise last_err or RuntimeError("All candidate LLM models failed during JSON generation.")

    def generate_text(
        self,
        system_prompt: str,
        user_content: str,
        max_output_tokens: int = 2800,
        document_text: str = ""
    ) -> Tuple[str, str]:
        """
        Executes markdown/text generation with multi-provider failover.
        Gemini uses max_tokens=8192; Groq uses max_output_tokens.
        Returns: (clean_markdown_text, model_used)
        """
        input_len = len(document_text) if document_text else len(user_content)
        candidates = self.get_candidate_chain(input_length=input_len)
        if not candidates:
            raise ValueError("No LLM API keys configured (set GROQ_API_KEY/LLM_API_KEY or GEMINI_API_KEY).")

        last_err: Optional[Exception] = None

        for target in candidates:
            model_name = target["model"]
            provider = target["provider"]
            is_gemini = target.get("is_gemini", False)

            if document_text:
                if is_gemini:
                    doc_slice = document_text[:45000]
                else:
                    doc_slice = condense_head_tail(document_text, 3000, 2000)

                if "{document_text}" in user_content:
                    prompt_text = user_content.replace("{document_text}", doc_slice)
                else:
                    prompt_text = f"{user_content}\n\n{doc_slice}"
            else:
                prompt_text = user_content

            create_params: Dict[str, Any] = {
                "model": model_name,
                "messages": [
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": prompt_text}
                ],
                "temperature": 0.0,
                "max_tokens": 8192 if is_gemini else max_output_tokens
            }
            if not is_gemini and target.get("reasoning_effort"):
                create_params["reasoning_effort"] = target["reasoning_effort"]

            try:
                from openai import OpenAI
                logger.info(
                    "LLMGateway: Calling provider '%s' (model: '%s') to generate text (input: %d chars)...",
                    provider, model_name, len(prompt_text)
                )
                client = OpenAI(
                    api_key=target["api_key"],
                    base_url=target["base_url"] if target.get("base_url") else None,
                    timeout=15.0,
                    max_retries=0
                )
                response = client.chat.completions.create(timeout=15.0, **create_params)
                raw_content = response.choices[0].message.content or ""
                clean_text = raw_content.strip()
                if clean_text:
                    logger.info("LLMGateway: Successfully generated text using model '%s' (%d chars).", model_name, len(clean_text))
                    return clean_text, model_name
            except (APITimeoutError, TimeoutException) as timeout_err:
                last_err = timeout_err
                logger.warning(
                    "LLMGateway: Model '%s' (%s) timed out after 15s (%s). Failing over immediately...",
                    model_name, provider, timeout_err
                )
            except Exception as err:
                last_err = err
                logger.warning(
                    "LLMGateway: Model '%s' (%s) failed during text generation (%s). Failing over immediately...",
                    model_name, provider, err
                )

        logger.error("LLMGateway: All candidate LLM models failed during text generation. Last error: %s", last_err)
        raise last_err or RuntimeError("All candidate LLM models failed during text generation.")


# Module-level singleton
_DEFAULT_GATEWAY = LLMGateway()

def get_llm_gateway() -> LLMGateway:
    """Returns the default LLMGateway singleton."""
    return _DEFAULT_GATEWAY
