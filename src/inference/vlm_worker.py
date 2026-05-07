"""Wrappers around llama.cpp for one image -> caption.

* ``VlmWorker`` (subprocess) — calls ``llama-mtmd-cli`` per request. Used by
  the standalone benchmarks (Steps 4, 5) where each call's load+infer time
  is the data point we want.
* ``VlmServerWorker`` (HTTP) — talks to a long-lived ``llama-server`` over
  ``/v1/chat/completions``. Used by Step 7b production where model load is
  amortized once at server startup. ``llama.cpp`` returns a ``timings``
  block on chat completions so we can populate the same ``VlmTiming``.
"""

from __future__ import annotations

import base64
import logging
import re
import subprocess
import time
from dataclasses import dataclass, field
from pathlib import Path

import requests

log = logging.getLogger(__name__)

# Match newer llama.cpp "llama_perf_..." lines AND older "llama_print_timings".
# Examples:
#   llama_perf_context_print:        load time =     1234.56 ms
#   llama_perf_context_print: prompt eval time =      842.10 ms /  2920 tokens (    0.29 ms per token,  3469.20 tokens per second)
#   llama_perf_context_print:        eval time =      512.30 ms /    96 runs   (    5.34 ms per token,   187.42 tokens per second)
#   llama_perf_context_print:       total time =     2400.00 ms /  3016 tokens
_RE_LOAD = re.compile(r"load time\s*=\s*([\d.]+)\s*ms")
_RE_PEVAL = re.compile(r"prompt eval time\s*=\s*([\d.]+)\s*ms\s*/\s*(\d+)\s*tokens")
# negative lookbehind: don't re-match the "prompt eval time" line.
_RE_EVAL = re.compile(r"(?<!prompt )eval time\s*=\s*([\d.]+)\s*ms\s*/\s*(\d+)\s*(?:runs|tokens)")
_RE_TOTAL = re.compile(r"total time\s*=\s*([\d.]+)\s*ms")
_RE_TPS = re.compile(r"([\d.]+)\s*tokens per second")
_RE_THINK = re.compile(r"<think>.*?</think>\s*", re.DOTALL)


@dataclass
class VlmTiming:
    load_ms: float = float("nan")
    prompt_eval_ms: float = float("nan")
    eval_ms: float = float("nan")
    total_ms: float = float("nan")
    n_prompt_tokens: int = 0
    n_eval_tokens: int = 0
    prompt_eval_tps: float = float("nan")
    eval_tps: float = float("nan")
    wall_ms: float = float("nan")  # outer subprocess wallclock

    @property
    def inference_ms(self) -> float:
        """Production-relevant time: prompt eval + token gen, excluding model load."""
        if self.prompt_eval_ms != self.prompt_eval_ms or self.eval_ms != self.eval_ms:
            return float("nan")
        return self.prompt_eval_ms + self.eval_ms


@dataclass
class VlmResult:
    caption: str
    timing: VlmTiming
    stdout: str
    stderr: str
    returncode: int


def _extract_caption(stdout: str, prompt: str) -> str:
    """Best-effort: strip echoed prompt + ``<think>...</think>`` tags.

    This build of mtmd-cli has no ``--no-display-prompt`` flag, so it prints
    the prompt before the generated tokens. Nemotron *-Reasoning models emit a
    ``<think>...</think>`` block before the answer; for our 0.5 fps caption
    use case we want only the visible answer text.
    """
    s = stdout.strip()
    p = prompt.strip()
    if p and s.startswith(p):
        s = s[len(p):].lstrip()
    elif p:
        idx = s.find(p)
        if 0 <= idx <= 200:
            s = s[idx + len(p):].lstrip()
    s = _RE_THINK.sub("", s)
    return s.strip()


def _parse_timing(stderr: str, wall_ms: float) -> VlmTiming:
    t = VlmTiming(wall_ms=wall_ms)
    if (m := _RE_LOAD.search(stderr)):
        t.load_ms = float(m.group(1))
    if (m := _RE_PEVAL.search(stderr)):
        t.prompt_eval_ms = float(m.group(1))
        t.n_prompt_tokens = int(m.group(2))
        # tps appears at end of same line
        line = stderr[m.start() : stderr.find("\n", m.start())]
        if (mt := _RE_TPS.search(line)):
            t.prompt_eval_tps = float(mt.group(1))
    if (m := _RE_EVAL.search(stderr)):
        t.eval_ms = float(m.group(1))
        t.n_eval_tokens = int(m.group(2))
        line = stderr[m.start() : stderr.find("\n", m.start())]
        if (mt := _RE_TPS.search(line)):
            t.eval_tps = float(mt.group(1))
    if (m := _RE_TOTAL.search(stderr)):
        t.total_ms = float(m.group(1))
    return t


@dataclass
class VlmWorker:
    binary: str
    model: str
    mmproj: str
    prompt: str = "これはなんですか？1〜2文の日本語で簡潔に答えてください。"
    ngl: int = 99
    ctx_size: int = 8192
    n_predict: int = 96
    temp: float = 0.2
    extra_args: list[str] = field(default_factory=list)

    def __post_init__(self) -> None:
        for label, p in [("binary", self.binary), ("model", self.model), ("mmproj", self.mmproj)]:
            if not Path(p).exists():
                log.warning("%s does not exist: %s", label, p)

    def predict_image(  # noqa: PLR0913
        self,
        image_path: str | Path,
        prompt: str | None = None,
        timeout: float = 180.0,
    ) -> VlmResult:
        used_prompt = prompt if prompt is not None else self.prompt
        cmd = [
            self.binary,
            "-m", self.model,
            "--mmproj", self.mmproj,
            "--image", str(image_path),
            "-p", used_prompt,
            "-c", str(self.ctx_size),
            "-n", str(self.n_predict),
            "-ngl", str(self.ngl),
            "--temp", str(self.temp),
            *self.extra_args,
        ]
        log.info("running: %s", " ".join(cmd))
        t0 = time.perf_counter()
        # ``errors='replace'`` because llama.cpp's stderr can contain non-UTF-8
        # bytes from the model-loading progress display.
        proc = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            encoding="utf-8",
            errors="replace",
            timeout=timeout,
        )
        wall_ms = (time.perf_counter() - t0) * 1000
        timing = _parse_timing(proc.stderr, wall_ms)
        return VlmResult(
            caption=_extract_caption(proc.stdout, used_prompt),
            timing=timing,
            stdout=proc.stdout,
            stderr=proc.stderr,
            returncode=proc.returncode,
        )


@dataclass
class VlmServerWorker:
    """HTTP client for a long-running ``llama-server --mmproj ...`` process.

    Sends one chat completion request per call: ``user`` message with text
    prompt + image (base64 data URL). Parses caption from ``choices[0].
    message.content`` and timings from llama.cpp's ``timings`` block.
    """

    base_url: str = "http://127.0.0.1:8081"
    prompt: str = "これはなんですか？1〜2文の日本語で簡潔に答えてください。"
    system_prompt: str = ""  # empty = no system message; set to lock output language
    n_predict: int = 96
    temperature: float = 0.2
    timeout: float = 60.0
    jpeg_quality: int = 90  # used only by callers; this class takes raw JPEG bytes

    def health(self, timeout: float = 2.0) -> bool:
        try:
            r = requests.get(f"{self.base_url.rstrip('/')}/health", timeout=timeout)
            return r.status_code == 200
        except requests.RequestException:
            return False

    def predict_jpeg(self, jpeg_bytes: bytes, prompt: str | None = None) -> VlmResult:
        b64 = base64.b64encode(jpeg_bytes).decode("ascii")
        messages: list[dict] = []
        if self.system_prompt:
            messages.append({"role": "system", "content": self.system_prompt})
        messages.append(
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": prompt if prompt is not None else self.prompt},
                    {
                        "type": "image_url",
                        "image_url": {"url": f"data:image/jpeg;base64,{b64}"},
                    },
                ],
            }
        )
        body = {
            "messages": messages,
            "max_tokens": self.n_predict,
            "temperature": self.temperature,
            "stream": False,
        }
        url = f"{self.base_url.rstrip('/')}/v1/chat/completions"
        t0 = time.perf_counter()
        try:
            r = requests.post(url, json=body, timeout=self.timeout)
        except requests.RequestException as e:
            wall_ms = (time.perf_counter() - t0) * 1000
            return VlmResult(
                caption="",
                timing=VlmTiming(wall_ms=wall_ms),
                stdout="",
                stderr=f"http error: {e}",
                returncode=-1,
            )
        wall_ms = (time.perf_counter() - t0) * 1000
        if r.status_code != 200:
            return VlmResult(
                caption="",
                timing=VlmTiming(wall_ms=wall_ms),
                stdout="",
                stderr=r.text[:500],
                returncode=r.status_code,
            )
        resp = r.json()
        text_raw = resp["choices"][0]["message"]["content"]
        text = _RE_THINK.sub("", text_raw).strip()

        timings = resp.get("timings", {})
        usage = resp.get("usage", {})
        timing = VlmTiming(
            wall_ms=wall_ms,
            load_ms=0.0,  # server is persistent, no load per call
            prompt_eval_ms=float(timings.get("prompt_ms", float("nan"))),
            eval_ms=float(timings.get("predicted_ms", float("nan"))),
            n_prompt_tokens=int(usage.get("prompt_tokens", 0)),
            n_eval_tokens=int(usage.get("completion_tokens", 0)),
            prompt_eval_tps=float(timings.get("prompt_per_second", float("nan"))),
            eval_tps=float(timings.get("predicted_per_second", float("nan"))),
            total_ms=wall_ms,
        )
        return VlmResult(caption=text, timing=timing, stdout=text_raw, stderr="", returncode=0)

    def predict_image(self, image_path: str | Path, prompt: str | None = None) -> VlmResult:
        return self.predict_jpeg(Path(image_path).read_bytes(), prompt=prompt)
