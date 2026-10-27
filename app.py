"""MangARTI web translator.

Serves the page in web/ and a small JSON API backed by the MangARTI adapter.

Usage:
    python app.py                                  # load the adapter from Hugging Face
    python app.py --model lora_translation_model   # load a locally trained adapter
    python app.py --mock                           # UI development without a GPU
    python app.py --host 0.0.0.0 --port 8000       # expose on your network

The server starts right away and loads the model in the background. The page
shows the loading status until the model is ready.
"""

import argparse
import threading
import time
import warnings
from pathlib import Path

import uvicorn
from fastapi import FastAPI, HTTPException
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field

from prompt import ALPACA_PROMPT, TO_INDONESIAN, TO_MANADO

ROOT = Path(__file__).resolve().parent
WEB_DIR = ROOT / "web"
DEFAULT_MODEL = "Jmnlalu/MangARTI"
MAX_INPUT_CHARS = 500

INSTRUCTIONS = {"id-mdo": TO_MANADO, "mdo-id": TO_INDONESIAN}


class Translator:
    """Loads the model once in a background thread and runs one generation at a time."""

    def __init__(self, model_name: str, mock: bool = False):
        self.model_name = model_name
        self.mock = mock
        self.model = None
        self.tokenizer = None
        self.state = "loading"  # loading | ready | error
        self.error = None
        self._gpu_lock = threading.Lock()

    def load(self):
        if self.mock:
            time.sleep(1.5)
            self.state = "ready"
            return
        try:
            from transformers import logging
            from unsloth import FastLanguageModel

            logging.set_verbosity_error()
            warnings.filterwarnings("ignore")
            print(f"Loading {self.model_name} ...")
            model, tokenizer = FastLanguageModel.from_pretrained(
                model_name=self.model_name,
                max_seq_length=512,
                dtype=None,
                load_in_4bit=True,
            )
            FastLanguageModel.for_inference(model)
            tokenizer.padding_side = "left"
            self.model, self.tokenizer = model, tokenizer
            self.state = "ready"
            print("Model ready.")
        except Exception as exc:  # surfaced to the page through /api/status
            self.state, self.error = "error", str(exc)
            print(f"Failed to load model: {exc}")

    def translate(self, text: str, direction: str, max_new_tokens: int, do_sample: bool,
                  temperature: float, top_p: float) -> str:
        if self.mock:
            time.sleep(0.8)
            return f"[mock] {text}"

        prompt = ALPACA_PROMPT.format(INSTRUCTIONS[direction], text)
        inputs = self.tokenizer([prompt], return_tensors="pt", padding=True).to(self.model.device)
        gen_kwargs = {
            "input_ids": inputs.input_ids,
            "attention_mask": inputs.attention_mask,
            "max_new_tokens": max_new_tokens,
            "use_cache": False,
            "pad_token_id": self.tokenizer.eos_token_id,
            "do_sample": do_sample,
        }
        if do_sample:
            gen_kwargs["temperature"] = temperature
            gen_kwargs["top_p"] = top_p

        with self._gpu_lock:
            outputs = self.model.generate(**gen_kwargs)
        # Decode only the newly generated tokens, not the prompt.
        new_tokens = outputs[0][inputs.input_ids.shape[1]:]
        return self.tokenizer.decode(new_tokens, skip_special_tokens=True).strip()


class TranslateRequest(BaseModel):
    text: str = Field(min_length=1, max_length=MAX_INPUT_CHARS)
    direction: str = Field(pattern="^(id-mdo|mdo-id)$")
    max_new_tokens: int = Field(default=128, ge=16, le=512)
    do_sample: bool = False
    temperature: float = Field(default=0.7, ge=0.1, le=2.0)
    top_p: float = Field(default=0.9, ge=0.1, le=1.0)


def create_app(translator: Translator) -> FastAPI:
    app = FastAPI(title="MangARTI", docs_url="/api/docs", redoc_url=None)

    @app.get("/api/status")
    def status():
        return {"state": translator.state, "model": translator.model_name,
                "mock": translator.mock, "error": translator.error}

    @app.post("/api/translate")
    def translate(req: TranslateRequest):
        if translator.state != "ready":
            raise HTTPException(503, "Model belum siap. Tunggu sebentar lalu coba lagi.")
        text = req.text.strip()
        if not text:
            raise HTTPException(422, "Teks tidak boleh kosong.")
        start = time.perf_counter()
        try:
            result = translator.translate(text, req.direction, req.max_new_tokens,
                                          req.do_sample, req.temperature, req.top_p)
        except Exception as exc:
            raise HTTPException(500, f"Terjemahan gagal: {exc}") from exc
        return {"translation": result, "seconds": round(time.perf_counter() - start, 2)}

    @app.get("/", include_in_schema=False)
    def index():
        return FileResponse(WEB_DIR / "index.html")

    app.mount("/", StaticFiles(directory=WEB_DIR), name="web")
    return app


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--model", default=DEFAULT_MODEL, help="Hugging Face repo id or local adapter folder")
    parser.add_argument("--mock", action="store_true", help="Skip the model and echo input (for UI work)")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=7860)
    args = parser.parse_args()

    translator = Translator(args.model, mock=args.mock)
    threading.Thread(target=translator.load, daemon=True).start()
    print(f"MangARTI running at http://{args.host}:{args.port}")
    uvicorn.run(create_app(translator), host=args.host, port=args.port, log_level="warning")


if __name__ == "__main__":
    main()
