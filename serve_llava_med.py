import argparse
import base64
import io
import time
from typing import Any, Dict, Optional, Tuple

import requests
import torch
from fastapi import FastAPI, Request
from PIL import Image
from transformers import AutoModelForVision2Seq, AutoProcessor
import uvicorn


def _decode_data_url(data_url: str) -> Image.Image:
    header, b64 = data_url.split(",", 1)
    if "base64" not in header:
        raise ValueError("Only base64 data URLs are supported.")
    raw = base64.b64decode(b64)
    return Image.open(io.BytesIO(raw)).convert("RGB")


def _fetch_image(url: str, timeout_s: int = 30) -> Image.Image:
    resp = requests.get(url, timeout=timeout_s)
    resp.raise_for_status()
    return Image.open(io.BytesIO(resp.content)).convert("RGB")


def _extract_text_image(payload: Dict[str, Any]) -> Tuple[str, Optional[Image.Image]]:
    messages = payload.get("messages", [])
    text_parts = []
    image = None
    for msg in messages:
        if msg.get("role") != "user":
            continue
        content = msg.get("content")
        if isinstance(content, str):
            text_parts.append(content)
            continue
        if not isinstance(content, list):
            continue
        for item in content:
            if not isinstance(item, dict):
                continue
            itype = item.get("type")
            if itype == "text":
                text_parts.append(item.get("text", ""))
            elif itype == "image_url":
                img_obj = item.get("image_url", {})
                url = img_obj.get("url") if isinstance(img_obj, dict) else img_obj
                if not url:
                    continue
                if url.startswith("data:"):
                    image = _decode_data_url(url)
                else:
                    image = _fetch_image(url)
    text = "\n".join([t for t in text_parts if t]).strip()
    return text, image


def _build_prompt(text: str) -> str:
    # LLaVA-style prompt with explicit image token.
    if text:
        return f"USER: <image>\n{text}\nASSISTANT:"
    return "USER: <image>\nASSISTANT:"


def _select_dtype(dtype_name: str):
    if dtype_name == "float16":
        return torch.float16
    if dtype_name == "float32":
        return torch.float32
    return torch.bfloat16


def create_app(args) -> FastAPI:
    app = FastAPI()
    device = torch.device(args.device)
    torch_dtype = _select_dtype(args.dtype)

    processor = AutoProcessor.from_pretrained(args.model, trust_remote_code=args.trust_remote_code)
    model = AutoModelForVision2Seq.from_pretrained(
        args.model,
        torch_dtype=torch_dtype,
        trust_remote_code=args.trust_remote_code,
        low_cpu_mem_usage=True,
    )
    model.to(device)
    model.eval()

    @app.post("/v1/chat/completions")
    async def chat_completions(request: Request):
        payload = await request.json()
        text, image = _extract_text_image(payload)
        if image is None:
            return {
                "error": {"message": "Missing image input.", "type": "invalid_request_error"}
            }

        prompt = _build_prompt(text)
        inputs = processor(text=prompt, images=image, return_tensors="pt")
        inputs = {k: v.to(device) for k, v in inputs.items()}

        max_tokens = int(payload.get("max_tokens") or args.max_new_tokens)
        temperature = float(payload.get("temperature") or 0.0)
        top_p = float(payload.get("top_p") or 1.0)

        gen_kwargs = {"max_new_tokens": max_tokens}
        if temperature <= 0:
            gen_kwargs["do_sample"] = False
        else:
            gen_kwargs["do_sample"] = True
            gen_kwargs["temperature"] = temperature
            gen_kwargs["top_p"] = top_p

        with torch.inference_mode():
            output_ids = model.generate(**inputs, **gen_kwargs)
        input_len = inputs["input_ids"].shape[1]
        gen_ids = output_ids[0, input_len:]
        decoded = processor.batch_decode(gen_ids, skip_special_tokens=True)[0].strip()

        n = int(payload.get("n") or 1)
        choices = [
            {
                "index": i,
                "message": {"role": "assistant", "content": decoded},
                "finish_reason": "stop",
            }
            for i in range(n)
        ]
        return {
            "id": f"llava-med-{int(time.time())}",
            "object": "chat.completion",
            "created": int(time.time()),
            "model": payload.get("model") or args.model,
            "choices": choices,
        }

    return app


def main():
    parser = argparse.ArgumentParser(description="LLaVA-Med OpenAI-compatible server")
    parser.add_argument("--host", default="0.0.0.0")
    parser.add_argument("--port", type=int, default=8002)
    parser.add_argument("--model", default="microsoft/llava-med-v1.5-mistral-7b")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--dtype", default="bfloat16", choices=["bfloat16", "float16", "float32"])
    parser.add_argument("--max-new-tokens", type=int, default=64)
    parser.add_argument("--trust-remote-code", action="store_true")
    args = parser.parse_args()

    app = create_app(args)
    uvicorn.run(app, host=args.host, port=args.port, log_level="info")


if __name__ == "__main__":
    main()
