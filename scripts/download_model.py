"""Download the Phase-0 base model (GPT-2, 124M params) via urllib.

Uses the local proxy (registry settings) which is the only reliable TLS path
on this machine. Downloads files into models/hf/gpt2 so that transformers can
load the model fully offline afterwards.
"""
from __future__ import annotations

import os
import sys
import time
import urllib.request

REPO = "https://huggingface.co/gpt2/resolve/main"
FILES = [
    "config.json", "generation_config.json", "merges.txt",
    "special_tokens_map.json", "tokenizer.json", "tokenizer_config.json",
    "vocab.json", "model.safetensors",
]
OUT_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                       "models", "hf", "gpt2")


def build_opener():
    proxies = urllib.request.getproxies()
    proxy_handler = urllib.request.ProxyHandler(proxies or {})
    return urllib.request.build_opener(proxy_handler)


def download(url: str, dest: str, retries: int = 6) -> bool:
    if os.path.exists(dest) and os.path.getsize(dest) > 0:
        print(f"skip (exists) {os.path.basename(dest)}")
        return True
    opener = build_opener()
    tmp = dest + ".part"
    for attempt in range(retries):
        try:
            t0 = time.time()
            req = urllib.request.Request(url, headers={"User-Agent": "dscns-setup"})
            with opener.open(req, timeout=60) as resp, open(tmp, "wb") as f:
                total = int(resp.headers.get("Content-Length", 0))
                done = 0
                while True:
                    chunk = resp.read(1 << 20)
                    if not chunk:
                        break
                    f.write(chunk)
                    done += len(chunk)
            os.replace(tmp, dest)
            print(f"ok {os.path.basename(dest)} {done / 1e6:.1f}MB "
                  f"in {time.time() - t0:.0f}s")
            return True
        except Exception as e:
            print(f"attempt {attempt + 1} failed for {os.path.basename(url)}: "
                  f"{type(e).__name__} {str(e)[:80]}")
            time.sleep(3)
    return False


def main():
    os.makedirs(OUT_DIR, exist_ok=True)
    ok = True
    for f in FILES:
        if not download(f"{REPO}/{f}", os.path.join(OUT_DIR, f)):
            ok = False
    if not os.path.exists(os.path.join(OUT_DIR, "model.safetensors")):
        print("safetensors missing -> trying pytorch_model.bin")
        ok = download(f"{REPO}/pytorch_model.bin",
                      os.path.join(OUT_DIR, "pytorch_model.bin")) and ok
    print("DONE" if ok else "INCOMPLETE")


if __name__ == "__main__":
    main()
