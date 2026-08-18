"""Reliable wheel downloader with resume support.

Tries the local proxy first; falls back to direct connection.  Implements
HTTP Range resume so interrupted transfers continue instead of restarting.
"""
from __future__ import annotations

import os
import sys
import time
import urllib.error
import urllib.request

CHUNK = 1 << 20  # 1 MiB


def _opener(use_proxy: bool):
    if use_proxy:
        proxies = urllib.request.getproxies()
        return urllib.request.build_opener(urllib.request.ProxyHandler(proxies))
    return urllib.request.build_opener(urllib.request.ProxyHandler({}))


def download(url: str, dest: str, max_retries: int = 12) -> bool:
    os.makedirs(os.path.dirname(dest), exist_ok=True)
    tmp = dest + ".part"
    offset = os.path.getsize(tmp) if os.path.exists(tmp) else 0
    if os.path.exists(dest) and os.path.getsize(dest) > 0:
        print(f"skip {os.path.basename(dest)}")
        return True
    for attempt in range(max_retries):
        for use_proxy in (True, False):
            try:
                req = urllib.request.Request(url, headers={
                    "User-Agent": "dscns-wheels",
                    "Range": f"bytes={offset}-",
                })
                opener = _opener(use_proxy)
                mode = "ab" if offset else "wb"
                t0 = time.time()
                with opener.open(req, timeout=120) as resp, open(tmp, mode) as f:
                    if resp.status == 200 and offset:
                        # server ignored Range -> restart
                        offset = 0
                        f.seek(0)
                        f.truncate()
                    total = int(resp.headers.get("Content-Length", 0))
                    while True:
                        chunk = resp.read(CHUNK)
                        if not chunk:
                            break
                        f.write(chunk)
                        offset += len(chunk)
                # verify: try reading back the wheel zip structure
                import zipfile

                with zipfile.ZipFile(tmp) as zf:
                    zf.testzip()
                os.replace(tmp, dest)
                print(f"ok {os.path.basename(dest)} ({offset / 1e6:.1f}MB, "
                      f"{time.time() - t0:.0f}s, {mode})")
                return True
            except Exception as e:
                print(f"  attempt {attempt + 1} ({'proxy' if use_proxy else 'direct'}): "
                      f"{type(e).__name__} {str(e)[:70]} (have {offset / 1e6:.1f}MB)")
                time.sleep(2)
    return False


def main():
    out_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                           "wheelhouse")
    os.makedirs(out_dir, exist_ok=True)
    url = sys.argv[1]
    name = os.path.basename(url).split("#")[0]
    ok = download(url, os.path.join(out_dir, name))
    print("DONE" if ok else "FAILED")


if __name__ == "__main__":
    main()
