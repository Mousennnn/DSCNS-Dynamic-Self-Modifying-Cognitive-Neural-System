"""Dependency resolver v2 + resumable wheel downloader for offline install.

Improvements over v1:
- checks each wheel's ``Requires-Python`` against Python 3.8,
- uses the PER-VERSION JSON metadata (original requirements; the 'latest'
  JSON on PyPI has been metadata-updated with py38-incompatible pins),
- skips torch (already installed as 1.13.1+cu117 in the target env).
"""
from __future__ import annotations

import json
import os
import re
import sys
import time
import urllib.request
from collections import deque

from packaging.markers import Marker
from packaging.requirements import Requirement
from packaging.utils import canonicalize_name
from packaging.version import Version, InvalidVersion

PY = ("cp38", "py3", "py38", "py2.py3", "abi3")
PLAT = ("win_amd64", "any")
CHUNK = 1 << 20
PYTHON_VERSION = "3.8.16"

TOP_LEVEL = [
    "transformers==4.45.2", "datasets==2.21.0", "peft==0.12.0",
    "accelerate==0.34.2", "huggingface_hub==0.25.2", "tokenizers==0.20.1",
    "tqdm", "matplotlib", "scikit-learn", "requests", "pyyaml",
]
SKIP = {"torch"}  # keep installed torch 1.13.1+cu117


def _opener(use_proxy: bool):
    if use_proxy:
        return urllib.request.build_opener(
            urllib.request.ProxyHandler(urllib.request.getproxies()))
    return urllib.request.build_opener(urllib.request.ProxyHandler({}))


def fetch_json(url: str, dest: str, retries: int = 10) -> dict:
    os.makedirs(os.path.dirname(dest), exist_ok=True)
    tmp = dest + ".part"
    offset = os.path.getsize(tmp) if os.path.exists(tmp) else 0
    for attempt in range(retries):
        for use_proxy in (True, False):
            try:
                req = urllib.request.Request(url, headers={
                    "User-Agent": "dscns-resolver",
                    "Range": f"bytes={offset}-",
                })
                with _opener(use_proxy).open(req, timeout=120) as resp, \
                        open(tmp, "ab" if offset else "wb") as f:
                    if resp.status == 200 and offset:
                        offset = 0
                        f.seek(0)
                        f.truncate()
                    while True:
                        chunk = resp.read(CHUNK)
                        if not chunk:
                            break
                        f.write(chunk)
                        offset += len(chunk)
                with open(tmp, "r", encoding="utf-8") as f:
                    data = json.load(f)
                os.replace(tmp, dest)
                return data
            except Exception as e:
                print(f"  fetch {url.split('/')[-1]} attempt {attempt + 1} "
                      f"({'proxy' if use_proxy else 'direct'}): "
                      f"{type(e).__name__} {str(e)[:60]}")
                time.sleep(1.5)
    raise RuntimeError(f"cannot fetch {url}")


def wheel_ok(filename: str, requires_python: str = None) -> bool:
    m = re.match(r"^([A-Za-z0-9_.\-]+)-([0-9][A-Za-z0-9.!+\-]*)-([^-]+)-([^-]+)-([^-]+)\.whl$", filename)
    if not m:
        return False
    py, abi, plat = m.group(3), m.group(4), m.group(5)
    if plat not in PLAT:
        return False
    if not any(p in py.split(".") for p in PY):
        return False
    if abi not in ("none", "cp38", "abi3"):
        return False
    if requires_python:
        from packaging.specifiers import SpecifierSet

        try:
            if not SpecifierSet(requires_python).contains(PYTHON_VERSION):
                return False
        except Exception:
            pass
    return True


def pick_version(data: dict, spec: Requirement):
    """Newest version with a compatible wheel satisfying the spec."""
    releases = data.get("releases", {})
    cands = []
    for ver, files in releases.items():
        try:
            v = Version(ver)
        except InvalidVersion:
            continue
        if not spec.specifier.contains(v, prereleases=True):
            continue
        for f in files:
            if f.get("packagetype") == "bdist_wheel" and wheel_ok(
                    f["filename"], f.get("requires_python")):
                cands.append((v, ver, f))
                break
    if not cands:
        return None
    cands.sort(key=lambda x: x[0], reverse=True)
    return cands[0]


def main():
    root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    wh = os.path.join(root, "wheelhouse")
    meta_dir = os.path.join(wh, "meta")
    os.makedirs(wh, exist_ok=True)
    os.makedirs(meta_dir, exist_ok=True)

    queue = deque()
    for req in TOP_LEVEL:
        queue.append((Requirement(req), frozenset()))
    done = {}
    pending_meta = {}  # name -> per-version json path

    while queue:
        req, req_extras = queue.popleft()
        name = canonicalize_name(req.name)
        if name in done or name in SKIP:
            continue
        needed_extras = set(req.extras) | req_extras
        meta_path = os.path.join(meta_dir, name + ".json")
        if not os.path.exists(meta_path):
            fetch_json(f"https://pypi.org/pypi/{req.name}/json", meta_path)
        data = json.load(open(meta_path, encoding="utf-8"))

        picked = pick_version(data, req)
        if picked is None:
            print(f"WARN: no py38-compatible wheel for {req.name} {req.specifier}")
            continue
        v, ver, f = picked
        done[name] = (str(ver), f["url"], f.get("sha256", ""))
        print(f"resolved {req.name}=={ver} <- {f['filename']}")

        # per-version metadata (original requirements)
        ver_meta = os.path.join(meta_dir, f"{name}-{ver}.json")
        if not os.path.exists(ver_meta):
            fetch_json(f"https://pypi.org/pypi/{req.name}/{ver}/json", ver_meta)
        vdata = json.load(open(ver_meta, encoding="utf-8"))
        requires = vdata.get("info", {}).get("requires_dist") or []
        for rd in requires:
            try:
                dep = Requirement(rd)
            except Exception:
                continue
            if dep.marker is not None:
                ok = False
                for ex in (needed_extras or {""}):
                    try:
                        if dep.marker.evaluate({"extra": ex}):
                            ok = True
                            break
                    except Exception:
                        if dep.marker.evaluate():
                            ok = True
                            break
                if not ok:
                    continue
            queue.append((dep, frozenset(needed_extras)))

    print(f"\n{len(done)} packages to download")
    ok_all = True
    for name, (ver, url, sha) in sorted(done.items()):
        fn = url.split("/")[-1].split("#")[0]
        dest = os.path.join(wh, fn)
        if os.path.exists(dest) and os.path.getsize(dest) > 0:
            print(f"skip {fn}")
        else:
            if not _download_resume(url, dest):
                ok_all = False
                print(f"FAILED {fn}")
    lock = os.path.join(wh, "requirements-lock.txt")
    with open(lock, "w", encoding="utf-8") as f:
        for name, (ver, url, sha) in sorted(done.items()):
            fn = url.split("/")[-1].split("#")[0]
            f.write(f"{name}=={ver} --hash=sha256:{sha}  # {fn}\n")
    print(f"\nlock written: {lock}")
    print("ALL DOWNLOADS OK" if ok_all else "SOME DOWNLOADS FAILED")


def _download_resume(url: str, dest: str, max_retries: int = 15) -> bool:
    tmp = dest + ".part"
    offset = os.path.getsize(tmp) if os.path.exists(tmp) else 0
    for attempt in range(max_retries):
        for use_proxy in (True, False):
            try:
                req = urllib.request.Request(url, headers={
                    "User-Agent": "dscns-wheels",
                    "Range": f"bytes={offset}-",
                })
                t0 = time.time()
                with _opener(use_proxy).open(req, timeout=180) as resp, \
                        open(tmp, "ab" if offset else "wb") as f:
                    if resp.status == 200 and offset:
                        offset = 0
                        f.seek(0)
                        f.truncate()
                    while True:
                        chunk = resp.read(CHUNK)
                        if not chunk:
                            break
                        f.write(chunk)
                        offset += len(chunk)
                import zipfile

                with zipfile.ZipFile(tmp) as zf:
                    zf.testzip()
                os.replace(tmp, dest)
                print(f"ok {os.path.basename(dest)} ({offset / 1e6:.1f}MB, "
                      f"{time.time() - t0:.0f}s)")
                return True
            except Exception as e:
                print(f"  dl {os.path.basename(dest)[:30]} attempt {attempt + 1} "
                      f"({'proxy' if use_proxy else 'direct'}): "
                      f"{type(e).__name__} {str(e)[:50]} (have {offset / 1e6:.1f}MB)")
                time.sleep(1.5)
    return False


if __name__ == "__main__":
    main()
