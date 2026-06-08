#!/usr/bin/env python3
"""Concurrency tester for the PaddleOCR-VL HTTP server.

Sends the same image (`sample/contract.jpg`) concurrently to the server's
`/process` endpoint with ``format=html`` and compares the returned HTML to
`output/contract.html`.

Defaults mirror `scripts/server.py` (host=0.0.0.0, port=5000). Concurrency
default is 8. The script uses only the Python standard library.
"""
from __future__ import annotations

import argparse
import base64
import json
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Any, Dict, Optional, Tuple
import urllib.request
import urllib.error


def build_arg_parser() -> argparse.ArgumentParser:
    root = Path(__file__).resolve().parent.parent
    p = argparse.ArgumentParser(
        description="Test PaddleOCR-VL HTTP server",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument("--host", default="0.0.0.0", help="Server bind host (default matches server.py)")
    p.add_argument("--port", type=int, default=5000, help="Server bind port")
    p.add_argument("--concurrency", type=int, default=8, help="Number of concurrent requests to send")
    p.add_argument(
        "--image",
        default=str(root / "sample" / "contract.jpg"),
        help="Path to the test image",
    )
    p.add_argument(
        "--expected",
        default=None,
        help="Path to expected HTML output for comparison (optional)",
    )
    p.add_argument(
        "--output",
        default=None,
        help="Path to output the result HTML if all workers are consistent (optional)",
    )
    p.add_argument("--health-timeout", type=int, default=120, help="Seconds to wait for /health ready")
    p.add_argument("--request-timeout", type=int, default=300, help="Per-request timeout seconds")
    return p


def http_get(url: str, timeout: int = 5) -> Tuple[Optional[int], Optional[dict], str]:
    req = urllib.request.Request(url, method="GET")
    try:
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            body = resp.read().decode("utf-8", errors="replace")
            headers = {k: v for k, v in resp.getheaders()}
            return resp.getcode(), headers, body
    except urllib.error.HTTPError as e:
        body = e.read().decode("utf-8", errors="replace")
        headers = {k: v for k, v in e.headers.items()} if e.headers else {}
        return e.code, headers, body
    except Exception as exc:
        return None, None, str(exc)


def http_post_json(url: str, obj: Any, timeout: int = 30) -> Tuple[Optional[int], Optional[dict], str]:
    data = json.dumps(obj).encode("utf-8")
    req = urllib.request.Request(url, data=data, headers={"Content-Type": "application/json"}, method="POST")
    try:
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            body = resp.read().decode("utf-8", errors="replace")
            headers = {k: v for k, v in resp.getheaders()}
            return resp.getcode(), headers, body
    except urllib.error.HTTPError as e:
        body = e.read().decode("utf-8", errors="replace")
        headers = {k: v for k, v in e.headers.items()} if e.headers else {}
        return e.code, headers, body
    except Exception as exc:
        return None, None, str(exc)


def wait_for_health(url: str, timeout: int = 120, interval: float = 1.0) -> bool:
    start = time.time()
    while time.time() - start < timeout:
        code, _, body = http_get(url, timeout=5)
        if code == 200:
            print("/health OK")
            return True
        print(f"Waiting for server ready (status={code})... {int(time.time()-start)}s elapsed", end="\r")
        time.sleep(interval)
    print()
    return False


def load_file_text(path: str) -> str:
    p = Path(path)
    return p.read_text(encoding="utf-8")


def send_single(process_url: str, payload: dict, timeout: int) -> Dict[str, Any]:
    code, headers, body = http_post_json(process_url, payload, timeout=timeout)
    result: Dict[str, Any] = {"status": code, "headers": headers, "body": body}
    return result


def main(argv: Optional[list] = None) -> int:
    parser = build_arg_parser()
    args = parser.parse_args(argv)

    # If server is bound to 0.0.0.0, use localhost for client connections.
    host_connect = "127.0.0.1" if args.host in ("0.0.0.0", "::", "::0") else args.host
    base = f"http://{host_connect}:{args.port}"
    health_url = f"{base}/health"
    process_url = f"{base}/process"

    print(f"Server: {base}")
    print(f"Test image: {args.image}")
    if args.expected:
        print(f"Expected HTML: {args.expected}")
    if args.output:
        print(f"Output file: {args.output}")
    print(f"Concurrency: {args.concurrency}")

    if not Path(args.image).exists():
        print(f"Test image not found: {args.image}")
        return 2
    if args.expected and not Path(args.expected).exists():
        print(f"Expected HTML not found: {args.expected}")
        return 2

    if not wait_for_health(health_url, timeout=args.health_timeout):
        print(f"Server did not become ready within {args.health_timeout} seconds")
        return 3

    # prepare payload
    img_b = Path(args.image).read_bytes()
    b64 = base64.b64encode(img_b).decode("ascii")
    payload = {"image": f"data:image/jpeg;base64,{b64}", "format": "html", "orientation": True}

    print("Sending concurrent requests...")
    results = []
    with ThreadPoolExecutor(max_workers=args.concurrency) as ex:
        futures = [ex.submit(send_single, process_url, payload, args.request_timeout) for _ in range(args.concurrency)]
        for fut in as_completed(futures):
            res = fut.result()
            results.append(res)

    # Step 1: Check if all worker results are consistent
    print("Checking consistency of all worker results...")
    if not results:
        print("No results received")
        return 4

    first_body = results[0].get("body", "")
    first_status = results[0].get("status")

    inconsistent_results = []
    for i, r in enumerate(results, 1):
        status = r.get("status")
        body = r.get("body", "")
        if status != first_status or body != first_body:
            inconsistent_results.append((i, r))

    if inconsistent_results:
        print(f"ERROR: Worker results are inconsistent! Found {len(inconsistent_results)} mismatches")
        print(f"\nFirst result (status={first_status}, body length={len(first_body)}):")
        print(f"  Body preview: {first_body[:200]}")
        print(f"\nInconsistent results:")
        for i, r in inconsistent_results:
            status = r.get("status")
            body = r.get("body", "")
            print(f"  [Worker {i}] status={status}, body length={len(body)}")
            print(f"    Body preview: {body[:200]}")
        return 4

    print(f"✓ All {len(results)} workers returned consistent results")

    # Step 2: Process the consistent result
    standard_result = results[0]
    status = standard_result.get("status")
    headers = standard_result.get("headers") or {}
    body = (standard_result.get("body") or "").strip()

    # Check HTTP status and Content-Type
    if status != 200:
        body_preview = body[:200]
        print(f"ERROR: HTTP {status} - {body_preview}")
        return 4

    ctype = headers.get("Content-Type", headers.get("content-type", ""))
    if "html" not in (ctype or ""):
        print(f"ERROR: wrong Content-Type: {ctype}")
        return 4

    # Step 3: Output to file if requested
    if args.output:
        output_path = Path(args.output)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text(body, encoding="utf-8")
        print(f"✓ Result written to {args.output}")

    # Step 4: Compare with expected HTML if provided
    if args.expected:
        expected_html = load_file_text(args.expected).strip()
        if body == expected_html:
            print("✓ Result matches expected HTML")
            return 0
        else:
            print("ERROR: Result differs from expected HTML")
            return 4

    print("✓ All checks passed (no expected HTML to compare)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
