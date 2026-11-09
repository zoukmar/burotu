"""Who used the live demo: a summary of the site's access logs (read on stdin).

Reads Caddy JSON access logs and/or uvicorn access lines, keeps the demo calls
(POST /api/demo/<sample> runs the models, including Muse Glimmer on the GPU), and prints
per day and per visitor (IP truncated to its first two numbers, browser family).
"""

from __future__ import annotations

import collections
import datetime as dt
import json
import re
import sys

UVICORN = re.compile(r'(?P<ip>[\d.:a-f]+):\d+ - "(?P<method>[A-Z]+) (?P<uri>\S+) HTTP/[\d.]+" (?P<status>\d+)')


def short_ip(ip: str) -> str:
    parts = ip.split(".")
    return ".".join(parts[:2]) + ".x.x" if len(parts) == 4 else ip.split(":")[0] + ":…"


def family(ua: str) -> str:
    ua = ua.lower()
    for key, name in [
        ("headlesschrome", "headless Chrome (script)"),
        ("curl", "curl (script)"),
        ("python", "Python (script)"),
        ("bot", "bot"),
        ("edg/", "Edge"),
        ("chrome", "Chrome"),
        ("firefox", "Firefox"),
        ("safari", "Safari"),
    ]:
        if key in ua:
            return name
    return ua[:30] or "?"


def main() -> None:
    runs, warms, pages = [], collections.Counter(), collections.Counter()
    for line in sys.stdin:
        line = line.strip()
        if line.startswith("{"):
            try:
                d = json.loads(line)
            except ValueError:
                continue
            r = d.get("request", {})
            uri, method = r.get("uri", ""), r.get("method", "")
            ip = (r.get("headers", {}).get("Cf-Connecting-Ip") or [r.get("client_ip") or r.get("remote_ip", "?")])[0]
            ua = (r.get("headers", {}).get("User-Agent") or [""])[0]
            when = dt.datetime.fromtimestamp(d.get("ts", 0), dt.UTC)
            status = d.get("status", 0)
        else:
            m = UVICORN.search(line)
            if not m:
                continue
            uri, method, ip, ua, status = m["uri"], m["method"], m["ip"], "", int(m["status"])
            when = None
        if method == "POST" and uri.startswith("/api/demo/warm"):
            warms[short_ip(ip)] += 1
        elif method == "POST" and uri.startswith("/api/demo/"):
            runs.append((when, short_ip(ip), family(ua), uri.split("/")[3].split("?")[0][-2:], status))
        elif method == "GET" and re.fullmatch(r"/(en|fr|es)/", uri.split("?")[0]):
            pages[(when.date().isoformat() if when else "?")] += 1
    print(f"Home page views per day: {dict(sorted(pages.items()))}")
    print(f"Page openings that woke the GPU, per visitor: {dict(warms.most_common())}")
    print(f"\nDemo runs (each one reads a lab file with every model, Glimmer on the GPU): {len(runs)}")
    by_visitor = collections.defaultdict(list)
    for when, ip, fam, sample, status in runs:
        by_visitor[(ip, fam)].append((when, sample, status))
    for (ip, fam), items in sorted(by_visitor.items(), key=lambda kv: -len(kv[1])):
        times = [w for w, _, _ in items if w]
        span = f"{min(times):%d/%m %H:%M} → {max(times):%d/%m %H:%M} UTC" if times else "?"
        samples = ",".join(sorted({s for _, s, _ in items}))
        print(f"  {ip:<14} {fam:<26} {len(items):>3} runs  samples {samples:<20} {span}")


if __name__ == "__main__":
    main()
