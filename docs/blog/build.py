"""Build the two self-contained articles with the Python standard library."""

import csv
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parent
SRC = ROOT / "src"
BENCHMARKS = ROOT.parents[1] / "benchmarks"
PAGES = {
    "en": ("index.html", "content.en.html", "Qwen3.8 27B | From pipeline scheduling to CUDA kernel fusion",
           "Interactive diagrams of folded VPP prefill, KV-length-aware decode and three CUDA kernel fusions on four RTX 5090 GPUs."),
    "zh-CN": ("zh-CN.html", "content.html", "千问3.8 27B｜从流水线调度到 CUDA 算子融合",
              "在四张 RTX 5090 上，通过交互图理解折返式 VPP、Decode 的 KV 分桶与三个 CUDA 融合算子。"),
}


def json_script(name, data):
    payload = json.dumps(data, ensure_ascii=False, separators=(",", ":")).replace("<", "\\u003c")
    return f'<script id="{name}" type="application/json">{payload}</script>'


def main():
    css = (SRC / "styles.css").read_text(encoding="utf-8")
    js = (SRC / "blog.js").read_text(encoding="utf-8")
    catalog = json.loads((SRC / "messages.json").read_text(encoding="utf-8"))
    data = {}
    for name in ("prefill", "decode", "http_fusion"):
        with (BENCHMARKS / f"{name}.csv").open(encoding="utf-8", newline="") as stream:
            data[name] = list(csv.DictReader(stream))

    alternates = "\n".join(f'<link rel="alternate" hreflang="{lang}" href="{meta[0]}">'
                           for lang, meta in PAGES.items())
    for lang, (filename, source, title, description) in PAGES.items():
        body = (SRC / source).read_text(encoding="utf-8")
        messages = {key: value[lang] for key, value in catalog.items()}
        page = f"""<!doctype html>
<html lang="{lang}" data-theme="light">
<head>
<meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<meta name="color-scheme" content="light dark"><meta name="theme-color" content="#faf9f6">
<title>{title}</title><meta name="description" content="{description}">
{alternates}
<script>
(() => {{
  let saved;
  try {{ saved = localStorage.getItem('qwen-theme'); }} catch {{}}
  const systemDark = matchMedia('(prefers-color-scheme: dark)').matches;
  document.documentElement.dataset.theme = saved === 'light' || saved === 'dark' ? saved : (systemDark ? 'dark' : 'light');
}})();
</script>
<style>{css}</style>
</head><body>{body}{json_script("benchmark-data", data)}{json_script("diagram-messages", messages)}
<script>{js}</script></body></html>
"""
        with (ROOT / filename).open("w", encoding="utf-8", newline="\n") as stream:
            stream.write(page)
        print(f"Built {filename}")


if __name__ == "__main__":
    main()
