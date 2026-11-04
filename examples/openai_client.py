"""Use native vLLM's OpenAI-compatible endpoint without another client SDK."""

import argparse
import json
from urllib.request import Request, urlopen

parser = argparse.ArgumentParser()
parser.add_argument("--base-url", default="http://127.0.0.1:8000")
parser.add_argument("--model", required=True)
parser.add_argument("--prompt", default="Explain virtual pipeline parallelism in one paragraph.")
args = parser.parse_args()
body = json.dumps(dict(model=args.model, prompt=args.prompt, max_tokens=64, temperature=0)).encode()
request = Request(
    args.base_url.rstrip("/") + "/v1/completions", data=body, headers={"Content-Type": "application/json"}
)
with urlopen(request, timeout=120) as response:
    print(json.dumps(json.load(response), indent=2))
