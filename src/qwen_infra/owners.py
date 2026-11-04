"""Persistent compiler operand identity for opaque attention owners."""

import hashlib

OWNER_ABI = "qwen_sm120.attention_fusion.named_owner.v1"


def stable_owner_key(name):
    """Negative signed-64 key derived from the ABI and full model layer name."""
    if not isinstance(name, str) or not name:
        raise ValueError("Expected a full attention layer name")
    digest = hashlib.sha256((OWNER_ABI + "\0" + name).encode("utf-8")).digest()
    return -1 - (int.from_bytes(digest[:8], "big") & ((1 << 63) - 1))
