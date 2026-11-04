"""Kernel ABI arithmetic bounds, distinct from measured context coverage."""

TILE = 256
PAGE_SIZE = 32
MAX_TABLE_WIDTH = ((1 << 31) - 1 - (TILE - 1)) // PAGE_SIZE
MAX_SEQUENCE_LENGTH = MAX_TABLE_WIDTH * PAGE_SIZE
