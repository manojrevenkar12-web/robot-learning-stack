"""Word-level instruction tokenizer with a fixed, versioned vocabulary.

No pretrained language model is reachable offline in this environment, so
instructions are embedded with a learned token table. The vocabulary includes
every colour the simulator knows, including colours that never appear in the
training data, so a fine-tuning run can teach a new word without changing the
model's shapes.
"""

from __future__ import annotations

import re

from rls.sim.scene import COLORS

SPECIAL = ["<pad>", "<unk>"]
WORDS = sorted({"put", "the", "cube", "in", "bin", "stack", "on", "pick", "place", "push", "to", "and", "then", *COLORS})
VOCAB = SPECIAL + WORDS
INDEX = {w: i for i, w in enumerate(VOCAB)}
MAX_TOKENS = 10


def tokenize(text: str, max_len: int = MAX_TOKENS) -> list[int]:
    ids = [INDEX.get(w, INDEX["<unk>"]) for w in re.findall(r"[a-z]+", text.lower())][:max_len]
    return ids + [0] * (max_len - len(ids))
