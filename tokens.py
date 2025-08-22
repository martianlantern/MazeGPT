# tokens.py
from dataclasses import dataclass
from typing import Dict, List, Tuple

PAD = "<PAD>"
BOS = "<BOS>"
SEP = "<SEP>"
EOS = "<EOS>"

# Fixed, stable vocabulary (do not change order once training starts)
# Indices are intentionally small and contiguous for efficient embedding lookup.
VOCAB_LIST = [
    PAD, BOS, SEP, EOS,  # 0..3 special
    "#", ".", "S", "G", "\n",  # 4..8 grid chars
    "U", "D", "L", "R",  # 9..12 path moves
]

# Convenience: set constants for indices
PAD_ID = 0
BOS_ID = 1
SEP_ID = 2
EOS_ID = 3

CHAR_TO_ID: Dict[str, int] = {ch: i for i, ch in enumerate(VOCAB_LIST)}
ID_TO_CHAR: Dict[int, str] = {i: ch for i, ch in enumerate(VOCAB_LIST)}

GRID_CHARS = {"#", ".", "S", "G", "\n"}
MOVE_CHARS = {"U", "D", "L", "R"}

@dataclass(frozen=True)
class EncodedExample:
    tokens: List[int]      # token ids length T (unpadded)
    loss_mask: List[int]   # 0/1 same length as tokens, 1 where we compute loss

def encode_example(maze_text: str, path_moves: str) -> EncodedExample:
    """
    Build token sequence: <BOS> + maze_text(as chars) + <SEP> + path_moves(chars) + <EOS>
    Mask loss only for positions AFTER the <SEP> (i.e., path + EOS).
    """
    toks: List[int] = [BOS_ID]
    for ch in maze_text:
        if ch not in GRID_CHARS:
            raise ValueError(f"Unexpected char in maze_text: {repr(ch)}")
        toks.append(CHAR_TO_ID[ch])
    toks.append(SEP_ID)
    for ch in path_moves:
        if ch not in MOVE_CHARS:
            raise ValueError(f"Unexpected move in path: {repr(ch)}")
        toks.append(CHAR_TO_ID[ch])
    toks.append(EOS_ID)

    mask: List[int] = [0] * len(toks)
    # Find first SEP index, mask everything strictly after it
    sep_idx = toks.index(SEP_ID)
    for i in range(sep_idx + 1, len(toks)):
        mask[i] = 1
    return EncodedExample(tokens=toks, loss_mask=mask)

def pad_for_lm(example: EncodedExample, seq_len: int, ignore_index: int = -100) -> Tuple[List[int], List[int]]:
    """
    Create padded idx and labels for LM with teacher forcing.
    labels[i] = tokens[i+1] where loss_mask[i+1]==1 else ignore_index, last label = ignore_index.
    """
    t = example.tokens
    m = example.loss_mask
    if len(t) > seq_len:
        raise ValueError(f"Sequence length {len(t)} exceeds seq_len {seq_len}")
    # labels aligned to next token
    labels: List[int] = [ignore_index] * len(t)
    for i in range(len(t) - 1):
        if m[i + 1] == 1:
            labels[i] = t[i + 1]
    # pad
    pad_len = seq_len - len(t)
    if pad_len > 0:
        t = t + [PAD_ID] * pad_len
        labels = labels + [ignore_index] * pad_len
    return t, labels

def decode_moves(ids: List[int]) -> str:
    return "".join(ID_TO_CHAR[i] for i in ids if ID_TO_CHAR.get(i, "") in MOVE_CHARS)
