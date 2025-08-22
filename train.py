# train.py
from __future__ import annotations
import argparse
import json
import math
import os
import time
from pathlib import Path
from typing import List, Tuple

import torch
import torch.nn.functional as F
import torch.optim as optim

from tokens import (
    PAD_ID, BOS_ID, SEP_ID, EOS_ID, VOCAB_LIST, encode_example, pad_for_lm, decode_moves
)
from maze import generate_maze, pick_start_goal_far_apart, solve_shortest_path, render_ascii
from model import TinyGPT
from utils import set_seed, select_device, save_json, append_jsonl, start_http_server, copy_viewer_into_run, run_id_from_time

IGNORE_INDEX = -100

def build_example(h: int, w: int, rng, seq_len: int) -> Tuple[List[int], List[int], str, str]:
    """
    Returns (idx_ids, label_ids, maze_text, path_text) for one maze.
    """
    grid = generate_maze(h, w, rng)
    S, G = pick_start_goal_far_apart(grid, rng)
    path_moves = "".join(solve_shortest_path(grid, S, G))
    maze_text = render_ascii(grid, S, G)
    enc = encode_example(maze_text, path_moves)
    idx_ids, labels = pad_for_lm(enc, seq_len, IGNORE_INDEX)
    return idx_ids, labels, maze_text, path_moves

def batchify(B: int, h: int, w: int, rng, seq_len: int, device: torch.device):
    idx_list: List[List[int]] = []
    label_list: List[List[int]] = []
    mazes: List[str] = []
    paths: List[str] = []
    for _ in range(B):
        idx_ids, labels, maze_text, path_text = build_example(h, w, rng, seq_len)
        idx_list.append(idx_ids)
        label_list.append(labels)
        mazes.append(maze_text)
        paths.append(path_text)
    idx = torch.tensor(idx_list, dtype=torch.long, device=device)
    labels = torch.tensor(label_list, dtype=torch.long, device=device)
    return idx, labels, mazes, paths

@torch.no_grad()
def evaluate(model: TinyGPT, steps: int, h: int, w: int, rng, seq_len: int, device: torch.device, batch_size: int) -> float:
    model.eval()
    losses = []
    for _ in range(steps):
        idx, labels, _, _ = batchify(batch_size, h, w, rng, seq_len, device)
        logits = model(idx)
        loss = F.cross_entropy(logits.view(-1, logits.size(-1)), labels.view(-1), ignore_index=IGNORE_INDEX)
        losses.append(loss.item())
    return sum(losses) / max(1, len(losses))

@torch.no_grad()
def sample_one(model: TinyGPT, h: int, w: int, rng, seq_len: int, device: torch.device,
               temperature: float, top_k: int):
    model.eval()
    # Create the prompt tokens: <BOS> + maze + <SEP>
    grid = generate_maze(h, w, rng)
    S, G = pick_start_goal_far_apart(grid, rng)
    target_path = "".join(solve_shortest_path(grid, S, G))
    maze_text = render_ascii(grid, S, G)

    # Build initial tokens without labels/padding
    from tokens import CHAR_TO_ID
    toks: List[int] = [BOS_ID]
    for ch in maze_text:
        toks.append(CHAR_TO_ID[ch])
    toks.append(SEP_ID)
    idx = torch.tensor([toks], dtype=torch.long, device=device)
    # Generate up to len(target_path)+10 (headroom)
    max_new = min(seq_len - idx.size(1), len(target_path) + 10)
    out = model.generate(idx, max_new_tokens=max_new, temperature=temperature, top_k=top_k)
    gen = out[0].tolist()
    # Extract generated segment after SEP until EOS
    try:
        sep_pos = gen.index(SEP_ID)
    except ValueError:
        sep_pos = len(gen) - 1
    after = gen[sep_pos + 1:]
    if EOS_ID in after:
        eos_i = after.index(EOS_ID)
        after = after[:eos_i]
    pred_path = decode_moves(after)
    return maze_text, target_path, pred_path

def main():
    p = argparse.ArgumentParser()
    # Data/maze
    p.add_argument("--maze-h", type=int, default=15)
    p.add_argument("--maze-w", type=int, default=15)
    p.add_argument("--seq-len", type=int, default=1024)
    p.add_argument("--batch-size", type=int, default=32)
    # Train
    p.add_argument("--steps", type=int, default=20000)
    p.add_argument("--eval-interval", type=int, default=200)
    p.add_argument("--eval-batches", type=int, default=10)
    p.add_argument("--sample-interval", type=int, default=200)
    p.add_argument("--save-interval", type=int, default=2000)
    p.add_argument("--lr", type=float, default=3e-4)
    p.add_argument("--warmup", type=int, default=500)
    p.add_argument("--weight-decay", type=float, default=0.1)
    p.add_argument("--grad-clip", type=float, default=1.0)
    p.add_argument("--amp", type=str, default="bf16", choices=["bf16", "fp16", "off"])
    # Model
    p.add_argument("--d-model", type=int, default=256)
    p.add_argument("--n-layer", type=int, default=4)
    p.add_argument("--n-head", type=int, default=4)
    p.add_argument("--d-ff", type=int, default=1024)
    # Runtime
    p.add_argument("--device", type=str, default="auto", choices=["auto", "cuda", "cpu"])
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--run-name", type=str, default="")
    p.add_argument("--runs-dir", type=str, default="runs")
    p.add_argument("--port", type=int, default=7860)
    p.add_argument("--viewer", type=str, default="on", choices=["on", "off"])
    # Generation params
    p.add_argument("--temperature", type=float, default=0.0)
    p.add_argument("--top-k", type=int, default=0)
    # Checkpoint
    p.add_argument("--resume", type=str, default="")  # path to ckpt
    args = p.parse_args()

    set_seed(args.seed)
    device = select_device(args.device)

    run_id = run_id_from_time(args.run_name or None)
    runs_root = Path(args.runs_dir)
    run_dir = runs_root / run_id
    (run_dir / "samples").mkdir(parents=True, exist_ok=True)
    (run_dir / "ckpt").mkdir(parents=True, exist_ok=True)

    # Prepare viewer
    if args.viewer == "on":
        copy_viewer_into_run(run_dir)
        start_http_server(run_dir, port=args.port, host="0.0.0.0")

    # Save run metadata
    meta = {
        "args": vars(args),
        "device": str(device),
        "vocab": VOCAB_LIST,
        "start_time": time.strftime("%Y-%m-%d %H:%M:%S"),
    }
    save_json(run_dir / "run_meta.json", meta)

    # Model
    model = TinyGPT(
        vocab_size=len(VOCAB_LIST),
        d_model=args.d_model,
        n_layer=args.n_layer,
        n_head=args.n_head,
        d_ff=args.d_ff,
        max_seq=args.seq_len,
        dropout=0.1,
        tie_embeddings=False,
    ).to(device)

    optimizer = optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay, betas=(0.9, 0.95))
    scaler = None
    use_amp = False
    amp_dtype = None
    if device.type == "cuda" and args.amp != "off":
        if args.amp == "bf16":
            amp_dtype = torch.bfloat16
            use_amp = True
        elif args.amp == "fp16":
            amp_dtype = torch.float16
            use_amp = True
            scaler = torch.cuda.amp.GradScaler(enabled=True)

    start_step = 0
    if args.resume:
        ckpt_path = Path(args.resume)
        if ckpt_path.exists():
            ckpt = torch.load(ckpt_path, map_location=device)
            model.load_state_dict(ckpt["model"])
            optimizer.load_state_dict(ckpt["optim"])
            start_step = ckpt.get("step", 0)
            print(f"[ckpt] Resumed from {ckpt_path} at step {start_step}")
        else:
            print(f"[ckpt] Resume path not found: {ckpt_path}")

    rng = __import__("random").Random(args.seed + 1)  # independent RNG

    model.train()
    t0 = time.time()
    for step in range(start_step, args.steps):
        # LR schedule
        lr = args.lr if args.steps <= 0 else args.lr * 0.5 * (1 + math.cos(math.pi * min(1.0, step / max(1, args.steps))))
        for g in optimizer.param_groups:
            g["lr"] = lr

        idx, labels, _, _ = batchify(args.batch_size, args.maze_h, args.maze_w, rng, args.seq_len, device)
        if use_amp and amp_dtype is not None and scaler is not None:
            with torch.cuda.amp.autocast(dtype=amp_dtype):
                logits = model(idx)
                loss = F.cross_entropy(logits.view(-1, logits.size(-1)), labels.view(-1), ignore_index=IGNORE_INDEX)
            optimizer.zero_grad(set_to_none=True)
            scaler.scale(loss).backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)
            scaler.step(optimizer)
            scaler.update()
        else:
            if use_amp and amp_dtype is not None:
                with torch.autocast(device_type=device.type, dtype=amp_dtype):
                    logits = model(idx)
                    loss = F.cross_entropy(logits.view(-1, logits.size(-1)), labels.view(-1), ignore_index=IGNORE_INDEX)
            else:
                logits = model(idx)
                loss = F.cross_entropy(logits.view(-1, logits.size(-1)), labels.view(-1), ignore_index=IGNORE_INDEX)
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)
            optimizer.step()

        if (step % 10) == 0 or step == args.steps - 1:
            append_jsonl(run_dir / "metrics.jsonl", {"step": step, "train_loss": float(loss.item())})

        if (step + 1) % args.eval_interval == 0:
            eval_loss = evaluate(model, args.eval_batches, args.maze_h, args.maze_w, rng, args.seq_len, device, args.batch_size)
            append_jsonl(run_dir / "metrics.jsonl", {"step": step + 1, "eval_loss": float(eval_loss)})
            print(f"[eval] step {step+1} eval_loss={eval_loss:.4f}")

        if (step + 1) % args.sample_interval == 0:
            maze_text, target_path, pred_path = sample_one(
                model, args.maze_h, args.maze_w, rng, args.seq_len, device,
                temperature=args.temperature, top_k=args.top_k
            )
            sample = {
                "step": step + 1,
                "maze_text": maze_text,
                "target_path": target_path,
                "pred_path": pred_path,
                "time": time.strftime("%Y-%m-%d %H:%M:%S"),
            }
            samp_path = run_dir / "samples" / f"step_{step+1:06d}.json"
            save_json(samp_path, sample)
            save_json(run_dir / "samples" / "latest.json", sample)
            # If viewer exists in run root, ensure index.html already copied (done at start)

        if (step + 1) % args.save_interval == 0 or (step + 1) == args.steps:
            ckpt = {
                "step": step + 1,
                "model": model.state_dict(),
                "optim": optimizer.state_dict(),
                "config": {
                    "vocab_size": len(VOCAB_LIST),
                    "d_model": args.d_model,
                    "n_layer": args.n_layer,
                    "n_head": args.n_head,
                    "d_ff": args.d_ff,
                    "max_seq": args.seq_len,
                },
            }
            ckpt_path = run_dir / "ckpt" / f"step_{step+1:06d}.pt"
            torch.save(ckpt, ckpt_path)
            print(f"[ckpt] Saved {ckpt_path}")

        if (step % 50) == 0:
            dt = time.time() - t0
            t0 = time.time()
            print(f"[train] step {step} loss={loss.item():.4f} lr={lr:.2e} dt={dt:.2f}s")

    print("[done] training complete")

if __name__ == "__main__":
    main()