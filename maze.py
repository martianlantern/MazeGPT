# maze.py
from __future__ import annotations
from typing import List, Tuple, Optional
import random
from collections import deque

# Grid representation:
# 1 = wall, 0 = open
Grid = List[List[int]]
Pos = Tuple[int, int]

DIRS = [(-1,0),(1,0),(0,-1),(0,1)]
MOVE_FROM_DELTA = {
    (-1,0): "U",
    (1,0): "D",
    (0,-1): "L",
    (0,1): "R",
}

def generate_maze(h: int, w: int, rng: random.Random) -> Grid:
    """
    Generate a perfect maze using DFS backtracker on an odd-sized grid (h,w odd recommended).
    """
    if h < 5 or w < 5:
        raise ValueError("Use h,w >= 5")
    # Ensure odd dimensions for clean walls/cells pattern
    if h % 2 == 0: h += 1
    if w % 2 == 0: w += 1

    grid: Grid = [[1 for _ in range(w)] for _ in range(h)]

    def in_bounds(r: int, c: int) -> bool:
        return 0 <= r < h and 0 <= c < w

    # Carve starting at a random odd cell
    start_r = rng.randrange(1, h, 2)
    start_c = rng.randrange(1, w, 2)
    grid[start_r][start_c] = 0

    stack = [(start_r, start_c)]
    while stack:
        r, c = stack[-1]
        # consider neighbors 2 steps away
        neighbors = []
        for dr, dc in DIRS:
            nr, nc = r + 2*dr, c + 2*dc
            if in_bounds(nr, nc) and grid[nr][nc] == 1:
                neighbors.append((nr, nc, dr, dc))
        if neighbors:
            nr, nc, dr, dc = rng.choice(neighbors)
            # carve wall between
            grid[r + dr][c + dc] = 0
            grid[nr][nc] = 0
            stack.append((nr, nc))
        else:
            stack.pop()

    return grid

def all_open_cells(grid: Grid) -> List[Pos]:
    cells = []
    for r in range(len(grid)):
        for c in range(len(grid[0])):
            if grid[r][c] == 0:
                cells.append((r, c))
    return cells

def bfs_dists(grid: Grid, start: Pos) -> Tuple[List[List[int]], Pos]:
    h, w = len(grid), len(grid[0])
    dist = [[-1]*w for _ in range(h)]
    q = deque()
    sr, sc = start
    dist[sr][sc] = 0
    q.append((sr, sc))
    farthest = start
    while q:
        r, c = q.popleft()
        if dist[r][c] > dist[farthest[0]][farthest[1]]:
            farthest = (r, c)
        for dr, dc in DIRS:
            nr, nc = r+dr, c+dc
            if 0 <= nr < h and 0 <= nc < w and grid[nr][nc] == 0 and dist[nr][nc] == -1:
                dist[nr][nc] = dist[r][c] + 1
                q.append((nr, nc))
    return dist, farthest

def pick_start_goal_far_apart(grid: Grid, rng: random.Random) -> Tuple[Pos, Pos]:
    cells = all_open_cells(grid)
    if not cells:
        raise RuntimeError("No open cells to place S/G")
    # Heuristic: farthest-of-farthest
    a = rng.choice(cells)
    _, b = bfs_dists(grid, a)
    dist_from_b, c = bfs_dists(grid, b)
    return b, c  # S, G

def solve_shortest_path(grid: Grid, S: Pos, G: Pos) -> List[str]:
    h, w = len(grid), len(grid[0])
    sr, sc = S
    gr, gc = G
    dist = [[-1]*w for _ in range(h)]
    parent: List[List[Optional[Pos]]] = [[None]*w for _ in range(h)]
    q = deque()
    dist[sr][sc] = 0
    q.append((sr, sc))
    while q:
        r, c = q.popleft()
        if (r, c) == (gr, gc):
            break
        for dr, dc in DIRS:
            nr, nc = r+dr, c+dc
            if 0 <= nr < h and 0 <= nc < w and grid[nr][nc] == 0 and dist[nr][nc] == -1:
                dist[nr][nc] = dist[r][c] + 1
                parent[nr][nc] = (r, c)
                q.append((nr, nc))
    if dist[gr][gc] == -1:
        raise RuntimeError("No path found from S to G (shouldn't happen in a perfect maze).")
    # reconstruct
    path_moves: List[str] = []
    r, c = gr, gc
    while (r, c) != (sr, sc):
        pr, pc = parent[r][c]  # type: ignore
        dr, dc = r - pr, c - pc  # type: ignore
        path_moves.append(MOVE_FROM_DELTA[(dr, dc)])
        r, c = pr, pc  # type: ignore
    path_moves.reverse()
    return path_moves

def render_ascii(grid: Grid, S: Pos, G: Pos) -> str:
    h, w = len(grid), len(grid[0])
    lines: List[str] = []
    for r in range(h):
        row_chars = []
        for c in range(w):
            if (r, c) == S:
                row_chars.append("S")
            elif (r, c) == G:
                row_chars.append("G")
            else:
                row_chars.append("." if grid[r][c] == 0 else "#")
        lines.append("".join(row_chars))
    return "\n".join(lines) + "\n"
