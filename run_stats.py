from __future__ import annotations

import os
import random
import statistics
import subprocess
import sys
import threading
import time
import json
from collections import defaultdict, deque
from dataclasses import dataclass, field, asdict
from pathlib import Path
from queue import Empty, Queue
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

# ---------------------------------------------------------------------------
# Frodo tester (Python version) - Enhanced with failure tracking
#
# Changes from original:
# - Saves failed test cases categorized by error type and code variation
# - Creates detailed failure reports with reproducible test data
# - Enables targeted debugging and regression testing
# ---------------------------------------------------------------------------

# Board constants
SIZE = 13
DIRS = [(0, 1), (1, 0), (0, -1), (-1, 0)]
SENTINEL = object()

# Paths to Python solutions
ROOT = Path(__file__).resolve().parent
PYTHON_BIN = Path(sys.executable)
ASTAR_SCRIPT = Path(os.environ.get("ASTAR_SCRIPT", str(ROOT / "ee.py")))
BACKTRACK_SCRIPT = Path(os.environ.get("BACKTRACK_SCRIPT", str(ROOT / "backtracking.py")))

# Simulation parameters
DEFAULT_MAPS_PER_RUN = 5
COMMAND_TIMEOUT_SEC = 5.0
PROCESS_TIMEOUT_SEC = 60.0
DEFAULT_SEED = None


@dataclass(frozen=True)
class Enemy:
    kind: str
    x: int
    y: int

    @property
    def pos(self) -> Tuple[int, int]:
        return (self.x, self.y)


@dataclass(frozen=True)
class MapDefinition:
    g_pos: Tuple[int, int]
    m_pos: Tuple[int, int]
    c_pos: Tuple[int, int]
    enemies: Tuple[Enemy, ...]

    def enemy_positions(self) -> set[Tuple[int, int]]:
        return {enemy.pos for enemy in self.enemies}

    def to_dict(self) -> Dict:
        """Convert to dictionary for JSON serialization"""
        return {
            "g_pos": self.g_pos,
            "m_pos": self.m_pos,
            "c_pos": self.c_pos,
            "enemies": [{"kind": e.kind, "x": e.x, "y": e.y} for e in self.enemies],
        }

    @classmethod
    def from_dict(cls, data: Dict) -> 'MapDefinition':
        """Create MapDefinition from dictionary"""
        enemies = tuple(Enemy(e["kind"], e["x"], e["y"]) for e in data["enemies"])
        return cls(
            g_pos=tuple(data["g_pos"]),
            m_pos=tuple(data["m_pos"]),
            c_pos=tuple(data["c_pos"]),
            enemies=enemies,
        )


@dataclass
class RunResult:
    success: bool
    reason: str
    moves: int
    toggles: int
    reported_length: Optional[int]
    runtime_sec: float
    claimed_unsolvable: bool
    was_solvable: bool
    stderr: str = ""
    log: List[str] = field(default_factory=list)


@dataclass
class FailedTest:
    """Container for failed test information"""
    test_id: int
    algorithm: str
    variant: int
    error_category: str
    map_definition: Dict
    result: Dict
    seed: int

    def to_dict(self) -> Dict:
        return {
            "test_id": self.test_id,
            "algorithm": self.algorithm,
            "variant": self.variant,
            "error_category": self.error_category,
            "map_definition": self.map_definition,
            "result": self.result,
            "seed": self.seed,
        }


class FailureTracker:
    """Tracks and organizes failed tests by category and variation"""

    def __init__(self):
        # Structure: {algo_variant: {error_category: [FailedTest]}}
        self.failures: Dict[str, Dict[str, List[FailedTest]]] = defaultdict(lambda: defaultdict(list))
        self.test_counter = 0

    def add_failure(
            self,
            algorithm: str,
            variant: int,
            map_def: MapDefinition,
            result: RunResult,
            seed: int,
    ):
        """Add a failed test to the tracker"""
        self.test_counter += 1
        algo_variant = f"{algorithm}_v{variant}"

        failed_test = FailedTest(
            test_id=self.test_counter,
            algorithm=algorithm,
            variant=variant,
            error_category=result.reason,
            map_definition=map_def.to_dict(),
            result={
                "success": result.success,
                "reason": result.reason,
                "moves": result.moves,
                "toggles": result.toggles,
                "reported_length": result.reported_length,
                "runtime_sec": result.runtime_sec,
                "claimed_unsolvable": result.claimed_unsolvable,
                "was_solvable": result.was_solvable,
                "stderr": result.stderr,
                "log": result.log,
            },
            seed=seed,
        )

        self.failures[algo_variant][result.reason].append(failed_test)

    def save_failures(self, output_dir: Path, seed: int):
        """Save all failures to organized JSON files"""
        failures_dir = output_dir / "failed_tests"
        failures_dir.mkdir(parents=True, exist_ok=True)

        # Save summary
        summary = {
            "seed": seed,
            "total_failures": self.test_counter,
            "breakdown": {},
        }

        for algo_variant, categories in self.failures.items():
            summary["breakdown"][algo_variant] = {
                category: len(tests) for category, tests in categories.items()
            }

            # Save each category to separate file
            for category, tests in categories.items():
                safe_category = category.replace("/", "_").replace("\\", "_")
                filename = f"{algo_variant}_{safe_category}.json"
                filepath = failures_dir / filename

                data = {
                    "algorithm_variant": algo_variant,
                    "error_category": category,
                    "test_count": len(tests),
                    "tests": [test.to_dict() for test in tests],
                }

                with filepath.open("w", encoding="utf-8") as f:
                    json.dump(data, f, indent=2, ensure_ascii=False)

        # Save summary file
        summary_path = failures_dir / "summary.json"
        with summary_path.open("w", encoding="utf-8") as f:
            json.dump(summary, f, indent=2, ensure_ascii=False)

        # Create human-readable summary
        summary_txt = failures_dir / "summary.txt"
        with summary_txt.open("w", encoding="utf-8") as f:
            f.write(f"Failure Summary (Seed: {seed})\n")
            f.write("=" * 60 + "\n\n")
            f.write(f"Total failed tests: {self.test_counter}\n\n")

            for algo_variant, categories in sorted(self.failures.items()):
                f.write(f"\n{algo_variant}:\n")
                f.write("-" * 40 + "\n")
                total = sum(len(tests) for tests in categories.values())
                f.write(f"  Total failures: {total}\n\n")

                for category, tests in sorted(categories.items(), key=lambda x: len(x[1]), reverse=True):
                    f.write(f"  • {category}: {len(tests)} tests\n")

        print(f"\nFailure reports saved to {failures_dir}")
        print(f"  - Summary: summary.txt / summary.json")
        print(f"  - Detailed failures: *_*.json files")


# -------------------- (unchanged helper functions) -------------------------

def inside(x: int, y: int) -> bool:
    return 0 <= x < SIZE and 0 <= y < SIZE


def moore(radius: int) -> List[Tuple[int, int]]:
    return [
        (dx, dy)
        for dx in range(-radius, radius + 1)
        for dy in range(-radius, radius + 1)
        if max(abs(dx), abs(dy)) <= radius
    ]


def von_neumann(radius: int) -> List[Tuple[int, int]]:
    return [
        (dx, dy)
        for dx in range(-radius, radius + 1)
        for dy in range(-radius, radius + 1)
        if abs(dx) + abs(dy) <= radius
    ]


def translate(pos: Tuple[int, int], offsets: Iterable[Tuple[int, int]]) -> List[Tuple[int, int]]:
    x, y = pos
    return [(x + dx, y + dy) for dx, dy in offsets if inside(x + dx, y + dy)]


def nazgul_zone(pos: Tuple[int, int], ring_on: bool) -> set[Tuple[int, int]]:
    radius = 2 if ring_on else 1
    zone = set(translate(pos, moore(radius)))
    extension = radius + 1
    for dx, dy in ((extension, 0), (-extension, 0), (0, extension), (0, -extension)):
        nx, ny = pos[0] + dx, pos[1] + dy
        if inside(nx, ny):
            zone.add((nx, ny))
    return zone


def watchtower_zone(pos: Tuple[int, int], ring_on: bool) -> set[Tuple[int, int]]:
    radius = 2
    zone = set(translate(pos, moore(radius)))
    if ring_on:
        for delta in (-1, 0, 1):
            candidates = (
                (pos[0] + radius, pos[1] + delta),
                (pos[0] - radius, pos[1] + delta),
                (pos[0] + delta, pos[1] + radius),
                (pos[0] + delta, pos[1] - radius),
            )
            for nx, ny in candidates:
                if inside(nx, ny):
                    zone.add((nx, ny))
    return zone


def enemy_zone(enemy: Enemy, ring_on: bool, has_coat: bool) -> set[Tuple[int, int]]:
    pos = enemy.pos
    if enemy.kind == "O":
        radius = 1
        if ring_on or has_coat:
            radius = max(0, radius - 1)
        offsets = von_neumann(radius)
        return set(translate(pos, offsets))
    if enemy.kind == "U":
        radius = 2
        if ring_on or has_coat:
            radius = max(0, radius - 1)
        offsets = von_neumann(radius)
        return set(translate(pos, offsets))
    if enemy.kind == "N":
        return nazgul_zone(pos, ring_on)
    if enemy.kind == "W":
        return watchtower_zone(pos, ring_on)
    raise ValueError(f"Unknown enemy type {enemy.kind}")

def neuman_zone(x: int, y: int, radius: int) -> List[Tuple[int, int]]:
    cells = []
    for i in range(x - radius, x + radius + 1):
        for j in range(y - radius, y + radius + 1):
            if abs(x - i) + abs(y - j) <= radius:
                cells.append((i, j))
    return cells


def moore_zone(
    x: int, y: int, radius: int, with_ears: bool
) -> List[Tuple[int, int]]:
    cells = []
    for i in range(x - radius, x + radius + 1):
        for j in range(y - radius, y + radius + 1):
            cells.append((i, j))

    if with_ears:
        cells.extend([
            (x - radius - 1, y - radius - 1),
            (x - radius - 1, y + radius + 1),
            (x + radius + 1, y - radius - 1),
            (x + radius + 1, y + radius + 1),
        ])
    return cells

def get_zone(enemy: Enemy, with_ring:bool, hasCoat:bool) -> set[Tuple[int, int]]:
    x = enemy.x
    y = enemy.y
    token = enemy.kind
    if token == "O":
        return set(neuman_zone(x, y, 1 - (with_ring or hasCoat)))
    if token == "U":
        return set(neuman_zone(x, y, 2 - (with_ring or hasCoat)))
    if token == "N":
        if with_ring:
            return set(moore_zone(x, y, 2, False))
        return set(moore_zone(x, y, 1, not hasCoat))
    if token == "W":
        return set(moore_zone(x, y, 2, with_ring))

    return {(x, y)}

def compute_hazard_cache(map_def: MapDefinition) -> Dict[Tuple[bool, bool], set[Tuple[int, int]]]:
    cache: Dict[Tuple[bool, bool], set[Tuple[int, int]]] = {}
    for ring_on in (False, True):
        for has_coat in (False, True):
            zone: set[Tuple[int, int]] = set()
            for enemy in map_def.enemies:
                zone.update(get_zone(enemy, ring_on, has_coat))
                
            cache[(ring_on, has_coat)] = zone
    return cache


def random_cell(rng: random.Random, excluded: set[Tuple[int, int]]) -> Tuple[int, int]:
    candidates = [(x, y) for x in range(SIZE) for y in range(SIZE) if (x, y) not in excluded]
    if not candidates:
        raise RuntimeError("No free cells available for placement")
    return rng.choice(candidates)


def generate_map(rng: random.Random) -> MapDefinition:
    while True:
        occupied = {(0, 0)}
        g_pos = random_cell(rng, occupied)
        occupied.add(g_pos)

        m_pos = random_cell(rng, occupied)
        occupied.add(m_pos)

        c_pos = random_cell(rng, occupied)
        occupied.add(c_pos)

        enemies: List[Enemy] = []

        w_pos = random_cell(rng, occupied)
        occupied.add(w_pos)
        enemies.append(Enemy("W", *w_pos))

        u_pos = random_cell(rng, occupied)
        occupied.add(u_pos)
        enemies.append(Enemy("U", *u_pos))

        if rng.random() < 0.6:
            n_pos = random_cell(rng, occupied)
            occupied.add(n_pos)
            enemies.append(Enemy("N", *n_pos))

        orc_count = rng.choice((1, 2))
        for _ in range(orc_count):
            o_pos = random_cell(rng, occupied)
            occupied.add(o_pos)
            enemies.append(Enemy("O", *o_pos))

        map_def = MapDefinition(g_pos=g_pos, m_pos=m_pos, c_pos=c_pos, enemies=tuple(enemies))
        hazards = compute_hazard_cache(map_def)
        base_hazard = hazards[(False, False)]
        start_safe = (0, 0) not in base_hazard
        g_safe = g_pos not in base_hazard
        c_safe = c_pos not in base_hazard
        if start_safe and g_safe and c_safe:
            return map_def


def stage_transition(stage: int, position: Tuple[int, int], g_pos: Tuple[int, int]) -> int:
    if stage == 0 and position == g_pos:
        return 1
    return stage


def compute_shortest_paths(map_def: MapDefinition) -> Dict[str, Optional[int]]:
    hazards = compute_hazard_cache(map_def)
    enemy_cells = map_def.enemy_positions()
    c_pos = map_def.c_pos
    g_pos = map_def.g_pos
    m_pos = map_def.m_pos

    start_state = (0, 0, 0, 0, 0)
    dist: Dict[Tuple[int, int, int, int, int], int] = {start_state: 0}
    dq: deque[Tuple[int, int, int, int, int]] = deque([start_state])

    while dq:
        state = dq.popleft()
        x, y, ring, coat, stage = state
        moves = dist[state]

        current_hazard = hazards[(bool(ring), bool(coat))]

        new_ring = 1 - ring
        new_hazard = hazards[(bool(new_ring), bool(coat))]
        if (x, y) not in new_hazard and (x, y) not in enemy_cells:
            nxt = (x, y, new_ring, coat, stage)
            if nxt not in dist or moves < dist[nxt]:
                dist[nxt] = moves
                dq.appendleft(nxt)

        for dx, dy in DIRS:
            nx, ny = x + dx, y + dy
            if not inside(nx, ny):
                continue
            if (nx, ny) in enemy_cells:
                continue
            if (nx, ny) in current_hazard:
                continue
            next_coat = 1 if (coat or (nx, ny) == c_pos) else 0
            next_stage = stage_transition(stage, (nx, ny), g_pos)
            nxt = (nx, ny, ring, next_coat, next_stage)
            if nxt not in dist or moves + 1 < dist[nxt]:
                dist[nxt] = moves + 1
                dq.append(nxt)

    g_candidates = [
        dist[state]
        for state in dist
        if state[0] == g_pos[0] and state[1] == g_pos[1] and state[4] == 1
    ]
    m_candidates = [
        dist[state]
        for state in dist
        if state[0] == m_pos[0] and state[1] == m_pos[1] and state[4] == 1
    ]

    return {
        "dist_to_g": min(g_candidates) if g_candidates else None,
        "dist_to_m": min(m_candidates) if m_candidates else None,
    }


def get_visible_entries(
        map_def: MapDefinition,
        hazards: Dict[Tuple[bool, bool], set[Tuple[int, int]]],
        position: Tuple[int, int],
        variant: int,
        ring_on: bool,
        has_coat: bool,
        g_active: Optional[Tuple[int, int]],
        c_active: Optional[Tuple[int, int]],
) -> List[Tuple[int, int, str]]:
    radius = 1 if variant == 1 else 2
    hazard_set = hazards[(ring_on, has_coat)]
    entries: List[Tuple[int, int, str]] = []
    px, py = position
    items = {}
    if g_active is not None:
        items[g_active] = "G"
    if c_active is not None:
        items[c_active] = "C"
    items[map_def.m_pos] = "M"

    enemy_at = {enemy.pos: enemy.kind for enemy in map_def.enemies}

    for dx in range(-radius, radius + 1):
        for dy in range(-radius, radius + 1):
            if max(abs(dx), abs(dy)) > radius:
                continue
            nx, ny = px + dx, py + dy
            if not inside(nx, ny):
                continue
            if (nx, ny) == (px, py):
                continue
            if (nx, ny) in enemy_at:
                entries.append((nx, ny, enemy_at[(nx, ny)]))
            elif (nx, ny) in items:
                entries.append((nx, ny, items[(nx, ny)]))
            elif (nx, ny) in hazard_set:
                entries.append((nx, ny, "P"))
    entries.sort()
    return entries


def spawn_algorithm(algo: str) -> subprocess.Popen:
    if algo == "astar":
        script = ASTAR_SCRIPT
    elif algo == "backtracking":
        script = BACKTRACK_SCRIPT
    else:
        raise ValueError(f"Unknown algorithm {algo}")

    if not script.exists():
        raise FileNotFoundError(
            f"Solution script for '{algo}' not found: {script}.\n"
            "Set ASTAR_SCRIPT or BACKTRACK_SCRIPT env vars or place scripts under <project>/bin/"
        )

    command = [str(PYTHON_BIN), "-u", str(script)]
    return subprocess.Popen(
        command,
        cwd=ROOT,
        stdin=subprocess.PIPE,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
        bufsize=1,
    )


def reader_thread(stream, queue: Queue):
    try:
        for line in iter(stream.readline, ""):
            if not line:
                break
            queue.put(line.rstrip("\n"))
    finally:
        queue.put(SENTINEL)


def send_line(proc: subprocess.Popen, data: str) -> None:
    assert proc.stdin is not None
    proc.stdin.write(data + "\n")
    proc.stdin.flush()


def receive_line(queue: Queue, timeout: float) -> Tuple[Optional[str], bool]:
    try:
        item = queue.get(timeout=timeout)
    except Empty:
        return None, False
    if item is SENTINEL:
        return None, True
    return item, False


def run_single_simulation(
        map_def: MapDefinition,
        variant: int,
        algo: str,
        rng: random.Random,
        hazards: Dict[Tuple[bool, bool], set[Tuple[int, int]]],
        map_stats: Dict[str, Optional[int]],
) -> RunResult:
    proc = spawn_algorithm(algo)
    stdout_queue: Queue = Queue()
    stderr_queue: Queue = Queue()
    enemy_cells = map_def.enemy_positions()

    stdout_thread = threading.Thread(target=reader_thread, args=(proc.stdout, stdout_queue))
    stderr_thread = threading.Thread(target=reader_thread, args=(proc.stderr, stderr_queue))
    stdout_thread.daemon = True
    stderr_thread.daemon = True
    stdout_thread.start()
    stderr_thread.start()

    start_time = time.perf_counter()

    ring_on = False
    has_coat = False
    position = (0, 0)
    g_active: Optional[Tuple[int, int]] = map_def.g_pos
    c_active: Optional[Tuple[int, int]] = map_def.c_pos
    g_found = False
    moves = 0
    toggles = 0
    claimed_unsolvable = False
    log: List[str] = []

    send_line(proc, str(variant))
    send_line(proc, f"{map_def.g_pos[0]} {map_def.g_pos[1]}")
    initial_entries = get_visible_entries(
        map_def,
        hazards,
        position,
        variant,
        ring_on,
        has_coat,
        g_active,
        c_active,
    )
    send_line(proc, str(len(initial_entries)))
    for x, y, kind in initial_entries:
        send_line(proc, f"{x} {y} {kind}")

    stdout_timeout = COMMAND_TIMEOUT_SEC

    def consume_stderr() -> str:
        collected: List[str] = []
        while True:
            try:
                line = stderr_queue.get_nowait()
            except Empty:
                break
            if line is SENTINEL:
                continue
            collected.append(line)
        return "\n".join(collected)

    try:
        while True:
            elapsed = time.perf_counter() - start_time
            if elapsed > PROCESS_TIMEOUT_SEC:
                proc.kill()
                return RunResult(
                    success=False,
                    reason="timeout",
                    moves=moves,
                    toggles=toggles,
                    reported_length=None,
                    runtime_sec=elapsed,
                    claimed_unsolvable=False,
                    was_solvable=map_stats["dist_to_m"] is not None,
                    stderr=consume_stderr(),
                    log=log,
                )

            line, eof = receive_line(stdout_queue, stdout_timeout)
            if eof:
                if proc.poll() is not None:
                    return RunResult(
                        success=False,
                        reason="unexpected_termination",
                        moves=moves,
                        toggles=toggles,
                        reported_length=None,
                        runtime_sec=time.perf_counter() - start_time,
                        claimed_unsolvable=False,
                        was_solvable=map_stats["dist_to_m"] is not None,
                        stderr=consume_stderr(),
                        log=log,
                    )
                continue
            if line is None:
                proc.kill()
                return RunResult(
                    success=False,
                    reason="no_output",
                    moves=moves,
                    toggles=toggles,
                    reported_length=None,
                    runtime_sec=time.perf_counter() - start_time,
                    claimed_unsolvable=False,
                    was_solvable=map_stats["dist_to_m"] is not None,
                    stderr=consume_stderr(),
                    log=log,
                )
            log.append(line)

            tokens = line.strip().split()
            if not tokens:
                continue
            cmd = tokens[0]

            if cmd == "m":
                if len(tokens) != 3:
                    proc.kill()
                    return RunResult(
                        success=False,
                        reason="invalid_move_format",
                        moves=moves,
                        toggles=toggles,
                        reported_length=None,
                        runtime_sec=time.perf_counter() - start_time,
                        claimed_unsolvable=False,
                        was_solvable=map_stats["dist_to_m"] is not None,
                        stderr=consume_stderr(),
                        log=log,
                    )
                nx, ny = int(tokens[1]), int(tokens[2])
                if not inside(nx, ny):
                    proc.kill()
                    return RunResult(
                        success=False,
                        reason="move_out_of_bounds",
                        moves=moves,
                        toggles=toggles,
                        reported_length=None,
                        runtime_sec=time.perf_counter() - start_time,
                        claimed_unsolvable=False,
                        was_solvable=map_stats["dist_to_m"] is not None,
                        stderr=consume_stderr(),
                        log=log,
                    )
                if abs(position[0] - nx) + abs(position[1] - ny) != 1:
                    proc.kill()
                    return RunResult(
                        success=False,
                        reason="non_adjacent_move",
                        moves=moves,
                        toggles=toggles,
                        reported_length=None,
                        runtime_sec=time.perf_counter() - start_time,
                        claimed_unsolvable=False,
                        was_solvable=map_stats["dist_to_m"] is not None,
                        stderr=consume_stderr(),
                        log=log,
                    )
                hazard = hazards[(ring_on, has_coat)]
                if (nx, ny) in hazard or (nx, ny) in enemy_cells:
                    proc.kill()
                    return RunResult(
                        success=False,
                        reason="stepped_into_hazard",
                        moves=moves,
                        toggles=toggles,
                        reported_length=None,
                        runtime_sec=time.perf_counter() - start_time,
                        claimed_unsolvable=False,
                        was_solvable=map_stats["dist_to_m"] is not None,
                        stderr=consume_stderr(),
                        log=log,
                    )
                position = (nx, ny)
                moves += 1
                just_met_g = False
                if c_active is not None and position == c_active:
                    has_coat = True
                    c_active = None
                if g_active is not None and position == g_active:
                    g_found = True
                    g_active = None
                    just_met_g = True
                entries = get_visible_entries(
                    map_def,
                    hazards,
                    position,
                    variant,
                    ring_on,
                    has_coat,
                    g_active,
                    c_active,
                )
                send_line(proc, str(len(entries)))
                for x, y, kind in entries:
                    send_line(proc, f"{x} {y} {kind}")
                if just_met_g:
                    send_line(proc, f"My precious! Mount Doom is {map_def.m_pos[0]} {map_def.m_pos[1]}")

            elif cmd == "r":
                if ring_on:
                    proc.kill()
                    return RunResult(
                        success=False,
                        reason="ring_already_on",
                        moves=moves,
                        toggles=toggles,
                        reported_length=None,
                        runtime_sec=time.perf_counter() - start_time,
                        claimed_unsolvable=False,
                        was_solvable=map_stats["dist_to_m"] is not None,
                        stderr=consume_stderr(),
                        log=log,
                    )
                new_hazard = hazards[(True, has_coat)]
                if position in new_hazard:
                    proc.kill()
                    return RunResult(
                        success=False,
                        reason="toggle_into_hazard",
                        moves=moves,
                        toggles=toggles,
                        reported_length=None,
                        runtime_sec=time.perf_counter() - start_time,
                        claimed_unsolvable=False,
                        was_solvable=map_stats["dist_to_m"] is not None,
                        stderr=consume_stderr(),
                        log=log,
                    )
                ring_on = True
                toggles += 1
                entries = get_visible_entries(
                    map_def,
                    hazards,
                    position,
                    variant,
                    ring_on,
                    has_coat,
                    g_active,
                    c_active,
                )
                send_line(proc, str(len(entries)))
                for x, y, kind in entries:
                    send_line(proc, f"{x} {y} {kind}")

            elif cmd == "rr":
                if not ring_on:
                    proc.kill()
                    return RunResult(
                        success=False,
                        reason="ring_already_off",
                        moves=moves,
                        toggles=toggles,
                        reported_length=None,
                        runtime_sec=time.perf_counter() - start_time,
                        claimed_unsolvable=False,
                        was_solvable=map_stats["dist_to_m"] is not None,
                        stderr=consume_stderr(),
                        log=log,
                    )
                new_hazard = hazards[(False, has_coat)]
                if position in new_hazard:
                    proc.kill()
                    return RunResult(
                        success=False,
                        reason="toggle_into_hazard",
                        moves=moves,
                        toggles=toggles,
                        reported_length=None,
                        runtime_sec=time.perf_counter() - start_time,
                        claimed_unsolvable=False,
                        was_solvable=map_stats["dist_to_m"] is not None,
                        stderr=consume_stderr(),
                        log=log,
                    )
                ring_on = False
                toggles += 1
                entries = get_visible_entries(
                    map_def,
                    hazards,
                    position,
                    variant,
                    ring_on,
                    has_coat,
                    g_active,
                    c_active,
                )
                send_line(proc, str(len(entries)))
                for x, y, kind in entries:
                    send_line(proc, f"{x} {y} {kind}")

            elif cmd == "e":
                if len(tokens) != 2:
                    proc.kill()
                    return RunResult(
                        success=False,
                        reason="invalid_end_format",
                        moves=moves,
                        toggles=toggles,
                        reported_length=None,
                        runtime_sec=time.perf_counter() - start_time,
                        claimed_unsolvable=False,
                        was_solvable=map_stats["dist_to_m"] is not None,
                        stderr=consume_stderr(),
                        log=log,
                    )
                value = int(tokens[1])
                proc.terminate()
                claimed_unsolvable = value == -1
                was_solvable = map_stats["dist_to_m"] is not None
                ended_on_goal = position == map_def.m_pos and g_found
                success = (value >= 0 and ended_on_goal and not claimed_unsolvable) or (
                        claimed_unsolvable and not was_solvable
                )
                if claimed_unsolvable and was_solvable:
                    reason = "false_unsolvable"
                elif not claimed_unsolvable and not ended_on_goal:
                    reason = "ended_without_goal"
                else:
                    reason = "ok" if success else "invalid_result"
                return RunResult(
                    success=success,
                    reason=reason,
                    moves=moves,
                    toggles=toggles,
                    reported_length=value if value >= 0 else None,
                    runtime_sec=time.perf_counter() - start_time,
                    claimed_unsolvable=claimed_unsolvable,
                    was_solvable=was_solvable,
                    stderr=consume_stderr(),
                    log=log,
                )
            else:
                proc.kill()
                return RunResult(
                    success=False,
                    reason="unknown_command",
                    moves=moves,
                    toggles=toggles,
                    reported_length=None,
                    runtime_sec=time.perf_counter() - start_time,
                    claimed_unsolvable=False,
                    was_solvable=map_stats["dist_to_m"] is not None,
                    stderr=consume_stderr(),
                    log=log,
                )
    finally:
        try:
            proc.kill()
        except Exception:
            pass
        consume_stderr()


def aggregate_statistics(results: Sequence[RunResult]) -> Dict[str, object]:
    runtimes_all = [res.runtime_sec for res in results]
    runtimes_success = [res.runtime_sec for res in results if res.success]
    moves_success = [res.moves for res in results if res.success]
    toggles_success = [res.toggles for res in results if res.success]
    success_count = sum(1 for res in results if res.success)
    failure_count = len(results) - success_count

    def safe_mode(values: Sequence[float | int]) -> Optional[float]:
        if not values:
            return None
        try:
            return float(statistics.mode(values))
        except statistics.StatisticsError:
            return None

    reason_counts: Dict[str, int] = {}
    for res in results:
        reason_counts[res.reason] = reason_counts.get(res.reason, 0) + 1

    claimed_unsolvable_total = sum(1 for res in results if res.claimed_unsolvable)
    correct_unsolvable = sum(
        1 for res in results if res.claimed_unsolvable and not res.was_solvable and res.success
    )
    false_unsolvable = sum(1 for res in results if res.reason == "false_unsolvable")

    summary = {
        "total_runs": len(results),
        "successes": success_count,
        "failures": failure_count,
        "success_rate": success_count / len(results) if results else 0.0,
        "failure_rate": failure_count / len(results) if results else 0.0,
        "runtime_mean_all": statistics.fmean(runtimes_all) if runtimes_all else None,
        "runtime_median_all": statistics.median(runtimes_all) if runtimes_all else None,
        "runtime_mode_all": safe_mode(runtimes_all),
        "runtime_std_all": statistics.pstdev(runtimes_all) if len(runtimes_all) > 1 else 0.0,
        "runtime_mean_success": statistics.fmean(runtimes_success) if runtimes_success else None,
        "runtime_median_success": statistics.median(runtimes_success) if runtimes_success else None,
        "runtime_mode_success": safe_mode(runtimes_success),
        "runtime_std_success": statistics.pstdev(runtimes_success) if len(runtimes_success) > 1 else 0.0,
        "moves_mean_success": statistics.fmean(moves_success) if moves_success else None,
        "moves_median_success": statistics.median(moves_success) if moves_success else None,
        "moves_mode_success": safe_mode(moves_success),
        "moves_std_success": statistics.pstdev(moves_success) if len(moves_success) > 1 else 0.0,
        "toggles_mean_success": statistics.fmean(toggles_success) if toggles_success else None,
        "toggles_median_success": statistics.median(toggles_success) if toggles_success else None,
        "toggles_mode_success": safe_mode(toggles_success),
        "toggles_std_success": statistics.pstdev(toggles_success) if len(toggles_success) > 1 else 0.0,
        "claimed_unsolvable_total": claimed_unsolvable_total,
        "claimed_unsolvable_false": false_unsolvable,
        "claimed_unsolvable_correct": correct_unsolvable,
        "reason_breakdown": reason_counts,
    }
    return summary


def run_experiment(map_count: int, seed: int) -> Tuple[Dict[str, Dict[str, object]], FailureTracker]:
    rng = random.Random(seed)
    results: Dict[str, List[RunResult]] = {
        "astar_v1": [],
        "astar_v2": [],
        "backtracking_v1": [],
        "backtracking_v2": [],
    }

    failure_tracker = FailureTracker()

    for idx in range(map_count):
        map_def = generate_map(rng)
        hazards = compute_hazard_cache(map_def)
        map_stats = compute_shortest_paths(map_def)

        for variant, key_prefix in ((1, "v1"), (2, "v2")):
            result_astar = run_single_simulation(
                map_def,
                variant,
                "astar",
                rng,
                hazards,
                map_stats,
            )
            results[f"astar_{key_prefix}"].append(result_astar)
            if not result_astar.success:
                failure_tracker.add_failure("astar", variant, map_def, result_astar, seed)

            result_backtracking = run_single_simulation(
                map_def,
                variant,
                "backtracking",
                rng,
                hazards,
                map_stats,
            )
            results[f"backtracking_{key_prefix}"].append(result_backtracking)
            if not result_backtracking.success:
                failure_tracker.add_failure("backtracking", variant, map_def, result_backtracking, seed)

        # Progress indicator
        if (idx + 1) % 100 == 0:
            print(f"Processed {idx + 1}/{map_count} maps...")

    aggregated = {key: aggregate_statistics(value) for key, value in results.items()}
    return aggregated, failure_tracker

# Добавьте эти функции в существующий код перед main()

def load_test_suite(json_file: str= "generated_tests.json") -> List[MapDefinition]:
    """Загружает тесты из JSON файла и конвертирует в MapDefinition"""
    with open(json_file, 'r', encoding='utf-8') as f:
        test_data = json.load(f)
    map_definitions = []
    
    for test_case in test_data["tests"]:
        # Извлекаем обязательные токены
        g_pos = None
        m_pos = None
        c_pos = None
        enemies = []
        
        for token in test_case["tokens"]:
            if token["type"] == "G":
                g_pos = (token["x"], token["y"])
            elif token["type"] == "M":
                m_pos = (token["x"], token["y"])
            elif token["type"] == "C":
                c_pos = (token["x"], token["y"])
            elif token["type"] in ["O", "U", "N", "W"]:
                enemies.append(Enemy(token["type"], token["x"], token["y"]))
        
        # Проверяем что все обязательные токены найдены
        if g_pos is None or m_pos is None or c_pos is None:
            print(f"Warning: Missing required tokens in test case {test_case.get('test_id', 'unknown')}")
            continue
        
        map_def = MapDefinition(
            g_pos=g_pos,
            m_pos=m_pos,
            c_pos=c_pos,
            enemies=tuple(enemies)
        )
        
        map_definitions.append(map_def)
    
    print(f"Loaded {len(map_definitions)} map definitions from {json_file}")
    return map_definitions

def run_experiment_from_json(json_file: str, seed: int = None) -> Tuple[Dict[str, Dict[str, object]], FailureTracker]:
    """Запускает эксперимент на тестах из JSON файла"""
    # Загружаем тесты
    map_definitions = load_test_suite(json_file)
    
    if seed is None:
        seed = int.from_bytes(os.urandom(8), "little")
    
    rng = random.Random(seed)
    results: Dict[str, List[RunResult]] = {
        "astar_v1": [],
        "astar_v2": [],
        "backtracking_v1": [],
        "backtracking_v2": [],
    }

    failure_tracker = FailureTracker()

    for idx, map_def in enumerate(map_definitions):
        hazards = compute_hazard_cache(map_def)
        map_stats = compute_shortest_paths(map_def)

        for variant, key_prefix in ((1, "v1"), (2, "v2")):
            result_astar = run_single_simulation(
                map_def,
                variant,
                "astar",
                rng,
                hazards,
                map_stats,
            )
            results[f"astar_{key_prefix}"].append(result_astar)
            if not result_astar.success:
                failure_tracker.add_failure("astar", variant, map_def, result_astar, seed)

            result_backtracking = run_single_simulation(
                map_def,
                variant,
                "backtracking",
                rng,
                hazards,
                map_stats,
            )
            results[f"backtracking_{key_prefix}"].append(result_backtracking)
            if not result_backtracking.success:
                failure_tracker.add_failure("backtracking", variant, map_def, result_backtracking, seed)

        # Progress indicator
        if (idx + 1) % 100 == 0:
            print(f"Processed {idx + 1}/{len(map_definitions)} maps...")

    aggregated = {key: aggregate_statistics(value) for key, value in results.items()}
    return aggregated, failure_tracker



def main() -> None:
    import argparse

    parser = argparse.ArgumentParser(
        description="Run Frodo pathfinding statistical comparison with failure tracking."
    )
    parser.add_argument(
        "--maps", 
        type=int, 
        default=DEFAULT_MAPS_PER_RUN,
        help="Number of maps to simulate (default: 1000)"
    )
    parser.add_argument(
        "--seed", 
        type=int, 
        default=DEFAULT_SEED, 
        help="PRNG seed (default: random)"
    )
    parser.add_argument(
        "--json-file",
        type=str,
        help="Use pre-generated test cases from JSON file instead of random generation"
    )
    parser.add_argument(
        "--validate",
        action="store_true",
        help="Validate JSON test suite without running tests"
    )
    
    args = parser.parse_args()
    json_file = "generated_tests.json"

    # Инициализируем переменные
    chosen_seed = None
    summary = None
    failure_tracker = None

    # Выбор источника тестов
    if json_file:
        print(f"Using test cases from: {json_file}")
        summary, failure_tracker = run_experiment_from_json(json_file, args.seed)
        source = f"JSON: {json_file}"
        maps_used = "all from JSON"  # Все тесты из JSON файла
    else:
        # Оригинальная случайная генерация
        if args.seed is None:
            chosen_seed = int.from_bytes(os.urandom(8), "little")
        else:
            chosen_seed = args.seed

        print(f"Running experiment with seed: {chosen_seed}")
        print(f"Testing {args.maps} maps...\n")
        summary, failure_tracker = run_experiment(args.maps, chosen_seed)
        source = f"seed: {chosen_seed}"
        maps_used = args.maps

    output_dir = ROOT / "analysis"
    output_dir.mkdir(parents=True, exist_ok=True)

    # Сохранение результатов
    out_path = output_dir / "stats_summary.txt"
    with out_path.open("w", encoding="utf-8") as f:
        f.write(f"Test source: {source}\n")
        f.write(f"maps: {maps_used}\n\n")
        for key, data in summary.items():
            f.write(f"=== {key} ===\n")
            for metric, value in data.items():
                f.write(f"{metric}: {value}\n")
            f.write("\n")
    print(f"\nStatistical summary written to {out_path}")

    # Сохранение отчетов об ошибках
    seed_for_failures = chosen_seed if chosen_seed is not None else (args.seed if args.seed is not None else 0)
    failure_tracker.save_failures(output_dir, seed_for_failures)


if __name__ == "__main__":
    main()