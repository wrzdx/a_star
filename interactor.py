import os
import random
import subprocess
from typing import List, Tuple
import time
from statistics import mean, median, stdev, mode

MAP_SIZE: int = 13


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


def get_zone(x: int, y: int, token: int, with_ring) -> List[Tuple[int, int]]:
    if token == "O":
        return neuman_zone(x, y, 1 - with_ring)
    if token == "U":
        return neuman_zone(x, y, 2 - with_ring)
    if token == "N":
        if with_ring:
            return moore_zone(x, y, 2, False)
        return moore_zone(x, y, 1, True)
    if token == "W":
        return moore_zone(x, y, 2, with_ring)

    return [(x, y)]


class Interactor:
    def __init__(
        self,
        radius: int,
    ):
        self.world_map = [
            [" " for i in range(MAP_SIZE)] for j in range(MAP_SIZE)
        ]
        self.with_ring = False
        self.radius = radius
        self.gollum = (-1, -1)
        self.mount = (-1, -1)

    def set_token(self, x: int, y: int, token: str) -> None:
        self.world_map[x][y] = token
        if token == "G":
            self.gollum = (x, y)
        if token == "M":
            self.mount = (x, y)

    def update_map(self, with_ring: bool) -> None:
        self.with_ring = with_ring
        for i in range(MAP_SIZE):
            for j in range(MAP_SIZE):
                if self.world_map[i][j] == "P":
                    self.world_map[i][j] = " "
        for i in range(MAP_SIZE):
            for j in range(MAP_SIZE):
                if self.world_map[i][j] != " ":
                    zone = get_zone(i, j, self.world_map[i][j], with_ring)
                    for x, y in zone:
                        if (
                            0 <= x < MAP_SIZE
                            and 0 <= y < MAP_SIZE
                            and self.world_map[x][y] == " "
                        ):
                            self.world_map[x][y] = "P"

    def get_perceptable_neighbours(
        self, x: int, y: int
    ) -> List[Tuple[int, int]]:
        return [
            (i, j)
            for i, j in moore_zone(x, y, self.radius, False)
            if (i, j) != (x, y)
            and 0 <= i < MAP_SIZE
            and 0 <= j < MAP_SIZE
            and self.world_map[i][j] != " "
        ]

    def set_token_randomly(self, token: str):
        x, y = random.randint(0, MAP_SIZE - 1), random.randint(0, MAP_SIZE - 1)
        zone = set(get_zone(x, y, token, False) + get_zone(x, y, token, True))
        while (0, 0) in zone or self.world_map[x][y] != " ":
            x, y = (
                random.randint(0, MAP_SIZE - 1),
                random.randint(0, MAP_SIZE - 1),
            )
            zone = get_zone(x, y, token, False) + get_zone(x, y, token, True)

        for i, j in zone:
            if 0 <= i < MAP_SIZE and 0 <= j < MAP_SIZE:
                self.world_map[i][j] = "P"

        self.set_token(x, y, token)
        self.world_map[x][y] = token

    def set_random_tokens(self):
        tokens = (
            random.randint(1, 2) * "O"
            + "U"
            + "N" * random.randint(0, 1)
            + "WGCM"
        )
        for token in tokens:
            self.set_token_randomly(token)
        self.update_map(False)

    def start(self, algorithm: str):
        if self.mount == (-1, -1) or self.gollum == (-1, -1):
            return -1, "Goals are not defined\nFailed"
        answer = -1
        history = ""
        proc = subprocess.Popen(
            ["python", algorithm],
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            text=True,
            bufsize=1,
        )

        response = f"{self.radius}\n"
        with_ring = False
        response += f"{self.gollum[0]} {self.gollum[1]}\n"
        history += response
        proc.stdin.write(response)
        proc.stdin.flush()
        x, y = 0, 0

        while True:
            neighbours = self.get_perceptable_neighbours(x, y)
            response = f"{len(neighbours)}\n"
            for i, j in neighbours:
                response += f"{i} {j} {self.world_map[i][j]}\n"

            if (x, y) == self.gollum:
                self.set_token(self.gollum[0], self.gollum[1], " ")
                self.gollum = (-1, -1)
                response += (
                    "My precious! Mount Doom is "
                    + f"{self.mount[0]} {self.mount[1]}\n"
                )
            proc.stdin.write(response)
            proc.stdin.flush()
            history += response
            line = proc.stdout.readline().strip()
            history += line + "\n"
            if not line:
                break
            if "e" in line:
                answer = int(line.split()[-1])
                line = proc.stdout.readline().strip()
                history += line + "\n"
                break

            if "rr" == line:
                self.update_map(False)
                if self.world_map[x][y] in "POUNW" or not with_ring:
                    proc.terminate()
                    return -1, history + "BadSwitch\nFailed"
                with_ring = False
                continue
            if "r" == line:
                self.update_map(True)
                if self.world_map[x][y] in "POUNW" or with_ring:
                    proc.terminate()
                    return -1, history + "BadSwitch\nFailed"
                with_ring = True
                continue

            new_x, new_y = map(int, line.split()[1:])
            if (
                self.world_map[new_x][new_y] in "POUNW"
                or (abs(new_x - x) + abs(new_y - y)) != 1
            ):
                proc.terminate()
                return (
                    -1,
                    history + f"{new_x=} {new_y=} {x=} {y=}BadMove\nFailed",
                )
            x, y = new_x, new_y
        proc.terminate()
        return answer, history


A_STAR = "ee.py"
BACKTRACKING = "backtracking.py"


def run_comparison(number_of_runs: int = 10):
    stats = {
        "a_star": {"wins": 0, "losses": 0, "times": []},
        "backtracking": {"wins": 0, "losses": 0, "times": []},
    }

    for i in range(1, number_of_runs + 1):
        # Setup interactor with random tokens
        interactor = Interactor(random.randint(1, 2))
        interactor.set_random_tokens()
        x, y = interactor.gollum

        # A*
        start_time = time.time()
        answer0, history0 = interactor.start(A_STAR)
        time0 = time.time() - start_time
        path0 = history0.splitlines()[-1]

        # Clear map for next run
        interactor.update_map(False)
        interactor.set_token(x, y, "G")

        # Backtracking
        start_time = time.time()
        answer1, history1 = interactor.start(BACKTRACKING)
        time1 = time.time() - start_time
        path1 = history1.splitlines()[-1]

        # Clear map for next run
        interactor.update_map(False)
        interactor.set_token(x, y, "G")
        # Answer

        interactor.radius = 5
        answer2, history2 = interactor.start(A_STAR)

        # Check for failures
        if path0 == "Failed" or path1 == "Failed":
            interactor.update_map(False)
            interactor.set_token(x, y, "G")
            print_map_colored(interactor.world_map)
            break

        # Update statistics
        stats["a_star"]["wins"] += answer0 == answer2
        stats["a_star"]["losses"] += answer0 != answer2
        stats["a_star"]["times"].append(time0)

        stats["backtracking"]["wins"] += answer1 == answer2
        stats["backtracking"]["losses"] += answer1 != answer2
        stats["backtracking"]["times"].append(time1)

        print(
            f"\rProgress: {100 * i / number_of_runs:.1f}%", end="", flush=True
        )

    return stats


def print_statistics(stats):
    print("\n" + "=" * 60)
    print("Statistics:")
    print("=" * 60)

    for algo in ["a_star", "backtracking"]:
        data = stats[algo]
        total = data["wins"] + data["losses"]
        win_rate = (data["wins"] / total * 100) if total > 0 else 0

        print(f"\n{algo.upper()}:")
        print(f"  Wins: {data['wins']}")
        print(f"  Losses: {data['losses']}")
        print(f"  Win Rate: {win_rate:.1f}%")

        if data["times"]:
            print("  Time Statistics (seconds):")
            print(f"    mean = {mean(data['times']):.3f}")
            print(f"    median = {median(data['times']):.3f}")
            print(
                f"    mode = {mode(map(lambda x: round(x, 3), data['times']))}"
            )
            print(f"    std = {stdev(data['times']):.3f}")


# Run the comparison and print statistics
# stats = run_comparison(10)
# print_statistics(stats)
# input("Нажмите Enter для выхода...")
# interactor = Interactor(random.randint(1, 2))
# interactor.set_random_tokens()
# x, y = interactor.gollum


def print_map_colored(world_map, path_str=None):
    colors = {
        " ": "\033[90m·\033[0m",
        "P": "\033[94mP\033[0m",
        "O": "\033[91mO\033[0m",
        "N": "\033[95mN\033[0m",
        "W": "\033[93mW\033[0m",
        "M": "\033[92mM\033[0m",
        "G": "\033[96mG\033[0m",
        "*": "\033[97m*\033[0m",
    }

    display_map = [row[:] for row in world_map]
    path = []
    if path_str and path_str != "Failed":
        for pair in path_str.split(") ("):
            clean = pair.strip("()\n")
            x, y = map(int, clean.split(","))
            path.append((x, y))
    if path:
        for cell in path:
            x, y = cell

            if 0 <= y < len(display_map) and 0 <= x < len(display_map[0]):
                if display_map[x][y] == " ":
                    display_map[x][y] = "*"

    print("   " + " ".join(f"{i:2}" for i in range(len(display_map[0]))))

    for i, row in enumerate(display_map):
        colored_row = [colors.get(str(cell), str(cell)) for cell in row]
        print(f"{i:2}  " + "  ".join(colored_row))


# random.seed(int.from_bytes(os.urandom(8), 'big') + time.time_ns())
# n = 100
# for i in range(1, n + 1):
#     interactor = Interactor(random.randint(1, 2))
#     interactor.set_random_tokens()
#     x, y = interactor.gollum
#     answer, history = interactor.start(A_STAR)
#     if history.splitlines()[-1] == "Failed":
#         interactor.update_map(False)
#         interactor.set_token(x, y, "G")
#         print_map_colored(interactor.world_map, history.splitlines()[-1])
#         print(f"Answer: {answer}")
#         with open("history.txt", 'w', encoding='utf-8') as f:
#             f.write(history)
#         break
#     print(f"\rProgress: {100 * i / n:.1f}%", end="")

interactor = Interactor(2)
interactor.set_token(12, 8, "G")
interactor.set_token(1, 10, "M")
interactor.set_token(8, 6, "C")
interactor.set_token(9, 9, "W")
interactor.set_token(0, 8, "U")
interactor.set_token(4, 9, "N")
interactor.set_token(2, 9, "O")
interactor.set_token(10, 11, "O")
interactor.update_map(False)
x, y = interactor.gollum
answer, history = interactor.start(A_STAR)
interactor.set_token(x, y, "G")
print_map_colored(interactor.world_map, history.splitlines()[-1])
print(f"Answer: {answer}")
with open("history.txt", "w", encoding="utf-8") as f:
    f.write(history)
