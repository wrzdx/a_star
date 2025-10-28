from random import randint
import subprocess
from typing import List, Tuple


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
        x, y = randint(0, MAP_SIZE - 1), randint(0, MAP_SIZE - 1)
        zone = set(get_zone(x, y, token, False) + get_zone(x, y, token, True))
        while (0, 0) in zone or self.world_map[x][y] != " ":
            x, y = randint(0, MAP_SIZE - 1), randint(0, MAP_SIZE - 1)
            zone = get_zone(x, y, token, False) + get_zone(x, y, token, True)

        for i, j in zone:
            if 0 <= i < MAP_SIZE and 0 <= j < MAP_SIZE:
                self.world_map[i][j] = "P"

        self.set_token(x, y, token)
        self.world_map[x][y] = token

    def set_random_tokens(self):
        tokens = randint(1, 2) * "O" + "U" + "N" * randint(0, 1) + "WGCM"
        for token in tokens:
            self.set_token_randomly(token)
        self.update_map(False)

    def start(self):
        if self.mount == (-1, -1) or self.gollum == (-1, -1):
            print("Goals are not defined")
            return -1, []
        answer = -1
        history = ""
        proc = subprocess.Popen(
            ["python", "a_star.py"],
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
            history += response
            proc.stdin.write(response)
            proc.stdin.flush()

            line = proc.stdout.readline().strip()
            history += line + "\n"
            if not line:
                break
            if "e" in line:
                answer = int(line.split()[-1])
                line = proc.stdout.readline().strip()
                history = line + "\n"
                break

            if "rr" == line:
                self.update_map(False)
                if self.world_map[x][y] in "POUNW" or not with_ring:
                    proc.terminate()
                    return -1, history + "Failed"
                with_ring = False
                continue
            if "r" == line:
                self.update_map(True)
                if self.world_map[x][y] in "POUNW" or with_ring:
                    proc.terminate()
                    return -1, history + "Failed"
                with_ring = True
                continue

            x, y = map(int, line.split())
            if self.world_map[x][y] in "POUNW":
                proc.terminate()
                return -1, history + "Failed"

        proc.terminate()
        return answer, history


def print_map_colored(world_map, path_str=None):
    colors = {
        " ": "\033[90m·\033[0m",
        "P": "\033[94mP\033[0m",
        "O": "\033[91mO\033[0m", 
        "N": "\033[95mN\033[0m",
        "W": "\033[93mW\033[0m",
        "M": "\033[92mM\033[0m",
        "G": "\033[96mG\033[0m",
        "*": "\033[97m*\033[0m",  # путь
    }

    # Создаем копию для отображения
    display_map = [row[:] for row in world_map]
    path = []
    for pair in path_str.split(") ("):
        clean = pair.strip('()\n')
        x, y = map(int, clean.split(','))
        path.append((x, y))
    # Отмечаем путь если передан
    if path:
        for cell in path:
            x, y = cell
                
            if 0 <= y < len(display_map) and 0 <= x < len(display_map[0]):
                # Ставим '*' только на пустых клетках, чтобы не перезаписать важные объекты
                if display_map[x][y] == " ":
                    display_map[x][y] = "*"

    print("   " + " ".join(f"{i:2}" for i in range(len(display_map[0]))))

    for i, row in enumerate(display_map):
        colored_row = [colors.get(str(cell), str(cell)) for cell in row]
        print(f"{i:2}  " + "  ".join(colored_row))


interactor = Interactor(randint(1,2))
interactor.set_random_tokens()
x, y = interactor.gollum
print(interactor.radius, x, y)

answer, history = interactor.start()
interactor.update_map(False)
interactor.world_map[x][y] = "G"
print_map_colored(interactor.world_map, history)
print()
print(history)
print(answer)
