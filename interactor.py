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
            (x - radius - 1, y - radius + 1),
            (x - radius + 1, y - radius - 1),
            (x - radius + 1, y - radius + 1),
        ])
    return cells


def get_zone(x: int, y: int, token: int, with_ring) -> List[Tuple[int, int]]:
    if token == "O":
        return neuman_zone(x, y, 1 - with_ring)
    if token == "U":
        return neuman_zone(x, y, 2 - with_ring)
    if token == "N":
        if with_ring:
            moore_zone(x, y, 2, False)
        return moore_zone(x, y, 1, True)
    if token == "W":
        return moore_zone(x, y, 2, with_ring)

    return [(x, y)]


class Interactor:
    def __init__(
        self,
        radius: int,
        gollum: Tuple[int, int],
        mount: Tuple[int, int],
    ):
        self.world_map = [
            [" " for i in range(MAP_SIZE)] for j in range(MAP_SIZE)
        ]
        self.with_ring = False
        self.radius = radius
        self.gollum = gollum
        self.mount = mount
        self.setToken(gollum[0], gollum[1], "G")
        self.setToken(mount[0], mount[1], "M")

    def setToken(self, x: int, y: int, token: str) -> None:
        self.world_map[x][y] = token

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
            for i, j in moore_zone(x, y, 1, False)
            if (i, j) != (x, y)
            and 0 <= i < MAP_SIZE
            and 0 <= j < MAP_SIZE
            and self.world_map[i][j] != " "
        ]

    def start(self):
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
        response += f"{self.gollum[0]} {self.gollum[1]}\n"
        history += response
        proc.stdin.write(response)
        proc.stdin.flush()
        is_mount_detected = False
        x, y = 0, 0

        while True:
            neighbours = self.get_perceptable_neighbours(x, y)
            response = f"{len(neighbours)}\n"
            for i, j in neighbours:
                response += f"{i} {j} {self.world_map[i][j]}\n"

            if (x, y) == self.gollum and not is_mount_detected:
                is_mount_detected = True
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
                history += line + "\n"
                break

            if "rr" == line:
                self.update_map(False)
                continue
            if "r" == line:
                self.update_map(True)
                continue

            x, y = map(int, line.split())

        proc.terminate()
        return answer, history


interactor = Interactor(1, (5, 5), (6, 6))
interactor.setToken(2, 0, "O")
interactor.setToken(1, 1, "O")
interactor.setToken(0, 3, "U")
# interactor.setToken(9, 5, "W")
interactor.update_map(False)
answer, history = interactor.start()
# print(history)
print(answer)
