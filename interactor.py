import random
import subprocess
from typing import List, Tuple

from utility import get_zone, moore_zone

MAP_SIZE: int = 13


class Interactor:
    def __init__(
        self,
        radius: int,
    ):
        self.world_map = [[" " for i in range(MAP_SIZE)] for j in range(MAP_SIZE)]
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

    def get_perceptable_neighbours(self, x: int, y: int) -> List[Tuple[int, int]]:
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
        tokens = random.randint(1, 2) * "O" + "U" + "N" * random.randint(0, 1) + "WGCM"
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
                    "My precious! Mount Doom is " + f"{self.mount[0]} {self.mount[1]}\n"
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
