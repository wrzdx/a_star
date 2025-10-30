from typing import List, Tuple
import json

FILE = "generated_tests.json"
# Чтение из файла
with open(FILE, "r", encoding="utf-8") as f:
    data = json.load(f)


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


for test in data["tests"]:
    danger_cells = set()
    for token in test["tokens"]:
        if token["type"] in "MGC": 
            continue
        danger_cells.update(
            get_zone(token["x"], token["y"], token["type"], True)
        )
        danger_cells.update(
            get_zone(token["x"], token["y"], token["type"], False)
        )
    valid = True
    for token in test["tokens"]:
        if token["type"] in "MGC":
            if (token["x"], token["y"]) in danger_cells:
                valid = False
    if (0, 0) in danger_cells:
        valid = False
    if not valid:
        print(test)
        print("Invalid tests")
        break
