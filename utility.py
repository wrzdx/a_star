from typing import List, Tuple


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


def neuman_zone(x: int, y: int, radius: int) -> List[Tuple[int, int]]:
    cells = []
    for i in range(x - radius, x + radius + 1):
        for j in range(y - radius, y + radius + 1):
            if abs(x - i) + abs(y - j) <= radius:
                cells.append((i, j))
    return cells


def moore_zone(x: int, y: int, radius: int, with_ears: bool) -> List[Tuple[int, int]]:
    cells = []
    for i in range(x - radius, x + radius + 1):
        for j in range(y - radius, y + radius + 1):
            cells.append((i, j))

    if with_ears:
        cells.extend(
            [
                (x - radius - 1, y - radius - 1),
                (x - radius - 1, y + radius + 1),
                (x + radius + 1, y - radius - 1),
                (x + radius + 1, y + radius + 1),
            ]
        )
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
