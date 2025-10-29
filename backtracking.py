from typing import List, Optional, Tuple

MAP_SIZE: int = 13
INF: float = float("inf")
IMPOSSIBLE: int = 0
WITH_RING: int = 1
WITHOUT_RING: int = 2
DANGEROUS_TOKENS = "POUNW"


class Cell:
    def __init__(
        self,
        x: int,
        y: int,
        cost: int = INF,
        move_mode: int = IMPOSSIBLE,
        parent: Optional["Cell"] = None,
        token: str = "",
    ):
        self.x, self.y = (x, y)
        self.cost = cost
        self.move_mode = move_mode
        self.parent = parent
        self.token = token

    def __eq__(self, other: Optional["Cell"]):
        return other and other.x == self.x and other.y == self.y


def read_obstacles(
    current_cell: Cell,
    world_map: List[List[Cell]] | None = None,
    radius: int = 1,
) -> List[Tuple[int, int]]:
    p = int(input())
    allowed_moves = []
    for x in range(current_cell.x - radius, current_cell.x + radius + 1):
        if x < 0 or x >= MAP_SIZE:
            continue
        for y in range(current_cell.y - radius, current_cell.y + radius + 1):
            if y < 0 or y >= MAP_SIZE or (x == current_cell.x and y == current_cell.y):
                continue
            allowed_moves.append((x, y))
    for i in range(p):
        x, y, token = input().split()
        x, y = map(int, [x, y])
        if world_map:
            world_map[x][y].token = token
        if token in DANGEROUS_TOKENS:
            allowed_moves = [pos for pos in allowed_moves if pos != (x, y)]
    return allowed_moves


def update_neighbours(
    current_cell: Cell,
    allowed_cells: List[Cell],
    move_mode: int,
) -> List[Cell]:
    cells_to_move = [
        (current_cell.x + 1, current_cell.y),
        (current_cell.x, current_cell.y + 1),
        (current_cell.x - 1, current_cell.y),
        (current_cell.x, current_cell.y - 1),
    ]
    updated = []
    for cell in allowed_cells:
        cell.move_mode |= move_mode
        if (cell.x, cell.y) in cells_to_move:
            new_cost = current_cell.cost + 1
            if new_cost < cell.cost:
                cell.cost = new_cost
                cell.parent = current_cell
                updated.append(cell)

    return updated


def switch_move_mode(move_mode: int) -> int:
    return WITH_RING if move_mode == WITHOUT_RING else WITHOUT_RING


def update_cell_state(
    cell: Cell,
    move_mode: int,
    radius: int,
    world_map: List[List[Cell]],
) -> List[Cell]:
    allowed_moves = read_obstacles(cell, world_map, radius)
    allowed_cells = [world_map[x][y] for x, y in allowed_moves]
    return update_neighbours(cell, allowed_cells, move_mode)


def try_switch_mode(cell: Cell, current_move_mode: int) -> int:
    if not (cell.move_mode & current_move_mode):
        current_move_mode = switch_move_mode(current_move_mode)
        print("r" if current_move_mode == WITH_RING else "rr")
        read_obstacles(cell)
    return current_move_mode


def get_path(cell: Cell) -> List[Cell]:
    path: List[Cell] = [cell]
    next_cell: Cell = cell.parent
    while next_cell:
        path.append(next_cell)
        next_cell = next_cell.parent

    return path[::-1]


def check_and_go(current: Cell, next_cell: Cell, current_move_mode: int) -> int:
    if current != next_cell.parent:
        while current.parent:
            current_move_mode = try_switch_mode(current.parent, current_move_mode)
            current = current.parent
            print(current.x, current.y)
            read_obstacles(current)

        path = get_path(next_cell)[1:-1]
        for cell in path:
            current_move_mode = try_switch_mode(cell, current_move_mode)
            print(cell.x, cell.y)
            read_obstacles(cell)

    return current_move_mode


def backtrack(
    current: Cell,
    goal: Cell,
    radius: int,
    world_map: List[List[Cell]],
    move_mode: int,
    start: Cell,
) -> None:
    if current == goal:
        return
    if current != start:
        print(current.x, current.y)
    updated = update_cell_state(current, move_mode, radius, world_map)
    new_move_mode = move_mode
    can_switch = switch_move_mode(new_move_mode) & current.move_mode
    if can_switch:
        new_move_mode = switch_move_mode(new_move_mode)
        print("r" if new_move_mode == WITH_RING else "rr")
        updated += update_cell_state(current, new_move_mode, radius, world_map)
    for cell in updated:
        new_move_mode = try_switch_mode(cell, new_move_mode)
        backtrack(cell, goal, radius, world_map, new_move_mode, start)

    if move_mode != new_move_mode:
        print("r" if move_mode == WITH_RING else "rr")
        read_obstacles(current)
    if current != start:
        print(current.parent.x, current.parent.y)
        read_obstacles(current.parent)


def backtracking(
    start: Tuple[int, int],
    goal: Tuple[int, int],
    radius: int,
    move_mode: int,
) -> Tuple[List[Cell], int]:
    world_map: List[List[Cell]] = [
        [Cell(i, j) for j in range(MAP_SIZE)] for i in range(MAP_SIZE)
    ]

    start_cell = world_map[start[0]][start[1]]
    goal_cell = world_map[goal[0]][goal[1]]

    start_cell.cost = 0
    start_cell.move_mode = WITH_RING | WITHOUT_RING

    backtrack(start_cell, goal_cell, radius, world_map, move_mode, start_cell)
    path = get_path(goal_cell)
    move_mode = check_and_go(start_cell, goal_cell, move_mode)
    move_mode = try_switch_mode(goal_cell, move_mode)
    return path, move_mode


def main():
    perception_radius = int(input())
    gollum_position = tuple(map(int, input().split()))
    answer = -1
    first_part: List[Cell] = []
    first_part, current_move_mode = backtracking(
        (0, 0), gollum_position, perception_radius, WITHOUT_RING
    )
    if len(first_part) > 1:
        print(*gollum_position)
        read_obstacles(first_part[-1])
        mount_position = tuple(map(int, input().split()[-2:]))
        print(first_part[-1].parent.x, first_part[-1].parent.y)
        read_obstacles(first_part[-1])
        print(*gollum_position)
        second_part: List[Cell] = []
        second_part, current_move_mode = backtracking(
            gollum_position,
            mount_position,
            perception_radius,
            current_move_mode,
        )
        if len(second_part) > 1:
            answer = second_part[-1].cost + first_part[-1].cost

    print("e", answer)
    print(*[(cell.x, cell.y) for cell in first_part[:-1] + second_part])


main()
