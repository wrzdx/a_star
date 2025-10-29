from typing import List, Optional, Tuple
from heapq import heappush, heappop

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
        visited: bool = False,
        parent: Optional["Cell"] = None,
        token: str = "",
    ):
        self.x, self.y = (x, y)
        self.cost = cost
        self.move_mode = move_mode
        self.parent = parent
        self.visited = visited
        self.token = token

    def __eq__(self, other: Optional["Cell"]):
        return other and other.x == self.x and other.y == self.y


class Heap:
    def __init__(self, goal: Cell):
        self.heap: List[Tuple[int, Cell]] = []
        self.goal: Cell = goal
        self.size = 0

    def push(self, cell: Cell) -> None:
        self.size += 1
        heappush(self.heap, (self.get_priority(cell), cell))

    def pop(self) -> Cell:
        self.size -= 1
        return heappop(self.heap)[1]

    def get_priority(self, cell: Cell) -> Tuple[int, int]:
        manhattan = abs(cell.x - self.goal.x) + abs(cell.y - self.goal.y)
        return (
            cell.cost + manhattan,
            manhattan,
            cell.token != "C",  # give priority to cell with coat
            cell.x,
            cell.y,
        )


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
    heap: Heap,
    current_cell: Cell,
    allowed_cells: List[Cell],
    move_mode: int,
) -> None:
    cells_to_move = [
        (current_cell.x + 1, current_cell.y),
        (current_cell.x, current_cell.y + 1),
        (current_cell.x - 1, current_cell.y),
        (current_cell.x, current_cell.y - 1),
    ]
    for cell in allowed_cells:
        cell.move_mode |= move_mode
        if (cell.x, cell.y) in cells_to_move:
            new_cost = current_cell.cost + 1
            if new_cost < cell.cost:
                cell.cost = new_cost
                cell.parent = current_cell
                heap.push(cell)


def switch_move_mode(move_mode: int) -> int:
    return WITH_RING if move_mode == WITHOUT_RING else WITHOUT_RING


def update_cell_state(
    heap: Heap,
    cell: Cell,
    move_mode: int,
    radius: int,
    world_map: List[List[Cell]],
) -> None:
    allowed_moves = read_obstacles(cell, world_map, radius)
    allowed_cells = [world_map[x][y] for x, y in allowed_moves]
    update_neighbours(heap, cell, allowed_cells, move_mode)


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


def a_star(
    start: Tuple[int, int],
    goal: Tuple[int, int],
    radius: int,
    move_mode: int = WITHOUT_RING,
) -> Tuple[List[Cell], int]:
    world_map: List[List[Cell]] = [
        [Cell(i, j) for j in range(MAP_SIZE)] for i in range(MAP_SIZE)
    ]
    current_cell: Cell = world_map[start[0]][start[1]]
    current_cell.cost = 0
    current_cell.move_mode = WITH_RING | WITHOUT_RING
    current_cell.visited = True
    current_move_mode = move_mode
    goal_cell: Cell = world_map[goal[0]][goal[1]]
    heap: Heap = Heap(goal_cell)
    update_cell_state(heap, current_cell, current_move_mode, radius, world_map)
    can_switch = switch_move_mode(current_move_mode) & current_cell.move_mode
    if can_switch:
        current_move_mode = switch_move_mode(current_move_mode)
        print("r" if current_move_mode == WITH_RING else "rr")
        update_cell_state(heap, current_cell, current_move_mode, radius, world_map)

    while heap.size:
        next_cell = heap.pop()
        if next_cell.visited:
            continue
        current_move_mode = check_and_go(current_cell, next_cell, current_move_mode)
        current_cell = next_cell
        current_move_mode = try_switch_mode(current_cell, current_move_mode)
        current_cell.visited = True
        if current_cell == goal_cell:
            break
        print(current_cell.x, current_cell.y)

        update_cell_state(heap, current_cell, current_move_mode, radius, world_map)
        can_switch = switch_move_mode(current_move_mode) & current_cell.move_mode
        if can_switch:
            current_move_mode = switch_move_mode(current_move_mode)
            print("r" if current_move_mode == WITH_RING else "rr")
            update_cell_state(heap, current_cell, current_move_mode, radius, world_map)

    return get_path(goal_cell), current_move_mode


def main():
    perception_radius = int(input())
    gollum_position = tuple(map(int, input().split()))
    answer = -1
    first_part: List[Cell] = []
    second_part: List[Cell] = []
    first_part, current_move_mode = a_star((0, 0), gollum_position, perception_radius)
    if len(first_part) > 1:
        print(*gollum_position)
        read_obstacles(first_part[-1])
        mount_position = tuple(map(int, input().split()[-2:]))
        print(first_part[-1].parent.x, first_part[-1].parent.y)
        read_obstacles(first_part[-1])
        print(*gollum_position)
        second_part, current_move_mode = a_star(
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
