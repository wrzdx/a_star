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
    ):
        self.x, self.y = (x, y)
        self.cost = cost
        self.move_mode = move_mode
        self.parent = parent
        self.visited = visited

    def __eq__(self, other: Optional["Cell"]):
        return other.x == self.x and other.y == self.x

    def reset(self, x, y):
        self.x, self.y = x, y
        self.cost = INF
        self.visited = False
        self.parent = None
        self.move_mode = IMPOSSIBLE


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
            cell.token == "C",  # give priority to cell with coat
            cell.x,
            cell.y,
        )


def read_obstacles(current_cell: Cell) -> List[Tuple[int, int]]:
    p = int(input())
    allowed_moves = [
        (current_cell.x + 1, current_cell.y),  # Down
        (current_cell.x, current_cell.y + 1),  # Right
        (current_cell.x - 1, current_cell.y),  # Up
        (current_cell.x, current_cell.y + 1),  # Left
    ]
    for i in range(p):
        x, y, token = input().split()
        x, y = map(int, [x, y])
        if token in DANGEROUS_TOKENS:
            allowed_moves.remove((x, y))
    return allowed_moves


def update_neighbours(
    heap: Heap,
    current_cell: Cell,
    allowed_cells: List[Cell],
    move_mode: int,
) -> None:
    for cell in allowed_cells:
        new_cost = current_cell.cost + 1
        if new_cost < cell.cost:
            cell.cost = new_cost
            cell.parent = current_cell
            cell.move_mode = move_mode
            heap.push(cell)
        elif new_cost == cell.cost and cell.parent == current_cell:
            cell.move_mode |= move_mode


def switch_move_mode(move_mode: int) -> int:
    return WITH_RING if move_mode == WITHOUT_RING else WITHOUT_RING


def update_cell_state(
    heap: Heap, cell: Cell, move_mode: int, world_map: List[List[Cell]]
) -> None:
    allowed_moves = read_obstacles(cell)
    allowed_cells = [world_map[x][y] for x, y in allowed_moves]
    update_neighbours(heap, cell, allowed_cells, move_mode)



def try_switch_mode(cell: Cell, current_move_mode: int) -> bool:
    can_switch = switch_move_mode(current_move_mode) & cell.move_mode
    if can_switch:
        new_mode = switch_move_mode(current_move_mode)
        print("r" if new_mode == WITH_RING else "rr")
        return True
    return False


def a_star(start: Tuple[int, int], goal: Tuple[int, int]) -> List[Cell]:
    world_map: List[List[Cell]] = [
        Cell(i, j) for i in range(MAP_SIZE) for j in range(MAP_SIZE)
    ]
    world_map[start[0]][start[1]] = Cell(
        0, 0, 0, WITH_RING | WITHOUT_RING, True
    )
    current_cell: Cell = world_map[start[0]][start[1]]
    current_move_mode = WITHOUT_RING
    goal_cell: Cell = world_map[goal[0]][goal[1]]
    heap: Heap = Heap(goal_cell)
    update_cell_state(heap, current_cell, current_move_mode, world_map)
    if try_switch_mode(heap, current_cell, current_move_mode):
        current_move_mode = switch_move_mode(current_move_mode)
        update_cell_state(heap, current_cell, current_move_mode, world_map)

    while heap.size and current_cell != goal_cell:
        current_cell = heap.pop()
        if current_cell.visited:
            continue
        current_cell.visited = True
        print(current_cell.x, current_cell.y)

        update_cell_state(heap, current_cell, current_move_mode)
        if try_switch_mode(heap, current_cell, current_move_mode):
            current_move_mode = switch_move_mode(current_move_mode)
            update_cell_state(heap, current_cell, current_move_mode, world_map)
    
    path: List[Cell] = [goal_cell]
    next_cell: Cell = goal_cell.parent
    while next_cell:
        path.append(next_cell)
        next_cell = next_cell.parent
    
    return path[::-1]


def main():
    perception_radius = int(input())
    gollum_position = tuple(map(int, input().split()))
    answer: int = -1
    first_part: List[Cell] = a_star((0,0), gollum_position)

    if len(first_part) > 1:
        print(*gollum_position)
        read_obstacles(first_part[-1])
        mount_position = tuple(map(int, input().split()[-2:]))
        print(first_part[-1].parent.x, first_part[-1].parent.y)
        read_obstacles(first_part[-1])
        print(*gollum_position)
        second_part: List[Cell] = a_star((gollum_position, mount_position))
        if len(second_part) > 1:
            answer = second_part[-1].cost
    
    print("e", answer)

        

if __name__ == "__main__":
    main()
