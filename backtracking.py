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
    """
    Return list of allowed moves around the current cell, considering obstacles.
    If world_map is provided, update the tokens of the cells."""
    p = int(input())
    allowed_moves = []
    # Generate all possible moves in the given radius
    for x in range(current_cell.x - radius, current_cell.x + radius + 1):
        if x < 0 or x >= MAP_SIZE:
            continue
        for y in range(current_cell.y - radius, current_cell.y + radius + 1):
            if y < 0 or y >= MAP_SIZE or (x == current_cell.x and y == current_cell.y):
                continue
            allowed_moves.append((x, y))

    # Read obstacles and update allowed moves
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
    """
    Update the neighbours of the current cell ability to move and
    if it's possible to move try to update cost. Return list of updated cells."""
    # Cells where we can move
    cells_to_move = [
        (current_cell.x + 1, current_cell.y),
        (current_cell.x, current_cell.y + 1),
        (current_cell.x - 1, current_cell.y),
        (current_cell.x, current_cell.y - 1),
    ]
    updated = []
    for cell in allowed_cells:
        # Update move mode
        cell.move_mode |= move_mode
        # If the cell is in the possible move positions
        if (cell.x, cell.y) in cells_to_move:
            # Try to update cost and push to heap
            new_cost = current_cell.cost + 1
            if new_cost < cell.cost:
                cell.cost = new_cost
                cell.parent = current_cell
                updated.append(cell)

    return updated


def switch_move_mode(move_mode: int) -> int:
    """Return the opposite move mode."""
    return WITH_RING if move_mode == WITHOUT_RING else WITHOUT_RING


def update_cell_state(
    cell: Cell,
    move_mode: int,
    radius: int,
    world_map: List[List[Cell]],
) -> List[Cell]:
    """Update the state of the cell's neighbours based on allowed moves. 
    Return list of updated cells."""
    allowed_moves = read_obstacles(cell, world_map, radius)
    allowed_cells = [world_map[x][y] for x, y in allowed_moves]
    return update_neighbours(cell, allowed_cells, move_mode)


def try_switch_mode(cell: Cell, current_move_mode: int) -> int:
    """If we can't move in the current mode, switch to the other mode.
    Work only with parent and child relationship. Return the current move mode."""
    if not (cell.move_mode & current_move_mode):
        current_move_mode = switch_move_mode(current_move_mode)
        print("r" if current_move_mode == WITH_RING else "rr", flush=True)
        read_obstacles(cell)
    return current_move_mode


def get_path(cell: Cell) -> List[Cell]:
    """Return the path from start to the given cell."""
    path: List[Cell] = [cell]
    next_cell: Cell = cell.parent
    while next_cell:
        path.append(next_cell)
        next_cell = next_cell.parent

    return path[::-1]


def check_and_go(current: Cell, next_cell: Cell, current_move_mode: int) -> int:
    """If the next cell is not a direct neighbour of the current cell, 
    move back to the common ancestor and then move forward to parent of the next cell. 
    Return the updated current move mode.
    """
    if current != next_cell.parent:
        while current.parent:
            current_move_mode = try_switch_mode(current.parent, current_move_mode)
            current = current.parent
            print("m", current.x, current.y, flush=True)
            read_obstacles(current)

        path = get_path(next_cell)[1:-1]
        for cell in path:
            current_move_mode = try_switch_mode(cell, current_move_mode)
            print("m", cell.x, cell.y, flush=True)
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
    """Backtracking algorithm implementation."""
    # Base case: reached the goal
    if current == goal:
        return
    # Move to the current cell if it's not the start
    if current != start:
        print("m", current.x, current.y, flush=True)
    # Update cell state and get updated cells
    updated = update_cell_state(current, move_mode, radius, world_map)
    # Do not change initial move mode for backtracking
    new_move_mode = move_mode
    can_switch = switch_move_mode(new_move_mode) & current.move_mode
    # Try switching move mode if possible to add more move rights to cells
    if can_switch:
        new_move_mode = switch_move_mode(new_move_mode)
        print("r" if new_move_mode == WITH_RING else "rr", flush=True)
        updated += update_cell_state(current, new_move_mode, radius, world_map)

    # Recur for all updated cells
    for cell in updated:
        new_move_mode = try_switch_mode(cell, new_move_mode)
        backtrack(cell, goal, radius, world_map, new_move_mode, start)

    # Return everything to the previous state
    if move_mode != new_move_mode:
        print("r" if move_mode == WITH_RING else "rr", flush=True)
        read_obstacles(current)
    if current != start:
        print("m", current.parent.x, current.parent.y, flush=True)
        read_obstacles(current.parent)


def backtracking(
    start: Tuple[int, int],
    goal: Tuple[int, int],
    radius: int,
    move_mode: int,
    move_mode_of_start: int = WITHOUT_RING | WITH_RING,
) -> Tuple[List[Cell], int]:
    """Backtracking algorithm main function. 
    Return the path from start to goal and the current move mode."""
    # Initialize the world map
    world_map: List[List[Cell]] = [
        [Cell(i, j) for j in range(MAP_SIZE)] for i in range(MAP_SIZE)
    ]
    # Initialize start and goal cells
    start_cell = world_map[start[0]][start[1]]
    goal_cell = world_map[goal[0]][goal[1]]

    start_cell.cost = 0
    start_cell.move_mode = move_mode_of_start
    # Start backtracking
    backtrack(start_cell, goal_cell, radius, world_map, move_mode, start_cell)
    path = get_path(goal_cell)
    # Follow the path to the goal
    move_mode = check_and_go(start_cell, goal_cell, move_mode)
    move_mode = try_switch_mode(goal_cell, move_mode)
    return path, move_mode


def main():
    # Read perception radius and Gollum's position
    perception_radius = int(input())
    gollum_position = tuple(map(int, input().split()))
    # Initialize answer impossible
    answer = -1
    # Initialize path from 0,0 to gollum and from gollum to mount
    first_part: List[Cell] = []
    second_part: List[Cell] = []
    # Find path to Gollum
    first_part, current_move_mode = backtracking(
        (0, 0), gollum_position, perception_radius, WITHOUT_RING
    )
    # If path to Gollum found, move to Gollum and find path to Mount Doom
    if len(first_part) > 1:
        print("m", *gollum_position, flush=True)
        read_obstacles(first_part[-1])
        # Read Mount Doom position
        mount_position = tuple(map(int, input().split()[-2:]))
        # Go back to parent of Gollum
        print("m", first_part[-1].parent.x, first_part[-1].parent.y, flush=True)
        read_obstacles(first_part[-1])
        # Go to gollum again to get correct info about surroundings
        print("m", *gollum_position, flush=True)
        # Find path to Mount Doom
        second_part, current_move_mode = backtracking(
            gollum_position,
            mount_position,
            perception_radius,
            current_move_mode,
            first_part[-1].move_mode,
        )
        # If path to Mount Doom found, calculate answer
        if len(second_part) > 1:
            print("m", *mount_position)
            answer = second_part[-1].cost + first_part[-1].cost
    print("e", answer, flush=True)
    # Print the full path
    print(*[(cell.x, cell.y) for cell in first_part[:-1] + second_part])


main()
