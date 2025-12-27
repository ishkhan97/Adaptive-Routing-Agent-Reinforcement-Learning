from collections import deque


def shortest_path_length(
    grid_size,
    start,
    goal,
    obstacles,
):
    """
    Compute shortest path length using BFS.
    Returns number of steps or None if unreachable.
    """

    queue = deque()
    queue.append((start, 0))
    visited = set([start])

    moves = [(0, 1), (0, -1), (1, 0), (-1, 0)]

    while queue:
        (x, y), dist = queue.popleft()

        if (x, y) == goal:
            return dist

        for dx, dy in moves:
            nx, ny = x + dx, y + dy

            if (
                0 <= nx < grid_size
                and 0 <= ny < grid_size
                and (nx, ny) not in obstacles
                and (nx, ny) not in visited
            ):
                visited.add((nx, ny))
                queue.append(((nx, ny), dist + 1))

    return None
