import random
from collections import deque

def is_grid_connected(grid):
    rows = len(grid)
    cols = len(grid[0])
    
    start = None
    for r in range(rows):
        for c in range(cols):
            if grid[r][c] == 0:
                start = (r, c)
                break
        if start:
            break

    if not start:
        return False
    
    visited = set([start])
    queue = deque([start])
    
    directions = [(-1, 0), (1, 0), (0, -1), (0, 1)]
    
    while queue:
        r, c = queue.popleft()
        for dr, dc in directions:
            nr, nc = r + dr, c + dc
            if 0 <= nr < rows and 0 <= nc < cols:
                if grid[nr][nc] == 0 and (nr, nc) not in visited:
                    visited.add((nr, nc))
                    queue.append((nr, nc))
    
    for r in range(rows):
        for c in range(cols):
            if grid[r][c] == 0 and (r, c) not in visited:
                return False
    return True

def generate_connected_grid(rows, cols, wall_probability=0.3):
    grid = []
    for r in range(rows):
        row_data = []
        for c in range(cols):
            if r == 0 or r == rows - 1  or r == rows - 2 or c == 0 or c == cols - 1:
                row_data.append(1)
            else:
                row_data.append(0)
        grid.append(row_data)

    interior_cells = [
        (r, c) 
        for r in range(1, rows - 2) 
        for c in range(1, cols - 1)
    ]
    
    random.shuffle(interior_cells)
    
    for (r, c) in interior_cells:
        if random.random() < wall_probability:
            grid[r][c] = 1
            grid[r+1][c] = 1
            
            if not is_grid_connected(grid):
                grid[r][c] = 0
                grid[r+1][c] = 0
    
    return grid

if __name__ == "__main__":
    rows, cols = 10, 10
    grid = generate_connected_grid(rows, cols, wall_probability=0.3)
    
    for row in grid:
        print(row)
