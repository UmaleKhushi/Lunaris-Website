import heapq
import numpy as np
import matplotlib.pyplot as plt
import random
import math

# ---------------- Grid Map ----------------
GRID_SIZE = 20
grid = np.zeros((GRID_SIZE, GRID_SIZE), dtype=int)

start = (0, 0)
goal = (19, 19)

# Place random static obstacles
for _ in range(60):
    x, y = random.randint(0, GRID_SIZE-1), random.randint(0, GRID_SIZE-1)
    if (x, y) != start and (x, y) != goal:
        grid[y, x] = 1  # obstacle

# ---------------- A* Algorithm ----------------
def heuristic(a, b):
    return math.sqrt((a[0]-b[0])**2 + (a[1]-b[1])**2)

def astar(grid, start, goal):
    open_set = []
    heapq.heappush(open_set, (heuristic(start, goal), 0, start, [start]))
    visited = set()
    while open_set:
        _, cost, current, path = heapq.heappop(open_set)
        if current == goal:
            return path
        if current in visited:
            continue
        visited.add(current)
        x, y = current
        for dx, dy in [(-1,0),(1,0),(0,-1),(0,1),(-1,-1),(1,-1),(-1,1),(1,1)]:
            nx, ny = x+dx, y+dy
            if 0 <= nx < GRID_SIZE and 0 <= ny < GRID_SIZE and grid[ny, nx] == 0:
                step_cost = math.sqrt(dx*dx + dy*dy)
                heapq.heappush(open_set, (cost+step_cost+heuristic((nx,ny), goal),
                                          cost+step_cost, (nx,ny), path+[(nx,ny)]))
    return None

# ---------------- Robot Simulation ----------------
path = astar(grid, start, goal)
if path is None:
    print("No path found initially!")
    exit()

robot_pos = start
plt.ion()
fig, ax = plt.subplots()
colors = ["b", "c", "m", "y", "orange", "purple"]
color_idx = 0
all_moves = []  # store all moves
final_path = [robot_pos]  # store the full robot path for final image

while robot_pos != goal:
    ax.clear()
    ax.imshow(grid, cmap='Greys')

    if path:
        px, py = zip(*path)
        ax.plot(px, py, color=colors[color_idx % len(colors)], linewidth=2, label="Path")

    ax.plot(goal[0], goal[1], 'go')  # goal
    ax.set_title("Robot Path Planning with Smooth 8-Way Movement")
    ax.legend(loc="upper left")

    # Move robot smoothly towards next cell
    next_pos = path[1]
    steps = 10
    dx = next_pos[0] - robot_pos[0]
    dy = next_pos[1] - robot_pos[1]

    # Determine move direction
    if dx == 1 and dy == 0: move = "RIGHT"
    elif dx == -1 and dy == 0: move = "LEFT"
    elif dx == 0 and dy == 1: move = "DOWN"
    elif dx == 0 and dy == -1: move = "UP"
    elif dx == 1 and dy == 1: move = "DOWN-RIGHT"
    elif dx == 1 and dy == -1: move = "UP-RIGHT"
    elif dx == -1 and dy == 1: move = "DOWN-LEFT"
    elif dx == -1 and dy == -1: move = "UP-LEFT"
    all_moves.append(move)

    for i in range(steps):
        interp_x = robot_pos[0] + dx * (i+1)/steps
        interp_y = robot_pos[1] + dy * (i+1)/steps
        ax.plot(interp_x, interp_y, 'ro')
        plt.pause(0.05)

    robot_pos = next_pos
    final_path.append(robot_pos)
    path = path[1:]

    # Dynamic obstacles
    if random.random() < 0.2:
        ox, oy = random.randint(0, GRID_SIZE-1), random.randint(0, GRID_SIZE-1)
        if grid[oy, ox] == 0 and (ox, oy) != robot_pos and (ox, oy) != goal:
            grid[oy, ox] = 1
            print(f"⚠️ New obstacle at {(ox, oy)} — recalculating path...")
            path = astar(grid, robot_pos, goal)
            if path is None:
                print("❌ No path available!")
                break
            color_idx += 1

plt.ioff()
plt.show()

# ---------------- Save Moves ----------------
with open("robot_moves.txt","w") as f:
    for m in all_moves:
        f.write(m + "\n")
print("Robot moves saved in robot_moves.txt")

# ---------------- Save Colored Path Image ----------------
path_grid = np.copy(grid)
for x, y in final_path:
    path_grid[y, x] = 0.5  # mark robot path

plt.figure(figsize=(6,6))
plt.imshow(path_grid, cmap='gray_r')
plt.title("Final Robot Path")
plt.axis('off')
plt.savefig("final_colored_path.png")
plt.show()
print("Final colored path saved as final_colored_path.png")
