import heapq
import time
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import numpy as np
import os

class Node:
    def __init__(self, position, parent=None):
        self.position = position
        self.parent = parent
        self.g = 0  
        self.h = 0  
        self.f = 0  

    def __lt__(self, other):
        return self.f < other.f

def astar_pathfinding(grid_size, start, goal, obstacles):
    open_list = []
    closed_set = set()

    start_node = Node(start)
    goal_node = Node(goal)

    heapq.heappush(open_list, start_node)
    directions = [(0, 1), (0, -1), (1, 0), (-1, 0)]

    while open_list:
        current_node = heapq.heappop(open_list)
        closed_set.add(current_node.position)

        if current_node.position == goal_node.position:
            path = []
            while current_node:
                path.append(current_node.position)
                current_node = current_node.parent
            return path[::-1] 

        for dx, dy in directions:
            node_position = (current_node.position[0] + dx, current_node.position[1] + dy)

            if (node_position[0] > (grid_size - 1) or node_position[0] < 0 or 
                node_position[1] > (grid_size - 1) or node_position[1] < 0):
                continue

            if node_position in obstacles:
                continue

            new_node = Node(node_position, current_node)

            if new_node.position in closed_set:
                continue

            new_node.g = current_node.g + 1
            new_node.h = abs(new_node.position[0] - goal_node.position[0]) + abs(new_node.position[1] - goal_node.position[1])
            new_node.f = new_node.g + new_node.h

            if any(open_node for open_node in open_list if new_node.position == open_node.position and new_node.g > open_node.g):
                continue

            heapq.heappush(open_list, new_node)

    return None

def print_environment(grid_size, current_pos, goal, obstacles, path, step_msg):
    print(f"--- {step_msg} ---")
    path_set = set(path) if path else set()
    
    for y in range(grid_size):
        row_str = ""
        for x in range(grid_size):
            # print(x)
            pos = (x, y)
            if pos == current_pos:
                row_str += "🤖"
            elif pos == goal:
                row_str += "🏁"
            elif pos in obstacles:
                row_str += "⬛"
            elif pos in path_set:
                row_str += "🟢"
            else:
                row_str += ". "
        print(f"{y}: \t{row_str}")
    print("========================================\n")

def visualize_grid(filename, grid_size, current_pos, goal, obstacles, path, title="Grid Visualization"):
    """Visualize the grid using Matplotlib"""
    fig, ax = plt.subplots(figsize=(10, 10))
    
    # Create grid background
    ax.set_xlim(-0.5, grid_size - 0.5)
    ax.set_ylim(grid_size - 0.5, -0.5)  # Inverted Y-axis
    ax.set_aspect('equal')
    ax.grid(True, alpha=0.3)
    
    # Draw obstacles
    for obs in obstacles:
        rect = patches.Rectangle((obs[0] - 0.5, obs[1] - 0.5), 1, 1, 
                                linewidth=1, edgecolor='black', facecolor='black')
        ax.add_patch(rect)
    
    # Draw path
    if path:
        path_x = [p[0] for p in path]
        path_y = [p[1] for p in path]
        ax.plot(path_x, path_y, 'g-', linewidth=2, alpha=0.6, label='Path')
        ax.scatter(path_x, path_y, c='green', s=20, alpha=0.5)
    
    # Draw goal
    ax.scatter(*goal, c='red', s=300, marker='*', label='Goal', zorder=5)
    
    # Draw current position
    ax.scatter(*current_pos, c='blue', s=200, marker='o', label='Agent', zorder=5)
    
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_title(title)
    ax.legend()
    plt.tight_layout()
    
    # Create figures directory if it doesn't exist
    figures_dir = os.path.join(os.path.dirname(__file__), 'figures')
    os.makedirs(figures_dir, exist_ok=True)
    
    filepath = os.path.join(figures_dir, filename)
    
    # Save figure
    # plt.savefig(filepath, dpi=100, bbox_inches='tight')
    # print(f"Figure saved to: {filepath}")
    plt.show()

def add_obstacle(x1, y1, cells):
    """Add rectangular obstacle region from (x1, y1) to (x2, y2)"""
    x2 = x1 + cells
    y2 = y1 + cells
    for x in range(x1, x2):
        for y in range(y1, y2):
            dynamic_obstacles.add((x, y))

def run_astar_test(config, is_dynamic=True, withVisual=False, runs=5):
    """Run A* pathfinding test with given configuration"""
    global dynamic_obstacles
    
    grid_size = config['grid_size']
    start_point = config['start_point']
    target_point = config['target_point']
    cells = config['cells']
    name = config['name']
    
    print(f"\n{'='*50}")
    print(f"Testing: {name} ({grid_size}x{grid_size})")
    print(f"{'='*50}\n")

    total_times = []

    for run_index in range(runs):
        # Reset obstacles for each run
        dynamic_obstacles = {(-1, -1)}

        # Add scaled obstacles
        for obs in config['base_obstacles']:
            x = obs[0] * cells
            y = obs[1] * cells
            add_obstacle(x, y, cells)

        if is_dynamic:
            # 1. Initial Plan
            print("STAGE 1: START A* CALCULATION")
            start_time = time.perf_counter()
            initial_path = astar_pathfinding(grid_size, start_point, target_point, dynamic_obstacles)

            # 2. Agent moves
            current_agent_index = min(config['reroute'], len(initial_path) - 1) if initial_path else 0
            current_pos = initial_path[current_agent_index] if initial_path else start_point
            print("STAGE 2: AGENT STARTS MOVING")
            print(f"\tAgent moved {current_agent_index} steps")

            # 3. Dynamic Obstacle Appears
            print("STAGE 3: DYNAMIC OBSTACLE APPEARS!")
            rand_obstacle = (1, 4)
            add_obstacle(rand_obstacle[0] * cells, rand_obstacle[1] * cells, cells)
            print(f"\tNew obstacle at {rand_obstacle}!")
            print("\tOld path invalid")

            # 4. Replan
            print("STAGE 4: A* FORCED TO REPLAN")
            print("\tCalculationg Replanned Time...")
            new_path = astar_pathfinding(grid_size, current_pos, target_point, dynamic_obstacles)

            end_time = time.perf_counter()
            total_planning_time = end_time - start_time
            total_times.append(total_planning_time)
            print(f"\tInitial Path (Time: {total_planning_time:.5f}s)")
            if withVisual and run_index == runs - 1:
                visualize_grid(f"Astar_{name}_dynamic_1.png", grid_size, start_point, target_point, dynamic_obstacles, new_path, 
                               f"{name} - Path (Time: {total_planning_time:.5f}s)")
        else:
            start_time = time.perf_counter()
            calculated_path = astar_pathfinding(grid_size, start_point, target_point, dynamic_obstacles)
            end_time = time.perf_counter()
            time_elapsed = end_time - start_time
            total_times.append(time_elapsed)
            print(f"{name} - A* Pathfinding (Time: {time_elapsed:.5f}s)")
            if withVisual and run_index == runs - 1:
                visualize_grid(f"Astar_{name}.png", grid_size, start_point, target_point, dynamic_obstacles, calculated_path, 
                               f"{name} - A* Pathfinding (Average Time: {time_elapsed:.5f}s)")

    average_time = sum(total_times) / len(total_times) if total_times else 0.0
    print(f"Average Total Time ({runs} runs): {average_time:.5f}s\n")


# Define test configurations
BASE_OBSTACLES = {
    (1, 0), (4, 0), (8, 0),
    (6, 1),
    (0, 2), (3, 2),
    (2, 3), (5, 3), (7, 3), (8, 3), 
    (0, 4), (3, 4),
    (6, 5), (7, 5), (5, 5), 
    (1, 6), (5, 6), (7, 6), 
    (3, 7), (5, 7), (7, 7),
    (0, 8)
}

test_configs = [
    {
        'name': '9x9', # 9 * 5
        'grid_size': 729,
        'start_point': (0, 0),
        'target_point': (32, 31), # Total Time (Charged): 0.09558s
        'cells': 81, # 5,
        'reroute': 26,
        'base_obstacles': BASE_OBSTACLES
    },
    {
        'name': '9x9', # 9 * 5
        'grid_size': 45,
        'start_point': (0, 0),
        'target_point': (32, 31), # Total Time (Charged): 0.09558s
        'cells': 5, # 5,
        'reroute': 26,
        'base_obstacles': BASE_OBSTACLES
    },
    {
        'name': '10x10', # 10 * 10
        'grid_size': 100,
        'start_point': (0, 0),
        'target_point': (71, 75), # Total Time (Charged): 12 sec
        'cells': 11,
        'reroute': 60,
        'base_obstacles': BASE_OBSTACLES
    },
    {
        'name': '15x15', # 15x15
        'grid_size': 135,
        'start_point': (0, 0),
        'target_point': (97, 107), # Total Time (Charged): 50.09718s
        'cells': 15,
        'reroute': 85,
        'base_obstacles': BASE_OBSTACLES
    },
    {
        'name': '20x20', # 20x20
        'grid_size': 180,
        'start_point': (0, 0),
        'target_point': (130, 146), # Total Time (Charged): 283.35224s
        'cells': 20,
        'reroute': 110,
        'base_obstacles': BASE_OBSTACLES
    }
]

# Run tests
run_astar_test(test_configs[0], is_dynamic=False, withVisual=True, runs=1)
# for config in test_configs:
#     run_astar_test(config, is_dynamic=False, withVisual=True)