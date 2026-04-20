import heapq
import time

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

def add_obstacle(x1, y1):
    """Add rectangular obstacle region from (x1, y1) to (x2, y2)"""
    x2 = x1 + 5
    y2 = y1 + 5
    for x in range(x1, x2):
        for y in range(y1, y2):
            dynamic_obstacles.add((x, y))


grid_size = 45            
start_point = (0, 0)     
target_point = (32, 31)    

dynamic_obstacles = {(10, 0)
    # (1, 0), (4, 0), (8, 0),
    # (6, 1),
    # (0, 2), (3, 2),
    # (2, 3), (5, 3), (7, 3), (8, 3), 
    # (0, 4), (3, 4),
    # (6, 5), (7, 5), (5, 5), 
    # (1, 6), (5, 6), (7, 6), 
    # (3, 7), (5, 7), (7, 7),
    # (0, 8)
}

obstacles = {
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

for i in obstacles:
    x = i[0] * 5
    y = i[1] * 5
    add_obstacle(x, y)

isDynamic = False

if isDynamic:
    # 1. Initial Plan
    print("STAGE 1: INITIAL A* CALCULATION")
    start_time = time.perf_counter()
    initial_path = astar_pathfinding(grid_size, start_point, target_point, dynamic_obstacles)
    end_time = time.perf_counter()
    total_planning_time = end_time - start_time
    print_environment(grid_size, start_point, target_point, dynamic_obstacles, initial_path, f"Initial Path Planned (Time: {total_planning_time:.5f}s)")

    # 2. Agent moves a few steps along the path
    current_agent_index = 26
    # current_pos = initial_path[current_agent_index]
    # current_pos = (9, 9)
    current_pos = (8, 18)
    print("STAGE 2: AGENT STARTS MOVING")
    print_environment(grid_size, current_pos, target_point, dynamic_obstacles, initial_path[current_agent_index:], f"Agent moved {current_agent_index} steps. Path looks clear.")

    # 3. Dynamic Obstacle Appears!
    print("STAGE 3: DYNAMIC OBSTACLE APPEARS!")
    # Place an obstacle directly in the agent's upcoming path
    # new_obstacle = initial_path[current_agent_index + 2] 
    # dynamic_obstacles.add(new_obstacle)
    rand_obstacle = (5, 20) 
    add_obstacle(rand_obstacle[0], rand_obstacle[1])
    # new_obstacle = dynamic_obstacles
    print(f"🚨 ALERT: New obstacle randomly generated at {rand_obstacle}! Path is blocked.")
    print_environment(grid_size, current_pos, target_point, dynamic_obstacles, initial_path[current_agent_index:], "Old path is now invalid.")

    # 4. A* must REPLAN from scratch
    print("STAGE 4: A* FORCED TO REPLAN ENTIRE ROUTE")
    replan_start_time = time.perf_counter()
    new_path = astar_pathfinding(grid_size, current_pos, target_point, dynamic_obstacles)
    replan_end_time = time.perf_counter()
    total_planning_time += (replan_end_time - replan_start_time)

    print_environment(grid_size, current_pos, target_point, dynamic_obstacles, new_path, f"New Path Calculated (Replanning Time: {replan_end_time - replan_start_time:.5f}s)")

    print("SUMMARY OF A* IN DYNAMIC ENVIRONMENT")
    print(f"Total time spent calculating/recalculating: {total_planning_time:.5f} seconds")

else:    
    # Start the timer
    start_time = time.perf_counter()

    # Run the algorithm
    calculated_path = astar_pathfinding(grid_size, start_point, target_point, dynamic_obstacles)

    # Stop the timer
    end_time = time.perf_counter()
    time_elapsed = end_time - start_time

    # Print the visual output
    print_environment(grid_size, start_point, target_point, dynamic_obstacles, calculated_path, time_elapsed)
    print("SUMMARY OF A* IN STATIC ENVIRONMENT")
    print(f"Total time spent calculating/recalculating: {time_elapsed:.5f} seconds")