import numpy as np
import time
import matplotlib.pyplot as plt

def simulate_q_learning(grid_sizes=[5, 10, 20, 40, 80, 100]):
    """
    Simulate Q-Learning and visualize memory and execution time scaling.
    """
    print("Simulating Q-Learning complexity scaling...")
    print(f"{'Grid Size':<12} | {'Total States':<12} | {'Q-Table Mem (Bytes)':<20} | {'Execution Time (s)':<18}")
    print("-" * 70)

    alpha = 0.1
    gamma = 0.9
    epsilon = 0.1
    episodes = 200

    grid_values = []
    memory_values = []
    time_values = []

    for size in grid_sizes:
        q_table = np.zeros((size, size, 4))
        mem_size_bytes = q_table.nbytes
        
        max_steps = size * size * 2 
        
        start_time = time.time()
        
        for _ in range(episodes):
            state_x, state_y = 0, 0
            goal_x, goal_y = size - 1, size - 1
            
            for _ in range(max_steps):
                if np.random.rand() < epsilon:
                    action = np.random.randint(4)
                else:
                    action = np.argmax(q_table[state_x, state_y])
                    
                next_x, next_y = state_x, state_y
                if action == 0 and state_x > 0:          next_x -= 1
                elif action == 1 and state_x < size - 1: next_x += 1
                elif action == 2 and state_y > 0:        next_y -= 1
                elif action == 3 and state_y < size - 1: next_y += 1
                
                if next_x == goal_x and next_y == goal_y:
                    reward = 100
                    done = True
                else:
                    reward = -1
                    done = False
                    
                best_next_action = np.argmax(q_table[next_x, next_y])
                td_target = reward + gamma * q_table[next_x, next_y, best_next_action]
                td_error = td_target - q_table[state_x, state_y, action]
                q_table[state_x, state_y, action] += alpha * td_error
                
                state_x, state_y = next_x, next_y
                
                if done:
                    break
                    
        elapsed_time = time.time() - start_time
        
        # Store metrics for plotting
        grid_values.append(size)
        memory_values.append(mem_size_bytes / (1024**2))  # Convert to MB
        time_values.append(elapsed_time)
        
        print(f"{size}x{size:<10} | {size**2:<12} | {mem_size_bytes:<20,} | {elapsed_time:.4f}")

    fig, axes = plt.subplots(1, 3, figsize=(15, 4))
    
    # Plot 1: Memory Consumption vs Grid Size
    axes[0].plot(grid_values, memory_values, 'b-o', linewidth=2, markersize=6)
    axes[0].set_xlabel('Grid Size', fontsize=12)
    axes[0].set_ylabel('Memory (MB)', fontsize=12)
    axes[0].set_title('Memory Consumption vs Grid Size', fontsize=13, fontweight='bold')
    axes[0].grid(True, alpha=0.3)
    
    # Plot 2: Execution Time vs Grid Size
    axes[1].plot(grid_values, time_values, 'r-s', linewidth=2, markersize=6)
    axes[1].set_xlabel('Grid Size', fontsize=12)
    axes[1].set_ylabel('Execution Time (s)', fontsize=12)
    axes[1].set_title('Execution Time vs Grid Size', fontsize=13, fontweight='bold')
    axes[1].grid(True, alpha=0.3)
    
    # Plot 3: Both on same graph with dual y-axis
    ax3_1 = axes[2]
    ax3_2 = ax3_1.twinx()
    
    line1 = ax3_1.plot(grid_values, memory_values, 'b-o', linewidth=2, markersize=6, label='Memory (MB)')
    line2 = ax3_2.plot(grid_values, time_values, 'r-s', linewidth=2, markersize=6, label='Time (s)')
    
    ax3_1.set_xlabel('Grid Size', fontsize=12)
    ax3_1.set_ylabel('Memory (MB)', color='b', fontsize=12)
    ax3_2.set_ylabel('Execution Time (s)', color='r', fontsize=12)
    ax3_1.tick_params(axis='y', labelcolor='b')
    ax3_2.tick_params(axis='y', labelcolor='r')
    ax3_1.set_title('Memory & Time vs Grid Size', fontsize=13, fontweight='bold')
    ax3_1.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    simulate_q_learning()