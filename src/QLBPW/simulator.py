import time
from QLBPW.agent import Agent
from QLBPW.environment import Environment

BASE_OBSTACLES = {
            (1, 0),                 (4, 0),                             (8, 0),
                                                    (6, 1),
    (0, 2),                 (3, 2),
                    (2, 3),                 (5, 3),         (7, 3),     (8, 3), 
    (0, 4),                 (3, 4),
                                            (5, 5), (6, 5), (7, 5), 
            (1, 6),                         (5, 6),         (7, 6), 
                            (3, 7),         (5, 7),         (7, 7),
    (0, 8)
}

environment = [
    {
        'name': '9x9',
        'grid': 9,
        'start': (0, 0),
        'goal': (6, 6),
        'base_obstacles': BASE_OBSTACLES, 
    },
    {
        'name': '10x10',
        'grid': 10,
        'start': (0, 0),
        'goal': (9, 9),
        'base_obstacles': Environment._generate_base_obstacles(
            grid_size=10,
            num_obstacles=20,
            start_state=(0, 0),
            goal_state=(9, 9),
            seed=None
        )
    },
    {
        'name': '15x15',
        'grid': 15,
        'start': (0, 0),
        'goal': (14, 14),
        'base_obstacles': Environment._generate_base_obstacles(
            grid_size=15,
            num_obstacles=30,
            start_state=(0, 0),
            goal_state=(14, 14),
            seed=None
        )
    },
    {
        'name': '20x20',
        'grid': 20,
        'start': (0, 0),
        'goal': (19, 19),
        'base_obstacles': Environment._generate_base_obstacles(
            grid_size=10,
            num_obstacles=40,
            start_state=(0, 0),
            goal_state=(19, 19),
            seed=None
        )
    },
]

agent = Agent(
    episodes=1000,
    alpha=0.1,
    gamma=0.9,
    beta=0.3,
    epsilon=0.9,
    max_buffer=5000
)

test = Environment(
    agent=agent,
    grid=9,
    start=(0, 0),
    goal=(6, 6),
    enable_obs=False,
    # obstacles=[],
    obstacles=BASE_OBSTACLES,
    num_dynamic_obs=0
)


print(f"\nStarting simulation...")
start_time = time.time() 
print("Simulating")
test.simulate()
end_time = time.time() 
elapsed_time = end_time - start_time
print(f"Simulation Done: {elapsed_time}")