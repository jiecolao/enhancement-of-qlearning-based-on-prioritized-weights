from dataclasses import dataclass

@dataclass
class EQLBPWAgent:
    state_dim: int          = 29
    action_dim: int         = 4
    learning_rate: float    = 0.0005
    gamma: float            = 0.95
    priority_alpha: float   = 0.6
    beta_start: float       = 0.4
    beta_end: float         = 1.0
    e: float                = 1.0
    e_min: float            = 0.05
    e_decay: float          = 0.995
    batch_size: float       = 64
    max_buffer: float       = 2000
    target_sync_freq: float = 20
    collision_weight: float = 1.0
    goal_weight: float      = 2.0
    distance_weight: float  = 0.5

@dataclass
class QLBPWAgent:
    alpha: float            = 0.1 
    gamma: float            = 0.9 
    beta: float             = 0.3
    e: float                = 0.9 
    no_of_actions: float    = 4
    max_buffer: int         = 2000
    batch_size: int         = 20

@dataclass
class EQLBPWTestAgent:
    state_dim: int          = 29
    no_of_actions: int      = 4
    learning_rate: float    = 0.0005
    gamma: float            = 0.95
    priority_alpha: float   = 0.6
    beta_start: float       = 0.4
    beta_end: float         = 1.0
    e: float                = 1.0
    e_min: float            = 0.05
    e_decay: float          = 0.995
    max_buffer: int         = 50000
    batch_size: int         = 64
    target_sync_freq: int   = 20

@dataclass
class QLBPWTestAgent:
    alpha: float            = 0.1 
    gamma: float            = 0.9 
    beta: float             = 0.3
    e: float                = 0.9 
    no_of_actions: float    = 4
    max_buffer: int         = 2000
    batch_size: int         = 20