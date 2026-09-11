from dataclasses import dataclass, field
from typing import Tuple, Set
from env_settings import PRESET_ENVIRONMENTS

@dataclass
class Intramuros:
    grid: int = 20

    start_state: Tuple[int, int] = PRESET_ENVIRONMENTS[1]["start_state"]["fort_santiago"]
    # start_state: Tuple[int, int] = PRESET_ENVIRONMENTS[1]["start_state"]["manila_cathedral"],
    # start_state: Tuple[int, int] = PRESET_ENVIRONMENTS[1]["start_state"]["enter_exit1"],
    # start_state: Tuple[int, int] = PRESET_ENVIRONMENTS[1]["start_state"]["dlm"],
    # start_state: Tuple[int, int] = PRESET_ENVIRONMENTS[1]["start_state"]["enter_exit2"],
    # start_state: Tuple[int, int] = PRESET_ENVIRONMENTS[1]["start_state"]["san_agustin_church"],
    # start_state: Tuple[int, int] = PRESET_ENVIRONMENTS[1]["start_state"]["enter_exit3"],
    # start_state: Tuple[int, int] = PRESET_ENVIRONMENTS[1]["start_state"]["bdsd"],
    # start_state: Tuple[int, int] = PRESET_ENVIRONMENTS[1]["start_state"]["enter_exit4"],

    # end_state: Tuple[int, int] = PRESET_ENVIRONMENTS[1]["start_state"]["fort_santiago"],
    # end_state: Tuple[int, int] = PRESET_ENVIRONMENTS[1]["start_state"]["manila_cathedral"],
    # end_state: Tuple[int, int] = PRESET_ENVIRONMENTS[1]["start_state"]["enter_exit1"],
    # end_state: Tuple[int, int] = PRESET_ENVIRONMENTS[1]["start_state"]["dlm"],
    # end_state: Tuple[int, int] = PRESET_ENVIRONMENTS[1]["start_state"]["enter_exit2"],
    # end_state: Tuple[int, int] = PRESET_ENVIRONMENTS[1]["start_state"]["san_agustin_church"],
    # end_state: Tuple[int, int] = PRESET_ENVIRONMENTS[1]["start_state"]["enter_exit3"],
    # end_state: Tuple[int, int] = PRESET_ENVIRONMENTS[1]["start_state"]["bdsd"],
    end_state: Tuple[int, int] = PRESET_ENVIRONMENTS[1]["start_state"]["enter_exit4"]

    episodes: int = 1000
    ep_tracker: int = 10
    no_of_obstacles: int = 0
    is_dynamic_obs: bool = False
    static_obstacles: Set[Tuple[int, int]] = field(
        default_factory=lambda: set(PRESET_ENVIRONMENTS[1]["obstacles"])
    )

@dataclass
class GeneralEnvironment:
    grid: int = 10
    start_state: Tuple[int, int] = (0, 0)
    end_state: Tuple[int, int] = (0, 0)
    episodes: int = 100
    ep_tracker: int = 10
    no_of_obstacles: int = 0
    is_dynamic_obs: bool = False
    static_obstacles: Set[Tuple[int, int]] = field(
        default_factory=lambda: set()
    )