from __future__ import annotations

from typing import List, Tuple, Optional, Set, Dict
from rl_thesis.environment.world import World
from rl_thesis.environment.entities import Position


# Action constants (must match entities.Direction order)
STAY  = 0
UP    = 1  # dy = -1
DOWN  = 2  # dy = +1
LEFT  = 3  # dx = -1
RIGHT = 4  # dx = +1

# Pre-computed (dx, dy) for each action index, used by _next_pos.
_ACTION_DELTAS = (
    (0, 0),   # STAY
    (0, -1),  # UP
    (0, 1),   # DOWN
    (-1, 0),  # LEFT
    (1, 0),   # RIGHT
)
_NUM_ACTIONS = len(_ACTION_DELTAS)


def _step_toward(src: Tuple[int, int], dst: Tuple[int, int]) -> int:
    """Return the single-step action that moves *src* toward *dst*.

    Uses Manhattan (axis-aligned) movement.  Prefers the axis with the
    larger gap so the agent doesn't zig-zag unnecessarily.
    """
    dx = dst[0] - src[0]
    dy = dst[1] - src[1]

    if dx == 0 and dy == 0:
        return STAY

    # Move along the axis with the bigger gap first
    if abs(dx) >= abs(dy):
        return RIGHT if dx > 0 else LEFT
    else:
        return DOWN if dy > 0 else UP


def _nearest_in_radius(
    pos: Tuple[int, int],
    candidates: Set[Tuple[int, int]] | Dict,
    radius: int,
) -> Optional[Tuple[int, int]]:
    """Return the closest candidate within *radius* (Manhattan) of *pos*, or None."""
    best, best_d = None, float("inf")
    items = candidates if isinstance(candidates, set) else candidates.keys()
    for c in items:
        d = abs(c[0] - pos[0]) + abs(c[1] - pos[1])
        if d <= radius and d < best_d:
            best, best_d = c, d
    return best


def _manhattan(a: Tuple[int, int], b: Tuple[int, int]) -> int:
    """Manhattan distance between two (x, y) positions."""
    return abs(a[0] - b[0]) + abs(a[1] - b[1])


def _next_pos(
    pos: Tuple[int, int], action: int, width: int, height: int,
) -> Tuple[int, int]:
    """Position after taking *action* at *pos*, clamped to grid bounds.

    Mirrors ``Position.move`` semantics exactly.
    """
    dx, dy = _ACTION_DELTAS[action]
    return (
        max(0, min(width - 1, pos[0] + dx)),
        max(0, min(height - 1, pos[1] + dy)),
    )


def _nearby_enemies(
    pos: Tuple[int, int],
    enemy_positions: Dict[Tuple[int, int], list],
    radius: int,
) -> List[Tuple[int, int]]:
    """Enemy positions within *radius* Manhattan distance of *pos*.

    Returns unique tile positions (not individual enemy objects),
    which is sufficient for distance-based threat scoring.
    """
    return [
        epos for epos in enemy_positions
        if _manhattan(pos, epos) <= radius
    ]


class HumanHeuristicAgent:
    """Forage-or-shelter heuristic agent with enemy avoidance.

    Decision priority:

    1. Flee if any enemy is within flee_radius and the agent is
       not safely inside a shelter, pick the action that maximises
       distance from threats (preferring shelter tiles as escape targets).
    2. Forage if hunger ratio drops below hunger_threshold, move
       toward the nearest visible food.
    3. Shelter otherwise, head for (or stay in) the nearest shelter.

    Parameters:
    hunger_threshold : float
        Hunger ratio (0–1) below which the agent switches from shelter
        mode to forage mode.
    flee_radius : int
        Manhattan distance within which an enemy triggers the flee
        response.
    """

    def __init__(
        self, hunger_threshold: float, flee_radius: int,
    ) -> None:
        self.hunger_threshold = hunger_threshold
        self.flee_radius = flee_radius

    # Public API (mirrors DQN agent's select_action signature loosely)

    def select_action(self, world: World) -> int:
        """Pick an action given the same local view as the RL agent.

        The scripted forager only considers entities within the
        observation radius defined in the world config, matching the RL
        agent's restricted field of view.
        """
        agent = world.agent
        pos = agent.position.as_tuple()
        radius = world.config.observation_radius

        # ---- Pre-emption: flee from nearby enemies ----
        # Shelters provide full protection (enemies cannot enter them),
        # so we only need to flee when the agent is exposed.
        if not agent.is_in_shelter:
            threats = _nearby_enemies(
                pos, world._enemy_positions, self.flee_radius,
            )
            if threats:
                return self._flee(pos, world, radius, threats)

        # ---- Normal mode: forage vs shelter ----
        if agent.hunger_ratio < self.hunger_threshold:
            return self._forage(pos, world, radius)
        else:
            return self._seek_shelter(pos, world, radius)

    # Private helpers

    def _flee(
        self,
        pos: Tuple[int, int],
        world: World,
        radius: int,
        threats: List[Tuple[int, int]],
    ) -> int:
        """Pick the safest action given nearby enemy *threats*.

        Each candidate action is scored as a lexicographic tuple so that
        higher-priority criteria always dominate:

        1. **min_dist** — minimum Manhattan distance from the resulting
           position to any threat (*maximised*).  This is the critical
           survival metric: keep the nearest enemy as far away as possible.
        2. **sum_dist** — total Manhattan distance to all threats
           (*maximised*).  Breaks ties by preferring positions that are
           generally far from *all* enemies, not just the closest.
        3. **on_shelter** — 1 if the resulting position is a shelter tile,
           0 otherwise (*maximised*).  Shelters are immune zones, so
           stepping onto one is always preferred when distances are equal.
        4. **shelter_prox** — negative distance to the nearest visible
           shelter (*maximised*, i.e. closer shelter ⇒ higher value).
           When everything else is tied, move toward the nearest escape
           route.
        """
        width = world.config.width
        height = world.config.height
        best_action = STAY
        best_score = None

        for action in range(_NUM_ACTIONS):
            npos = _next_pos(pos, action, width, height)

            min_d = min(_manhattan(npos, t) for t in threats)
            sum_d = sum(_manhattan(npos, t) for t in threats)
            on_shelter = 1 if npos in world._shelter_positions else 0

            nearest_shelter = _nearest_in_radius(
                npos, world._shelter_positions, radius,
            )
            shelter_prox = (
                -_manhattan(npos, nearest_shelter)
                if nearest_shelter is not None
                else -(radius + 1)
            )

            score = (min_d, sum_d, on_shelter, shelter_prox)
            if best_score is None or score > best_score:
                best_score = score
                best_action = action

        return best_action

    def _forage(self, pos: Tuple[int, int], world: World, radius: int) -> int:
        """Move toward the nearest visible food item."""
        target = _nearest_in_radius(pos, world._food_positions, radius)
        if target is not None:
            return _step_toward(pos, target)
        # No visible food — try shelter; if no shelter either, just stay put
        shelter = _nearest_in_radius(pos, world._shelter_positions, radius)
        if shelter is not None:
            return _step_toward(pos, shelter)
        return STAY

    def _seek_shelter(self, pos: Tuple[int, int], world: World, radius: int) -> int:
        """Move toward (or stay in) the nearest visible shelter tile."""
        if pos in world._shelter_positions:
            return STAY  # Already safe — don't move
        target = _nearest_in_radius(pos, world._shelter_positions, radius)
        if target is not None:
            return _step_toward(pos, target)
        # No visible shelters — forage for food; if nothing, stay put
        food = _nearest_in_radius(pos, world._food_positions, radius)
        if food is not None:
            return _step_toward(pos, food)
        return STAY
