import random
from typing import Sequence

from minesweeper import Minesweeper


Action = tuple[int, int]


class State:
    @staticmethod
    def _hash(board: Minesweeper.Board) -> int:
        hash_val: int = 0

        minesweeper_state: Minesweeper.State
        for minesweeper_state in board:
            hash_val *= 10
            hash_val += minesweeper_state.value

        return hash_val

    cell_repr: tuple[tuple[int, ...], ...] = tuple(
        tuple(
            int(i == j)
            for j in range(10)
        ) for i in range(10)
    )

    def __init__(self, board: Minesweeper.Board):
        self.height: int = board.height
        self.width: int = board.width

        # Bombs in the board are unrevealed, values range from 0-9
        minesweeper_state: Minesweeper.State
        i: int
        self.states: tuple[tuple[int, ...], ...] = tuple(
            State.cell_repr[minesweeper_state.value]
            for minesweeper_state in board
        )

        self._hash_val: int = State._hash(board)

    def get_actions(self) -> list[Action]:
        actions: list[Action] = []

        index: int
        cell: tuple[int, ...]
        for index, cell in enumerate(self.states):
            if cell[9] == 1:
                actions.append(divmod(index, self.width))

        return actions

    def get_unrevealed_cells(self) -> list[tuple[int, int]]:
        row: int
        col: int
        return [
            (row, col)
            for row in range(self.height)
            for col in range(self.width)
            if self[row, col][9] == 1
        ]

    def _get_state_index(self, index: int | tuple[int, int]) -> int:
        if isinstance(index, int):
            return index
        elif isinstance(index, tuple) and len(index) == 2:
            try:
                return index[0] * self.width + index[1]
            except TypeError:
                raise TypeError("Indices must be integers")
        else:
            raise TypeError(f"Expected int or tuple[int, int],"
                            f" got {type(index)}")

    def __getitem__(self,
                    index: int | tuple[int, int]) -> tuple[int, ...]:
        return self.states[self._get_state_index(index)]

    def __hash__(self) -> int:
        return self._hash_val

    def __eq__(self, other: object) -> bool:
        return isinstance(other, State) and self._hash_val == other._hash_val

    def __str__(self) -> str:
        return str(self._hash_val)


class Policy(dict[State, Action]):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        self.found_count: int = 0
        self.guess_count: int = 0

    def reset_count(self) -> None:
        self.found_count = self.guess_count = 0

    def __getitem__(self, state: State) -> Action:
        action: Action | None = super().get(state, None)

        if action is None:
            self.guess_count += 1
            action = random.choice(state.get_unrevealed_cells())
        else:
            self.found_count += 1

        return action


def generate_episode(ms: Minesweeper,
                     policy: dict[State, Action],
                     epsilon: float
                     ) -> list[tuple[State, Action, float, float]]:
    episode: list[tuple[State, Action, float, float]] = []
    unrevealed_cells: list[tuple[int, int]] = ms.get_unrevealed_cells()

    while True:
        state: State = State(ms.get_board())
        rand_val: float = random.random()

        action: Action
        action_probability_reciprocal: float
        if state in policy:
            if rand_val < epsilon:
                action = random.choice(unrevealed_cells)
            else:
                action = policy[state]

            n: int = len(unrevealed_cells)
            action_probability_reciprocal = (
                    n / (epsilon + n * (1 - epsilon)
                         if action == policy[state]
                         else epsilon)
            )
        else:
            action = random.choice(unrevealed_cells)
            action_probability_reciprocal = len(unrevealed_cells)

        reward: float
        updated_cells: set[tuple[int, int]] | None
        reward, updated_cells = ms.reveal_cell(*action)

        episode.append((state, action, reward, action_probability_reciprocal))

        if updated_cells is None:
            break

        cell: tuple[int, int]
        for cell in updated_cells:
            unrevealed_cells.remove(cell)

    return episode


def get_best_action(state: State,
                    values: dict[tuple[State, Action], float],
                    actions: Sequence[Action]) -> Action:
    best_action: Action = actions[0]
    best_action_value: float = values.get((state, best_action), 0)

    action: Action
    for action in actions:
        value: float = values.get((state, action), 0)

        if value > best_action_value:
            best_action = action
            best_action_value = value

    return best_action


def off_policy_control(ms: Minesweeper,
                       episode_count: int,
                       discount_factor: float,
                       epsilon: float) -> Policy:
    # Initialize
    values: dict[tuple[State, Action], float] = {}
    cumulative_weights: dict[tuple[State, Action], float] = {}

    policy: Policy = Policy()

    epoch: int
    for epoch in range(episode_count):
        if not epoch % 1000:
            print(f"Epoch {epoch}")

        # (State, Action, Reward, 1 / b(Action | State))
        episode: list[tuple[State, Action, float, float]] \
            = generate_episode(ms.copy(), policy, epsilon)

        return_val: float = 0
        weight: float = 1

        index: int
        state: State
        action: Action
        next_reward: float
        action_probability_reciprocal: float
        for (index,
             (state, action, next_reward, action_probability_reciprocal)) \
                in enumerate(reversed(episode)):
            return_val = return_val * discount_factor + next_reward

            state_action_pair: tuple[State, Action] = state, action

            cumulative_weights[state_action_pair] \
                = cumulative_weights.get(state_action_pair, 0) + weight

            values[state_action_pair] = values.get(state_action_pair, 0)
            values[state_action_pair] += (
                    weight / cumulative_weights[state_action_pair]
                    * (return_val - values[state_action_pair])
            )

            all_actions: list[Action] = state.get_unrevealed_cells()
            best_action: Action = get_best_action(state, values, all_actions)

            policy[state] = best_action

            if action != best_action:
                break

            weight *= action_probability_reciprocal

    return policy


def main():
    random.seed(2)
    ms: Minesweeper = Minesweeper(16, 30, 99)

    # print(ms)

    policy: Policy = off_policy_control(
        ms,
        episode_count=100_000,
        discount_factor=0.95,
        epsilon=0.001
    )
    policy.reset_count()

    state: State = State(ms.get_board())

    updated: set[tuple[int, int]] = set()
    while updated is not None:
        action: Action = policy[state]
        # print(action)
        _, updated = ms.reveal_cell(*action)

        # print(ms)

        if updated is None:
            if ms.has_lost():
                print("You lost!")
            else:
                print("You won!")

        state = State(ms.get_board())

    print(f"State found in policy {policy.found_count} time(s).")
    print(f"Random guess made {policy.guess_count} time(s).")


if __name__ == "__main__":
    main()
