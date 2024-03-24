import json
import random
from typing import Sequence, TextIO

from minesweeper import Minesweeper


Action = tuple[int, int]


class AgentState:
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

    def __init__(self, board: Minesweeper.Board | dict[str, int]):
        if isinstance(board, Minesweeper.Board):
            self.height: int = board.height
            self.width: int = board.width

            # Bombs in the board are unrevealed, values range from 0-9
            minesweeper_state: Minesweeper.State
            i: int
            self.states: tuple[tuple[int, ...], ...] = tuple(
                AgentState.cell_repr[minesweeper_state.value]
                for minesweeper_state in board
            )

            self._hash_val: int = AgentState._hash(board)
        elif isinstance(board, dict):
            try:
                self.height: int = board["height"]
                self.width: int = board["width"]

                hash_val_str: str = str(board["hash_val"])
                leading_zeroes: int \
                    = self.height * self.width - len(hash_val_str)

                state: str
                self.states: tuple[tuple[int, ...], ...] = (
                    *(AgentState.cell_repr[0] for _ in range(leading_zeroes)),
                    *(AgentState.cell_repr[int(i)] for i in hash_val_str)
                )

                self._hash_val: int = board["hash_val"]
            except KeyError:
                print("Invalid board dict.")
        else:
            raise TypeError(f"Expected Board or dict, got {type(board)}")

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

    def as_dict(self) -> dict[str, int]:
        return {
            "height": self.height,
            "width": self.width,
            "hash_val": self._hash_val
        }

    def __getitem__(self,
                    index: int | tuple[int, int]) -> tuple[int, ...]:
        return self.states[self._get_state_index(index)]

    def __hash__(self) -> int:
        return self._hash_val

    def __eq__(self, other: object) -> bool:
        return (isinstance(other, AgentState)
                and self._hash_val == other._hash_val)

    def __str__(self) -> str:
        return str(self._hash_val)


class Policy(dict[AgentState, Action]):
    @staticmethod
    def from_file(file: TextIO) -> "Policy":
        json_obj: list[list[dict[str, int] | Action]] = json.load(file)

        policy: Policy = Policy()

        board_action_pair: list[dict[str, int] | Action]
        for board_action_pair in json_obj:
            state: AgentState = AgentState(board_action_pair[0])

            policy[state] = board_action_pair[1]

        return policy

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        self.found_count: int = 0
        self.guess_count: int = 0

    def reset_count(self) -> None:
        self.found_count = self.guess_count = 0

    def as_json(self) -> list[tuple[dict[str, int], Action]]:
        state: AgentState
        action: Action
        return [
            (state.as_dict(), action) for state, action in self.items()
        ]

    def to_file(self, file: TextIO) -> None:
        json.dump(self.as_json(), file)

    def __getitem__(self, state: AgentState) -> Action:
        action: Action | None = super().get(state, None)

        if action is None:
            self.guess_count += 1
            action = random.choice(state.get_unrevealed_cells())
        else:
            self.found_count += 1

        return action


def generate_episode(ms: Minesweeper,
                     policy: dict[AgentState, Action],
                     epsilon: float
                     ) -> list[tuple[AgentState, Action, float, float]]:
    episode: list[tuple[AgentState, Action, float, float]] = []
    unrevealed_cells: list[tuple[int, int]] = ms.get_unrevealed_cells()

    while True:
        state: AgentState = AgentState(ms.get_board())
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


def get_best_action(state: AgentState,
                    values: dict[tuple[AgentState, Action], float],
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


def run_policy(ms: Minesweeper, policy: Policy) -> bool:
    state: AgentState = AgentState(ms.get_board())

    updated: set[tuple[int, int]] = set()
    while updated is not None:
        action: Action = policy[state]
        _, updated = ms.reveal_cell(*action)

        if updated is None:
            return ms.has_won()

        state = AgentState(ms.get_board())


def evaluate_policy(ms: Minesweeper,
                    policy: Policy,
                    evaluate_count: int = 30) -> float:
    success_count: int = 0

    for _ in range(evaluate_count):
        policy.reset_count()

        ms_copy: Minesweeper = ms.copy()
        success_count += 1 if run_policy(ms_copy, policy) else 0

        # Policy fully deterministic when following policy at start state
        if not policy.guess_count:
            return 1 if ms_copy.has_won() else 0

    return success_count / evaluate_count


def off_policy_control(ms: Minesweeper,
                       maximum_episode_count: int,
                       discount_factor: float,
                       epsilon: float) -> Policy:
    # Initialize
    values: dict[tuple[AgentState, Action], float] = {}
    cumulative_weights: dict[tuple[AgentState, Action], float] = {}

    policy: Policy = Policy()

    evaluation_interval: int = 4

    epoch: int
    for epoch in range(maximum_episode_count):
        if not epoch % evaluation_interval:
            success_rate: float = evaluate_policy(ms, policy)
            print(f"Epoch {epoch}")

            if success_rate == 1:
                return policy
            else:
                evaluation_interval = min(1024, evaluation_interval * 2)

        # (AgentState, Action, Reward, 1 / b(Action | AgentState))
        episode: list[tuple[AgentState, Action, float, float]] \
            = generate_episode(ms.copy(), policy, epsilon)

        return_val: float = 0
        weight: float = 1

        index: int
        state: AgentState
        action: Action
        next_reward: float
        action_probability_reciprocal: float
        for (index,
             (state, action, next_reward, action_probability_reciprocal)) \
                in enumerate(reversed(episode)):
            return_val = return_val * discount_factor + next_reward

            state_action_pair: tuple[AgentState, Action] = state, action

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
    response: str = input("Enter seed, or leave blank for random: ")

    if response:
        random.seed(int(response))

    difficulty = input("Enter difficulty level (easy/normal/hard): ")
    ms: Minesweeper \
        = Minesweeper(*Minesweeper.default_difficulties[difficulty])

    # print(ms)

    policy: Policy

    load_policy: bool | None = None

    while load_policy is None:
        response: str \
            = (input("Do you want to load a policy from file (y/n): ")
               .strip()
               .lower())

        if response == "y":
            load_policy = True
        elif response == "n":
            load_policy = False

    if load_policy:
        filename: str = input("Enter the name of the file: ")

        file: TextIO
        with open(filename, "r") as file:
            policy = Policy.from_file(file)
    else:
        policy = off_policy_control(
            ms.copy(),
            maximum_episode_count=1_000_000,
            discount_factor=0.95,
            epsilon=0.001
        )

    policy.reset_count()

    has_won: bool = run_policy(ms, policy)
    print("You won!" if has_won else "You lost!")

    print(f"State found in policy {policy.found_count} time(s).")
    print(f"Random guess made {policy.guess_count} time(s).")

    write_to_file: bool | None = None
    while write_to_file is None:
        response: str \
            = (input("Do you want to write your policy to file (y/n): ")
               .strip()
               .lower())

        if response == "y":
            write_to_file = True
        elif response == "n":
            write_to_file = False

    if write_to_file:
        filename: str = input("Enter the name of the file: ")

        file: TextIO
        with open(filename, "w") as file:
            policy.to_file(file)


if __name__ == "__main__":
    main()
