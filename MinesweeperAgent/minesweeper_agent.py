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

    @staticmethod
    def states_from_hash(height: int,
                         width: int,
                         hash_str: str) -> tuple[tuple[int, ...], ...]:
        leading_zeroes: int = height * width - len(hash_str)

        state: str
        return (
            *(AgentState.cell_repr[0] for _ in range(leading_zeroes)),
            *(AgentState.cell_repr[int(i)] for i in hash_str)
        )

    __slots__ = "height", "width", "states", "hash_val"

    def __init__(self, board: Minesweeper.Board | tuple[int, int, str]):
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

            self.hash_val: int = AgentState._hash(board)
        elif isinstance(board, tuple):
            try:
                self.height: int
                self.width: int

                hash_str: str
                self.height, self.width, hash_str = board

                self.states: tuple[tuple[int, ...], ...] \
                    = AgentState.states_from_hash(
                    self.height,
                    self.width,
                    hash_str
                )

                self.hash_val: int = int(hash_str)
            except KeyError:
                print(f"Expected a tuple (height, width, hash_val), "
                      f"got {board}")
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

    def __getitem__(self,
                    index: int | tuple[int, int]) -> tuple[int, ...]:
        return self.states[self._get_state_index(index)]

    def __hash__(self) -> int:
        return self.hash_val

    def __eq__(self, other: object) -> bool:
        return (isinstance(other, AgentState)
                and self.hash_val == other.hash_val)

    def __str__(self) -> str:
        return str(self.hash_val)


class Policy(dict[AgentState, Action]):
    @staticmethod
    def from_file(file: TextIO) -> "Policy":
        json_obj: dict[str, dict[str, Action] | int] = json.load(file)

        height: int = json_obj["height"]
        width: int = json_obj["width"]

        policy: Policy = Policy(height, width)

        state_hash: str
        action: Action
        for state_hash, action in json_obj["policy"].items():
            state: AgentState = AgentState((height, width, state_hash))

            policy[state] = action

        return policy

    __slots__ = "height", "width", "found_count", "guess_count"

    def __init__(self, height: int, width: int, *args, **kwargs):
        super().__init__(*args, **kwargs)

        self.height: int = height
        self.width: int = width

        self.found_count: int = 0
        self.guess_count: int = 0

    def reset_count(self) -> None:
        self.found_count = self.guess_count = 0

    def as_json(self) -> dict[str, dict[int, Action] | int]:
        state: AgentState
        action: Action
        return {
            "height": self.height,
            "width": self.width,
            "policy": {
                state.hash_val: action for state, action in self.items()
            }
        }

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


def run_policy(ms: Minesweeper, policy: Policy) -> tuple[bool, float]:
    state: AgentState = AgentState(ms.get_board())
    total_reward: float = 0

    while True:
        action: Action = policy[state]

        new_reward: float
        updated: set[tuple[int, int]]
        new_reward, updated = ms.reveal_cell(*action)

        total_reward += new_reward

        if updated is None:
            return ms.has_won(), total_reward

        state = AgentState(ms.get_board())


def policy_succeeds(ms: Minesweeper,
                    policy: Policy,
                    num_runs: int = 100) -> tuple[bool, float]:
    assert num_runs > 0

    # Use first run to determine if policy will deterministically reach
    # the goal from the start state
    policy.reset_count()

    ms_copy: Minesweeper = ms.copy()

    has_won: bool
    total_reward: float
    has_won, total_reward = run_policy(ms_copy, policy)

    # Policy fully deterministic when following policy at start state
    if not policy.guess_count:
        return has_won, total_reward

    for _ in range(num_runs - 1):
        ms_copy: Minesweeper = ms.copy()
        next_reward: float
        _, next_reward = run_policy(ms_copy, policy)

        total_reward += next_reward

    return False, total_reward / num_runs


def off_policy_control(ms: Minesweeper,
                       maximum_episode_count: int,
                       discount_factor: float,
                       epsilon: float) -> Policy:
    # Initialize
    values: dict[tuple[AgentState, Action], float] = {}
    cumulative_weights: dict[tuple[AgentState, Action], float] = {}

    policy: Policy = Policy(ms.height, ms.width)

    evaluation_interval: int = 4
    prev_avg_reward: float | None = None
    stop_delta_threshold: float \
        = min(abs(ms.progress_reward), abs(ms.random_penalty))

    epoch: int
    for epoch in range(1, maximum_episode_count + 1):
        if not epoch % evaluation_interval:
            print(f"Epoch {epoch}")

            has_won: bool
            average_rewards: float
            has_won, average_rewards = policy_succeeds(ms, policy)

            if prev_avg_reward is not None:
                print(f"{prev_avg_reward=:.2f}", f"{average_rewards=:.2f}")

            if (prev_avg_reward is not None
                    and has_won
                    and abs(average_rewards - prev_avg_reward)
                        < stop_delta_threshold):
                return policy
            else:
                evaluation_interval = min(1024, evaluation_interval * 2)

            prev_avg_reward = average_rewards

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
    response: str | None = None
    while response is None:
        try:
            response = input("Enter seed, or leave blank for random: ")

            if response:
                random.seed(int(response))
        except ValueError:
            print("Invalid seed.")
            response = None

    difficulty: str | None = None
    while difficulty is None:
        difficulty = input("Enter difficulty level (easy/normal/hard): ")

        if difficulty not in Minesweeper.default_difficulties:
            difficulty = None
            print("Invalid difficulty level.")

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
        filename: str | None = None

        while filename is None:
            try:
                filename = input("Enter the name of the file: ")

                file: TextIO
                with open(filename, "r") as file:
                    policy = Policy.from_file(file)
            except FileNotFoundError:
                print("File cannot be found.")
                filename = None
    else:
        policy = off_policy_control(
            ms.copy(),
            maximum_episode_count=1_000_000,
            discount_factor=0.95,
            epsilon=0.01
        )

    policy.reset_count()

    has_won: bool
    total_reward: float
    has_won, total_reward = run_policy(ms, policy)
    print(f"You won! {total_reward=:.2f}" if has_won
          else f"You lost! {total_reward=:.2f}")

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
        else:
            print("Invalid response. Please try again.")

    if write_to_file:
        filename: str = input("Enter the name of the file: ")

        file: TextIO
        with open(filename, "w") as file:
            policy.to_file(file)


if __name__ == "__main__":
    main()
