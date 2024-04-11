import json
import random
from collections.abc import Mapping
from dataclasses import dataclass
from typing import Sequence, TextIO

from minesweeper import Minesweeper
from minesweeper_plots import plot_stats

import numpy as np
import numpy.typing as npt


# An action in Minesweeper is defined as a tuple of two integers,
# representing row and column respectively.
Action = tuple[int, int]


class AgentState:
    """
    Represents the state of the Minesweeper game from the agent's perspective.
    The state is encoded as a one-hot vector for each cell on the board.
    """
    @staticmethod
    def _hash(board: Minesweeper.Board) -> int:
        """
        Compute a unique hash for a given board state.
        """
        hash_val: int = 0

        minesweeper_state: Minesweeper.State
        for minesweeper_state in board:
            hash_val *= 10
            hash_val += minesweeper_state.value

        return hash_val

    # Precompute the one-hot representations for the cell states.
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

        """
        Reconstructs the state representation from a hash string.
        """
        state: str
        return (
            *(AgentState.cell_repr[0] for _ in range(leading_zeroes)),
            *(AgentState.cell_repr[int(i)] for i in hash_str)
        )

    __slots__ = "height", "width", "states", "hash_val"

    def __init__(self, board: Minesweeper.Board | tuple[int, int, str]):
        """
        Initializes using a Minesweeper board or a tuple.
        :param board: a Minesweeper board or a (height, width, hash) tuple.
        """
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
        """
        Retrieves a list of all possible actions (cell reveals) for the current
            state.
        """
        index: int
        cell: tuple[int, ...]
        return [
            divmod(index, self.width)
            for index, cell in enumerate(self.states) if cell[9] == 1
        ]

    def get_unrevealed_cells(self) -> list[tuple[int, int]]:
        """
        Retrieves a list of all unrevealed cells in the current state.
        """
        row: int
        col: int
        return [
            (row, col)
            for row in range(self.height)
            for col in range(self.width)
            if self[row, col][9] == 1
        ]

    def _get_state_index(self, index: int | tuple[int, int]) -> int:
        """
        Converts a tuple index to a single index.
        """
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
    """
    Represents a policy for the Minesweeper game.
    """
    @staticmethod
    def from_file(file: TextIO) -> "Policy":
        """
        Loads a policy from a file.
        """
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
        """
        Initializes the policy with a given height and width.
        """
        super().__init__(*args, **kwargs)

        self.height: int = height
        self.width: int = width

        self.found_count: int = 0
        self.guess_count: int = 0

    def reset_count(self) -> None:
        """
        Resets the count of found and guessed states.
        """
        self.found_count = self.guess_count = 0

    def as_json(self) -> dict[str, dict[int, Action] | int]:
        """
        Converts the policy to a JSON-serializable format.
        """
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
        """
        Writes the policy to a file.
        """
        json.dump(self.as_json(), file)

    def __getitem__(self, state: AgentState) -> Action:
        """
        Retrieves the action for a given state.
        """
        action: Action | None = super().get(state, None)

        if action is None:
            self.guess_count += 1
            action = random.choice(state.get_unrevealed_cells())
        else:
            self.found_count += 1

        return action


@dataclass(frozen=True, slots=True)
class EpisodeStep:
    """
    Represents a step in an episode of the Minesweeper game.
    """
    state: AgentState
    action: Action
    next_reward: float
    action_probability_reciprocal: float


def generate_episode(ms: Minesweeper,
                     policy: Mapping[AgentState, Action],
                     epsilon: float
                     ) -> list[EpisodeStep]:
    """
    Generates an episode of the Minesweeper game using the given policy.
    """
    episode: list[EpisodeStep] = []
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

        reveal_info: Minesweeper.RevealInfo = ms.reveal_cell(*action)

        episode.append(
            EpisodeStep(
                state,
                action,
                reveal_info.reward,
                action_probability_reciprocal
            )
        )

        if reveal_info.cells_updated is None:
            break

        cell: tuple[int, int]
        for cell in reveal_info.cells_updated:
            unrevealed_cells.remove(cell)

    return episode


def get_best_action(state: AgentState,
                    values: Mapping[tuple[AgentState, Action], float],
                    actions: Sequence[Action]) -> Action:
    """
    Retrieves the best action for a given state.
    """
    best_action: Action = actions[0]
    best_action_value: float = values.get((state, best_action), 0)

    action: Action
    for action in actions:
        value: float = values.get((state, action), 0)

        if value > best_action_value:
            best_action = action
            best_action_value = value

    return best_action


@dataclass(frozen=True, slots=True)
class PolicyRunData:
    has_won: bool
    rewards: float
    num_time_steps: int


def run_policy(ms: Minesweeper, policy: Policy) -> PolicyRunData:
    """
    Runs the Minesweeper game using the given policy. Does not copy the env.
    """
    state: AgentState = AgentState(ms.get_board())
    total_reward: float = 0
    time_steps: int = 0

    while True:
        time_steps += 1

        action: Action = policy[state]

        reveal_info: Minesweeper.RevealInfo = ms.reveal_cell(*action)

        total_reward += reveal_info.reward

        if reveal_info.cells_updated is None:
            return PolicyRunData(ms.has_won, total_reward, time_steps)

        state = AgentState(ms.get_board())


def policy_succeeds(ms: Minesweeper,
                    policy: Policy,
                    num_runs: int = 5) -> PolicyRunData:
    """
    Determines if the policy succeeds in the Minesweeper game.

    Return data represents if the policy is both deterministic and wins,
    the average reward per run, and the average number of steps per episode.
    """
    assert num_runs > 0

    # Use first run to determine if policy will deterministically reach
    # the goal from the start state
    policy.reset_count()

    ms_copy: Minesweeper = ms.copy()

    policy_run_data: PolicyRunData = run_policy(ms_copy, policy)

    # Policy fully deterministic when following policy at start state
    if not policy.guess_count:
        return policy_run_data

    total_reward: float = policy_run_data.rewards
    total_time_steps: int = policy_run_data.num_time_steps

    for _ in range(num_runs - 1):
        next_policy_run_data: PolicyRunData = run_policy(ms_copy, policy)

        total_reward += next_policy_run_data.rewards
        total_time_steps += next_policy_run_data.num_time_steps

    return PolicyRunData(
        False,
        total_reward / num_runs,
        round(total_time_steps / num_runs)
    )


@dataclass(slots=True)
class Stats:
    episode_lengths: npt.NDArray[np.int32]
    episode_rewards: npt.NDArray[np.float64]


def off_policy_control(ms: Minesweeper,
                       maximum_episode_count: int,
                       discount_factor: float,
                       epsilon: float
                       ) -> tuple[Policy, list[int], list[float]]:
    """
    Off-policy control algorithm for the Minesweeper game.


    :param ms: Minesweeper environment.
    :param maximum_episode_count: the maximum number of episodes to run.
    :param discount_factor: the discount factor.
    :param epsilon: the chance to take a random action.
    :return: a tuple containing the policy, the episode lengths, and the
        episode rewards.
    """
    # Initialize
    values: dict[tuple[AgentState, Action], float] = {}
    cumulative_weights: dict[tuple[AgentState, Action], float] = {}

    policy: Policy = Policy(ms.height, ms.width)

    evaluation_interval: int = 4
    prev_avg_reward: float | None = None
    stop_delta_threshold: float \
        = min(abs(ms.progress_reward), abs(ms.random_penalty))

    episode_lengths: list[int] = []
    episode_rewards: list[float] = []

    epoch: int
    for epoch in range(1, maximum_episode_count + 1):
        policy_run_data: PolicyRunData = run_policy(ms.copy(), policy)
        episode_lengths.append(policy_run_data.num_time_steps)
        episode_rewards.append(policy_run_data.rewards)

        if not epoch % evaluation_interval:
            print(f"Epoch {epoch}")

            policy_run_data: PolicyRunData = policy_succeeds(ms, policy)

            if prev_avg_reward is not None:
                print(f"{prev_avg_reward=:.2f}",
                      f"{policy_run_data.rewards=:.2f}")

            if (prev_avg_reward is not None
                    and policy_run_data.has_won
                    and abs(policy_run_data.rewards - prev_avg_reward)
                        < stop_delta_threshold):
                return policy, episode_lengths, episode_rewards
            else:
                evaluation_interval = min(1024, evaluation_interval * 2)

            prev_avg_reward = policy_run_data.rewards

        # (AgentState, Action, Reward, 1 / b(Action | AgentState))
        episode: list[EpisodeStep] \
            = generate_episode(ms.copy(), policy, epsilon)

        return_val: float = 0
        weight: float = 1

        index: int
        episode_step: EpisodeStep
        for index, episode_step in enumerate(reversed(episode)):
            return_val = (return_val * discount_factor
                          + episode_step.next_reward)

            state_action_pair: tuple[AgentState, Action] \
                = episode_step.state, episode_step.action

            cumulative_weights[state_action_pair] \
                = cumulative_weights.get(state_action_pair, 0) + weight

            values[state_action_pair] = values.get(state_action_pair, 0)
            values[state_action_pair] += (
                    weight / cumulative_weights[state_action_pair]
                    * (return_val - values[state_action_pair])
            )

            all_actions: list[Action] \
                = episode_step.state.get_unrevealed_cells()
            best_action: Action \
                = get_best_action(episode_step.state, values, all_actions)

            policy[episode_step.state] = best_action

            if episode_step.action != best_action:
                break

            weight *= episode_step.action_probability_reciprocal

    return policy, episode_lengths, episode_rewards


def main():
    """
    Main function for the Minesweeper agent.
    This function allows the user to play a game of Minesweeper using a policy
    generated by the off-policy control algorithm.
    """
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
        episode_lengths: list[int]
        episode_rewards: list[float]
        policy, episode_lengths, episode_rewards = off_policy_control(
            ms.copy(),
            maximum_episode_count=1_000_000,
            discount_factor=0.95,
            epsilon=0.01
        )

        plot_stats(
            "Training Stats",
            "Episode Rewards",
            Stats(np.array(episode_lengths), np.array(episode_rewards))
        )

    policy.reset_count()

    policy_run_data: PolicyRunData = run_policy(ms, policy)
    print(f"You won! reward={policy_run_data.rewards:.2f}"
          if policy_run_data.has_won
          else f"You lost! reward={policy_run_data.rewards:.2f}")

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
