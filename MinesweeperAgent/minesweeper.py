import itertools
import random
from copy import deepcopy
from dataclasses import dataclass
from enum import Enum
from typing import Iterator, Callable, Generator


class Minesweeper:
    class State(Enum):
        BOMB = -1
        ZERO = 0
        ONE = 1
        TWO = 2
        THREE = 3
        FOUR = 4
        FIVE = 5
        SIX = 6
        SEVEN = 7
        EIGHT = 8
        UNREVEALED = 9

    class Board:
        __slots__ = "height", "width", "num_bombs", "board"

        def __init__(self, height: int, width: int, num_bombs: int):
            self.height: int = height
            self.width: int = width
            self.num_bombs: int = num_bombs

            # Can't be all bombs
            assert 0 <= num_bombs < self.height * self.width

            self.board: list[Minesweeper.State] \
                = [Minesweeper.State.UNREVEALED] * (self.height * self.width)

            self._set_bombs()

        @property
        def cell_count(self) -> int:
            return len(self.board)

        def _set_bombs(self) -> None:
            remaining_bombs: int = self.num_bombs

            cell: int
            for cell in range(self.height * self.width - 1, -1, -1):
                rand_val: int = random.randint(0, cell)
                if rand_val < remaining_bombs:
                    self.board[cell] = Minesweeper.State.BOMB
                    remaining_bombs -= 1

                    if not remaining_bombs:
                        return

        def cell_to_rc(self, cell: int) -> tuple[int, int]:
            """
            :param cell: the state to convert into a (row, column) tuple.
            :return: a tuple containing the row and column of the state.
            """
            return cell // self.width, cell % self.width

        def cell_from_rc(self, r: int, c: int) -> int:
            """
            :param r: the row of the state.
            :param c: the column of the state.
            :return: the index of the state.
            """
            return r * self.width + c

        def is_cell_inbounds(self, cell: int) -> bool:
            return 0 <= cell < self.cell_count

        def is_cell_inbounds_rc(self, r: int, c: int) -> bool:
            return 0 <= r < self.height and 0 <= c < self.width

        def count(self, state: "Minesweeper.State") -> int:
            return self.board.count(state)

        def _get_board_index(self, index: int | tuple[int, int]) -> int:
            if isinstance(index, int):
                return index
            elif isinstance(index, tuple) and len(index) == 2:
                try:
                    return self.cell_from_rc(*index)
                except TypeError:
                    raise TypeError("Indices must be integers")
            else:
                raise TypeError(f"Expected int or tuple[int, int],"
                                f" got {type(index)}")

        def surrounding_cells(self,
                              row: int,
                              col: int
                              ) -> Generator[tuple[int, int], None, None]:
            r: int
            for r in range(max(0, row - 1), min(row + 2, self.height)):
                c: int
                for c in range(max(0, col - 1), min(col + 2, self.width)):
                    yield r, c

        def __getitem__(self,
                        index: int | tuple[int, int]) -> "Minesweeper.State":
            return self.board[self._get_board_index(index)]

        def __setitem__(self,
                        key: int | tuple[int, int],
                        value: "Minesweeper.State") -> None:
            self.board[self._get_board_index(key)] = value

        def as_formatted_str(self,
                             horizontal_spacing: int = 12,
                             bomb_repr: Callable[[int, int], str]
                             = lambda _1, _2: "BOMB",
                             unrevealed_repr: Callable[[int, int], str]
                             = lambda _1, _2: "UNREVEALED",
                             other_repr: Callable[
                                 [int, int, "Minesweeper.State"],
                                 str
                             ] = lambda _1, _2, val: str(val)
                             ) -> str:
            """
            Gets a string representation of the object.

            :param horizontal_spacing: the amount of space given for each cell.
            :param bomb_repr: gets the representation for the bomb.
            :param unrevealed_repr: gets the representation for an unrevealed
                cell.
            :param other_repr: gets the representation for other cells.
            :return: a string representation of the object.
            """
            output_format: str = "{:>" + str(horizontal_spacing) + "}"

            state_iterator: Iterator[Minesweeper.State] = iter(self.board)

            row: int
            column: int
            value: Minesweeper.State
            output_str: str = '\n'.join(
                ''.join(
                    output_format.format(
                        bomb_repr(row, column)
                        if value == Minesweeper.State.BOMB
                        else unrevealed_repr(row, column)
                        if value == Minesweeper.State.UNREVEALED
                        else other_repr(row, column, value)
                    ) for column, value in
                    enumerate(itertools.islice(state_iterator, self.width))
                ) for row in range(self.height)
            )

            return output_str

        def __str__(self):
            return self.as_formatted_str()

        def copy(self, obfuscate_bombs: bool = True) -> "Minesweeper.Board":
            ms_copy: Minesweeper.Board = deepcopy(self)

            if obfuscate_bombs:
                index: int
                value: Minesweeper.State
                for index, value in enumerate(ms_copy.board):
                    if value == Minesweeper.State.BOMB:
                        ms_copy.board[index] = Minesweeper.State.UNREVEALED

            return ms_copy

        def __deepcopy__(self, memo):
            cls = self.__class__
            result: Minesweeper.Board = cls.__new__(cls)
            memo[id(self)] = result
            for k in Minesweeper.Board.__slots__:
                v = getattr(self, k)
                setattr(result, k, deepcopy(v, memo))
            return result

    @dataclass(slots=True)
    class RevealInfo:
        reward: float
        cells_updated: set[tuple[int, int]] | None

    default_difficulties: dict[str, tuple[int, int, int]] = {
        "easy": (9, 9, 10),
        "normal": (16, 16, 40),
        "hard": (16, 30, 99)
    }

    __slots__ = (
        "_board",
        "progress_reward",
        "random_penalty",
        "first_reveal",
        "_game_state"
    )

    def __init__(self,
                 height: int,
                 width: int,
                 num_bombs: int,
                 progress_reward: float = 0.3,
                 random_penalty: float = -0.3):
        self._board: Minesweeper.Board \
            = Minesweeper.Board(height, width, num_bombs)

        self.progress_reward: float = progress_reward
        self.random_penalty: float = random_penalty

        self.first_reveal: bool = True

        # 0: running
        # -1: lost
        # 1: win
        self._game_state: int = 0

    @property
    def height(self) -> int:
        return self._board.height

    @property
    def width(self) -> int:
        return self._board.width

    @property
    def cell_count(self) -> int:
        return self.height * self.width

    @property
    def _is_winner(self) -> bool:
        num_unrevealed: int = self._board.count(Minesweeper.State.UNREVEALED)
        return num_unrevealed == 0

    @property
    def is_running(self) -> bool:
        return not self._game_state

    @property
    def has_lost(self) -> bool:
        return self._game_state == -1

    @property
    def has_won(self) -> bool:
        return self._game_state == 1

    def cell_to_rc(self, cell: int) -> tuple[int, int]:
        """
        :param cell: the state to convert into a (row, column) tuple.
        :return: a tuple containing the row and column of the state.
        """
        return self._board.cell_to_rc(cell)

    def cell_from_rc(self, r: int, c: int) -> int:
        """
        :param r: the row of the state.
        :param c: the column of the state.
        :return: the index of the state.
        """
        return self._board.cell_from_rc(r, c)

    def is_cell_inbounds(self, cell: int) -> bool:
        return 0 <= cell < self.cell_count

    def is_cell_inbounds_rc(self, r: int, c: int) -> bool:
        return 0 <= r < self.height and 0 <= c < self.width

    def _is_bomb(self, row: int, col: int) -> bool:
        if not self._board.is_cell_inbounds_rc(row, col):
            return False
        else:
            return self._board[row, col] == Minesweeper.State.BOMB

    def _relocate_bomb(self, row: int, col: int) -> None:
        assert self._is_bomb(row, col)

        cell: int
        for cell in range(self.height * self.width):
            if self._board[cell] == Minesweeper.State.UNREVEALED:
                self._board[cell] = Minesweeper.State.BOMB
                self._board[row, col] = Minesweeper.State.UNREVEALED
                return

        raise Exception("Can't relocate the bomb.")

    def _count_surrounding_bombs(self, row: int, col: int) -> int:
        surrounding_bombs: int = 0

        r: int
        c: int
        for r, c in self._board.surrounding_cells(row, col):
            surrounding_bombs += 1 if self._is_bomb(r, c) else 0

        return surrounding_bombs

    def _get_reward(self, row: int, col: int) -> float:
        r: int
        c: int
        for r, c in self._board.surrounding_cells(row, col):
            state: Minesweeper.State = self._board[r, c]
            if (state != Minesweeper.State.UNREVEALED
                    and state != Minesweeper.State.BOMB):
                return self.progress_reward

        return self.random_penalty

    def get_unrevealed_cells(self) -> list[tuple[int, int]]:
        row: int
        col: int
        return [
            (row, col)
            for row in range(self.height)
            for col in range(self.width)
            if self._board[row, col] == Minesweeper.State.UNREVEALED
            or self._board[row, col] == Minesweeper.State.BOMB
        ]

    def reveal_cell(self,
                    row: int,
                    col: int
                    ) -> RevealInfo:
        if self.first_reveal:
            self.first_reveal = False

            if self._board[row, col] == Minesweeper.State.BOMB:
                self._relocate_bomb(row, col)

        if self._is_bomb(row, col):
            self._game_state = -1
            return Minesweeper.RevealInfo(-1, None)
        elif self._board[row, col] != Minesweeper.State.UNREVEALED:
            return Minesweeper.RevealInfo(0, set())
        else:
            reward: float = self._get_reward(row, col)

            self._board[row, col] \
                = Minesweeper.State(self._count_surrounding_bombs(row, col))
            reveal_info: Minesweeper.RevealInfo \
                = Minesweeper.RevealInfo(reward, {(row, col)})

            if self._is_winner:
                self._game_state = 1
                return Minesweeper.RevealInfo(1, None)
            elif self._board[row, col] != Minesweeper.State.ZERO:
                return reveal_info

            r: int
            for r in range(row - 1, row + 2):
                c: int
                for c in range(col - 1, col + 2):
                    if self._board.is_cell_inbounds_rc(r, c):
                        sub_update: Minesweeper.RevealInfo \
                            = self.reveal_cell(r, c)

                        if sub_update.cells_updated is None:
                            return sub_update
                        else:
                            reveal_info.cells_updated.update(
                                sub_update.cells_updated
                            )

            return reveal_info

    def get_board(self) -> "Minesweeper.Board":
        return self._board.copy()

    def copy(self) -> "Minesweeper":
        cls = self.__class__
        result: Minesweeper = cls.__new__(cls)

        # Shallow copy
        var_name: str
        for var_name in cls.__slots__:
            setattr(result, var_name, getattr(self, var_name))

        # Deep copy the board
        result._board = result._board.copy(False)

        return result

    def __str__(self):
        """
        Pretty print the state values.

        :return: string version of state values.
        """

        return str(self._board)


if __name__ == '__main__':
    random.seed(2)
    ms: Minesweeper = Minesweeper(16, 16, 40)

    print(ms)

    terminated: bool = False
    while not terminated:
        move = (int(s) for s in input("Enter a cell to reveal: ").split())
        update: dict[tuple[int, int], Minesweeper.State] | None
        _, update = ms.reveal_cell(*move)

        print(ms)

        if update is None:
            if ms.has_lost:
                print("You lost!")
            else:
                print("You won!")

            terminated = True
