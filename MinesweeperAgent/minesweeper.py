import itertools
import random
from copy import deepcopy
from enum import Enum
from typing import Iterator, Callable


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

    def __init__(self, height: int, width: int, num_bombs: int):
        self.height: int = height
        self.width: int = width
        self.num_bombs: int = num_bombs

        self._board: list[Minesweeper.State] \
            = [Minesweeper.State.UNREVEALED] * (self.height * self.width)

        self.first_reveal: bool = True

        # 0: running
        # -1: lost
        # 1: win
        self._game_state: int = 0

        self._set_bombs()

    def _cell_to_rc(self, cell: int) -> tuple[int, int]:
        """
        :param cell: the state to convert into a (row, column) tuple.
        :type cell: int
        :return: a tuple containing the row and column of the state.
        :rtype: tuple[int, int]
        """
        return cell // self.width, cell % self.width

    def _cell_from_rc(self, r: int, c: int) -> int:
        """
        :param r: the row of the state.
        :type r: int
        :param c: the column of the state.
        :type c: int
        :return: the index of the state.
        :rtype: int
        """
        return r * self.width + c

    def _is_cell_inbounds(self, cell: int) -> bool:
        return 0 <= cell < self.height * self.width

    def _is_cell_inbounds_rc(self, r: int, c: int) -> bool:
        return 0 <= r < self.height and 0 <= c < self.width

    def _set_bombs(self) -> None:
        remaining_cells: int = self.height * self.width
        remaining_bombs: int = self.num_bombs

        cell: int
        for cell in range(self.height * self.width):
            rand_val: int = random.randint(0, remaining_cells - 1)
            if rand_val < remaining_bombs:
                self._board[cell] = Minesweeper.State.BOMB
                remaining_bombs -= 1

                if not remaining_bombs:
                    return

            remaining_cells -= 1

    def _is_bomb(self, state: int) -> bool:
        if not self._is_cell_inbounds(state):
            return False
        else:
            return self._board[state] == Minesweeper.State.BOMB

    def _relocate_bomb(self, row: int, col: int) -> None:
        assert self._is_bomb(self._cell_from_rc(row, col))

        cell: int
        for cell in range(self.height * self.width):
            if self._board[cell] == Minesweeper.State.UNREVEALED:
                self._board[cell] = Minesweeper.State.BOMB
                self._board[self._cell_from_rc(row, col)] \
                    = Minesweeper.State.UNREVEALED
                return

        raise BaseException

    def _count_surrounding_bombs(self, row: int, col: int) -> int:
        surrounding_bombs: int = 0

        r: int
        for r in range(row - 1, row + 2):
            c: int
            for c in range(col - 1, col + 2):
                surrounding_bombs \
                    += 1 if (self._is_cell_inbounds_rc(r, c)
                             and self._is_bomb(self._cell_from_rc(r, c))) \
                    else 0

        return surrounding_bombs

    def _is_winner(self) -> bool:
        num_unrevealed: int = self._board.count(Minesweeper.State.UNREVEALED)
        return num_unrevealed == 0

    def reveal_cell(self,
                    row: int,
                    col: int
                    ) -> dict[tuple[int, int], "Minesweeper.State"] | None:
        cell: int = self._cell_from_rc(row, col)

        if self.first_reveal:
            self.first_reveal = False

            if self._board[cell] == Minesweeper.State.BOMB:
                self._relocate_bomb(row, col)

        if self._is_bomb(cell):
            self._game_state = -1
            return None
        elif self._board[cell] != Minesweeper.State.UNREVEALED:
            return {}
        else:
            self._board[cell] \
                = Minesweeper.State(self._count_surrounding_bombs(row, col))

            if self._is_winner():
                self._game_state = 1
                return None
            elif self._board[cell] != Minesweeper.State.ZERO:
                return {(row, col): self._board[cell]}

            updated: dict[tuple[int, int], Minesweeper.State] \
                = {(row, col): self._board[cell]}

            r: int
            for r in range(row - 1, row + 2):
                c: int
                for c in range(col - 1, col + 2):
                    if self._is_cell_inbounds_rc(r, c):
                        rc_updated: dict[tuple[int, int], Minesweeper.State] \
                            = self.reveal_cell(r, c)

                        if rc_updated is None:
                            return None
                        else:
                            updated.update(rc_updated)

            return updated

    def is_running(self) -> bool:
        return not self._game_state

    def has_lost(self) -> bool:
        return self._game_state == -1

    def has_won(self) -> bool:
        return self._game_state == 1

    def __copy__(self):
        cls = self.__class__
        result = cls.__new__(cls)
        result.__dict__.update(self.__dict__)
        return result

    def __deepcopy__(self, memo):
        cls = self.__class__
        result: Minesweeper = cls.__new__(cls)
        memo[id(self)] = result
        for k, v in self.__dict__.items():
            setattr(result, k, deepcopy(v, memo))
        return result

    def as_formatted_str(self,
                         horizontal_spacing: int = 12,
                         bomb_repr: Callable[[int, int], str]
                            = lambda _1, _2: "BOMB",
                         unrevealed_repr: Callable[[int, int], str]
                            = lambda _1, _2: "UNREVEALED",
                         other_repr: Callable[[int, int, State], str]
                            = lambda _1, _2, val: str(val)
                         ) -> str:
        """
        Gets a string representation of the object.

        :param horizontal_spacing: the amount of space given for each cell.
        :type horizontal_spacing: int
        :param bomb_repr: gets the representation for the bomb.
        :type bomb_repr: Callable[[int, int], str]
        :param unrevealed_repr: gets the representation for an unrevealed cell.
        :type unrevealed_repr: Callable[[int, int], str]
        :param other_repr: gets the representation for other cells.
        :type other_repr: Callable[[int, int, int], str]
        :return: a string representation of the object.
        :rtype str:
        """
        output_format: str = "{:>" + str(horizontal_spacing) + "}"

        state_iterator: Iterator[Minesweeper.State] = iter(self._board)

        row: int
        column: int
        value: Minesweeper.State
        output_str: str = '\n'.join(
            ''.join(
                output_format.format(
                    bomb_repr(row, column) if value == Minesweeper.State.BOMB
                    else unrevealed_repr(row, column)
                    if value == Minesweeper.State.UNREVEALED
                    else other_repr(row, column, value)
                ) for column, value in
                enumerate(itertools.islice(state_iterator, self.width))
            ) for row in range(self.height)
        )

        return output_str

    def __str__(self):
        """
        Pretty print the state values.

        :return: string version of state values.
        :rtype: str
        """

        return self.as_formatted_str()


if __name__ == '__main__':
    random.seed(2)
    ms: Minesweeper = Minesweeper(5, 5, 10)

    print(ms)

    terminated: bool = False
    while not terminated:
        move = (int(s) for s in input("Enter a cell to reveal: ").split())
        update: dict[tuple[int, int], Minesweeper.State] | None \
            = ms.reveal_cell(*move)

        if update is None:
            if ms.has_lost():
                print("You lost!")
            else:
                print("You won!")

            terminated = True

        print(ms)
