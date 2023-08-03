from dataclasses import dataclass
from functools import cached_property


class Env:
    columns: int = 7
    rows: int = 6
    inarow: int = 4


@dataclass(eq=True)
class State:
    board: list
    mark: int  # 1 or 2
    step: int
    winner: int | None = None  # 1 or 2 or None

    def __init__(self, board, mark, step, winner=None):
        self.board = board
        self.mark = mark
        self.step = step
        self.winner = winner
        self.board_hash = tuple(self.board)

    @cached_property
    def board_2d(self) -> list[list[int]]:
        return [
            self.board[r * Env.columns : (r + 1) * Env.columns] for r in range(Env.rows)
        ]

    @cached_property
    def valid_moves(self) -> list[int]:
        return [col for col in range(Env.columns) if self.board[col] == 0]

    def __hash__(self):
        return hash((self.board_hash, self.mark, self.step))

    def print_board(self):
        for row in self.board_2d:
            print(row)


def init_state():
    return State(board=[0] * Env.rows * Env.columns, mark=1, step=0)


def count(
    board: list[int],
    mark: int,
    row: int,
    col: int,
    offset_row: int,
    offset_column: int,
):
    """Count number of marks in a row from (row, col) using the offsets as
    direction, up to Env.inarow"""
    for i in range(1, Env.inarow):
        r = row + offset_row * i
        c = col + offset_column * i
        if (
            r < 0
            or r >= Env.rows
            or c < 0
            or c >= Env.columns
            or board[c + (r * Env.columns)] != mark
        ):
            return i - 1
    return Env.inarow - 1


def is_win(board: list[int], row: int, col: int, mark: int):
    """Check if a token at (row, col) wins the game for the given player (mark)"""
    return (
        count(board, mark, row, col, 1, 0) >= Env.inarow - 1  # vertical.
        or (count(board, mark, row, col, 0, 1) + count(board, mark, row, col, 0, -1))
        >= Env.inarow - 1  # horizontal.
        or (count(board, mark, row, col, -1, -1) + count(board, mark, row, col, 1, 1))
        >= Env.inarow - 1  # left diagonal.
        or (count(board, mark, row, col, -1, 1) + count(board, mark, row, col, 1, -1))
        >= Env.inarow - 1  # right diagonal.
    )


def get_row(board: list[int], col):
    """Gets the lowest empty row in a column"""
    for row in range(Env.rows - 1, -1, -1):
        if board[col + row * Env.columns] == 0:
            return row
    return None


def play_move(state: State, col: int):
    """
    Plays a move given a game state.

    Args:
        state (State): The current state of the game.
        col (int): The column in which the move is to be played.
    Returns:
        State: The updated game state after playing the move.
    """
    if state.winner is not None:
        raise RuntimeError("Game is over")
    if col not in state.valid_moves:
        raise ValueError(f"Invalid move: {col}")

    row = get_row(state.board, col)
    next_board = state.board[:]
    next_board[col + row * Env.columns] = state.mark

    if is_win(next_board, row, col, state.mark):
        winner = state.mark
    elif not 0 in next_board:
        winner = 0  # board is full, tie game
    else:
        winner = None
    next_mark = 1 if state.mark == 2 else 2
    return State(next_board, next_mark, state.step + 1, winner)
