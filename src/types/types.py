from enum import Enum
from typing import Dict, Iterator, Iterable


from typing_extensions import SupportsIndex


class Action(Enum):
    first_card = 0
    second_card = 1
    third_card = 2
    not_chosen_yet = None


class PygameState(Enum):
    MainMenu = 0
    Playing = 1


class CurrentPlayer:
    def __init__(self, total: int, current: int = 0):
        self.tot = total
        self.starting = current
        self.current = current

    def next(self):
        self.current = (self.current + 1) % self.tot

    def prossimo(self):
        return CurrentPlayer(self.tot, (int(self) + 1) % self.tot)

    def is_back_at_start(self) -> bool:
        return self.current == self.starting

    def __int__(self):
        """Allow usage in arithmetic operations."""
        return self.current

    def __index__(self) -> int:
        """Allow usage in list/tuple indexing and slicing."""
        return int(self)

    def __hash__(self) -> int:
        """Allow usage in dict"""
        return hash(int(self.current))

    def __eq__(self, other) -> bool:
        return int(self) == int(other)

    def __repr__(self) -> str:
        """Make printing show the current value nicely."""
        return f"CurPlayer: {self.current}, starting {self.starting}/{self.tot}"

    # def __iter__(self):
    #     for i in range(self.tot):
    #         yield (self.starting+i) % self.tot


a=CurrentPlayer(4,2)
b = [5,6,7][a]
c = {0: 'a', 1: 'b', 2: 'c'}
d=c[a]
a.next()
a.next()
a==a.starting