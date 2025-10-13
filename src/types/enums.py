from enum import Enum

class Action(Enum):
    first_card = 0
    second_card = 1
    third_card = 2
    not_chosen_yet = None

class Positions(Enum):
    hand_0 = (-100,0)
    hand_1 = (0,0)
    hand_2 = (100, 0)
