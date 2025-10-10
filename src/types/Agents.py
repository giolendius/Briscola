from abc import ABC, abstractmethod
import numpy as np
from random import choice

from src.types.Card import Observation

namelist = ["Pieruc", "Iuanin", "Barbacec", "Vecia", "Pinotu", "Parin", "Lenciu"]

class Agent(ABC):
    def __init__(self, name=None):
        if name:
            self.name = name
        else:
            self.name = choice(namelist)

    @abstractmethod
    def action(self, observation: Observation) -> (int, np.array):
        pass

    def __str__(self):
        return self.name

    def __repr__(self):
        return type(self).__name__ + ": "+self.name



class RandomAgent(Agent):
    """An agent who plays a random card of the available ones"""
    def action(self, observation: Observation) -> (int, float):
        poss = observation.indices_card_in_hand()
        return choice(poss), np.array([0,0,0])


class AgentOnlyFirst(Agent):
    def action(self, observation: Observation) -> (int, float):
        return min([i - 1 for i in range(1, 4) if observation[i].val]), 0