from abc import ABC, abstractmethod
import numpy as np
from random import choice

from .Card import Observation
from .enums import Action

namelist = ["Pieruc", "Iuanin", "Barbacec", "Vecia", "Pinotu", "Parin", "Lenciu"]


class Agent(ABC):
    def __init__(self, name=None):
        if name:
            self.name = name
        else:
            self.name = choice(namelist)

    @abstractmethod
    def action(self, observation: Observation) -> (Action, np.array):
        pass

    def __str__(self):
        return self.name

    def __repr__(self):
        return type(self).__name__ + ": "+self.name


class RandomAgent(Agent):
    """An agent who plays a random card of the available ones"""
    def action(self, observation: Observation) -> (Action, float):
        poss = observation.indices_card_in_hand()
        return Action(choice(poss)), np.array([0, 0, 0])


class AgentOnlyFirst(Agent):
    def action(self, observation: Observation):
        return Action(min(observation.indices_card_in_hand())), 0


class Human(Agent):
    action_chosen: Action = Action.not_chosen_yet

    def action(self, observation: Observation):
        action_chosen = self.action_chosen
        if action_chosen != Action.not_chosen_yet:
            self.action_chosen = Action.not_chosen_yet
        return action_chosen, np.array([0, 0, 0])
