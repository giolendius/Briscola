from typing import List

import numpy as np
import pandas as pd

from src.types.Agents import Agent, Human
from src.types.briscola_cards import Card, Deck, Hand, BriscolaCard, Table, Observation, TurnMemory
from src.types.types import Action, CurrentPlayer

protagonist = 0


class BriscolaEnv:
    running: bool
    turn: int
    player: CurrentPlayer
    points: list
    phase: str

    deck: Deck
    table: Table
    briscola: BriscolaCard
    players_hands: dict

    turn_memory: TurnMemory
    game_memories: list[TurnMemory]

    # def __init__(self, n_players: int):
    #
    #     self.reset(n_players)

    def __repr__(self):
        return f"Briscola"

    def reset(self, agents: List[Agent]):
        self.agents = agents
        self.player = CurrentPlayer(len(agents))
        self.game_memories = []
        self.deck = Deck()
        self.turn = 1
        # self.starting_player = 0

        self.points = [0, 0] if self.player.teams else [0 for _ in range(self.player.tot)]
        self.players_hands = {}
        self.phase = "P"  # "P" play, "C" Calculates points "D" Draw
        self.initial_draws()

    def initial_draws(self):
        self.briscola = BriscolaCard(self.deck)
        self.table = Table(n_players=self.player.tot)
        for player in range(self.player.tot):
            self.players_hands[player] = Hand(self.deck)

    @staticmethod
    def initial_checks(agents):
        assert not any([isinstance(a, Human) for a in agents[1:]]), 'Human player first pos'
        if isinstance(agents[0], Human):
            pass

    def game_engine(self):

        if self.phase == "P":
            self._play_a_card()

        elif self.phase == "C":
            self._end_round_operations()

        elif self.phase == "D":
            self._draw_at_end_turn()

        elif self.phase == "test":
            print("we reached the test phase")

    def _play_a_card(self):

        observation = Observation.from_sets(self.briscola,
                                            self.players_hands[self.player],
                                            self.table)  # here we always exclude player 0, he IS playing
        #TODO give self.table[1:4] back
        azione, q_val = self.agents[self.player].action(observation)

        if azione == Action.not_chosen_yet:
            self.awaiting_user_input = True
        else:

            if self.player == protagonist:
                self.turn_memory = TurnMemory(turn=self.turn,
                                              observation=observation,
                                              action=azione.value)

            card = self.players_hands[self.player].play_this_card(azione)
            self.table[self.player] = card
            self.player.next()
            if self.player.is_back_at_start():
                self.phase = "C"


    def _end_round_operations(self):
        # determine who takes

        pt, takes_player = self._who_takes(self.table, self.player.starting)
        pt = sum([carta.points() for carta in self.table])
        self.message = f"Player {takes_player} takes"

        if self.player.teams:
            self.points[takes_player % 2] += pt
        else:
            self.points[takes_player] += pt
        rewards = [-pt] * self.player.tot
        rewards[takes_player] = pt
        for player in range(self.player.tot):
            if self.table[player].suit == self.briscola.suit:
                rewards[player] -= 1

        # preparing next turn
        self.player.starting = takes_player
        self.player.current = takes_player
        self.turn += 1
        self.message = ""
        self.table.empty()

        self.turn_memory.reward = rewards[protagonist]
        self.game_memories.append(self.turn_memory)

        if len(self.deck) > self.player.tot - 2:  # usually, proceed to draw
            self.phase = "D"
        elif self.turn >= 10 * 4 // self.player.tot + 1:  # if very last turn, game ended
            self.phase = "test"
            if sum(self.points) != 120:
                print(f"{self.points}. Achtung score is not 120! turni {self.turn}")
            else:
                print(f"Game ended, final score {self.points} in turns")

            self.running = False
        else:  # if no more card but last 3 turns, don't draw but play
            self.phase = "P"

    def _draw_at_end_turn(self):
        """Determine drawing order and implement it"""
        # for player in self.player:
        #     if len(self.deck) == 0:  # last round, last player draws briscola
        #         self.players_hands[player].draw_replacement(draw_briscola_last_round=self.briscola)
        #     else:
        #         self.players_hands[player].draw_replacement()

        if len(self.deck) == 0:  # last round, last player draws briscola
            self.players_hands[self.player].draw_replacement(draw_briscola_last_round=self.briscola)
        else:
            self.players_hands[self.player].draw_replacement()
        self.player.next()
        if self.player.is_back_at_start():
            self.phase = "P"

    def _who_takes(self, table: list[type(Card(0, 0))] | Table, starting_player: int) -> (int, int):
        commanding_suit = table[starting_player].suit \
            if self.briscola.suit not in [c.suit for c in table.cards] else self.briscola.suit
        takes_player = np.argmax([card.ia()[commanding_suit] for card in table.cards])
        pt = sum([carta.points() for carta in table.cards])
        self.message = f"Player {takes_player} takes"
        return pt, int(takes_player)

    def run_env(self, agents: list):
        """Run a single game"""
        self.reset(agents)

        self.running = True
        while self.running:
            self.game_engine()

    def train_model(self, agent, data, epochs=5, save_name=None):
        if isinstance(data, pd.DataFrame):
            df = data
        else:
            df = pd.read_csv(data)

        br = np.stack(df["briscola"])
        tb = np.stack(df["table0"])
        hand0 = np.stack(df["hand0"])
        hand1 = np.stack(df["hand1"])
        hand2 = np.stack(df["hand2"])
        reward = df["reward"].to_numpy().reshape(-1, 1)
        hst = agent.model.fit([br, tb, hand0, hand1, hand2], reward, verbose=2, epochs=epochs)
        history = hst.history["loss"]

        if save_name:
            agent.model.save_weights(save_name)


