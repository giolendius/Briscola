import pandas as pd

from src.types.Agents import *
from src.types.Card import Card, Deck, Hand, BriscolaCard, Table, Observation, TurnMemory, players_hands


protagonist = 0


class BriscolaEnv:
    running: bool
    turn: int
    starting_player: int
    current_player: int
    points: list
    phase: str

    deck: Deck
    table: Table
    briscola: BriscolaCard
    players_hands: players_hands

    turn_memory: TurnMemory
    game_memories: list[TurnMemory]

    def __init__(self, players: int):
        self.tot_players = players
        self.teams = True if self.tot_players == 4 else False
        self.reset()

    def __repr__(self):
        return f"Briscola a {self.tot_players}"

    def reset(self):
        self.game_memories = []
        self.deck = Deck()
        self.turn = 1
        self.starting_player = 0
        self.current_player = self.starting_player
        self.points = [0, 0] if self.teams else [0 for _ in range(self.tot_players)]

        self.briscola = BriscolaCard(self.deck)
        self.players_hands = {}
        self.table = Table([Card(None, 0) for _ in range(self.tot_players)])
        self.phase = "P"  # "P" play, "C" Calculates points "D" Draw

        for player in range(self.tot_players):
            self.players_hands[player] = Hand(self.deck)

    def game_engine(self, agents):

        turn_order = [pl % self.tot_players for pl in
                      range(self.starting_player, self.starting_player + self.tot_players)]

        if self.phase == "P":
            self._play_a_card(agents=agents)

        elif self.phase == "C":
            self._end_round_operations()

        elif self.phase == "D":
            self._draw_at_end_turn(turn_order=turn_order)

        elif self.phase == "test":
            print("we reached the test phase")

    def _play_a_card(self, agents):

        observation = Observation.from_sets(self.briscola,
                                            self.players_hands[self.current_player],
                                            self.table[1:4])  # here we always exclude player 0, he IS playing

        azione, q_val = agents[self.current_player].action(observation)

        if azione == Action.not_chosen_yet:
            self.awaiting_user_input = True
        else:
            if self.current_player == protagonist:
                self.turn_memory = TurnMemory(turn=self.turn,
                                              observation=observation,
                                              action=azione.value)

            self.table[self.current_player] = self.players_hands[self.current_player].play_this_card(azione)
            self.current_player = (self.current_player + 1) % self.tot_players

            if self.current_player == self.starting_player:
                self.phase = "C"

    def _end_round_operations(self):
        # determine who takes

        pt, takes_player = self._who_takes(self.table, self.starting_player)
        pt = sum([carta.points() for carta in self.table])
        self.message = f"Player {takes_player} takes"

        if self.teams:
            self.points[takes_player % 2] += pt
        else:
            self.points[takes_player] += pt
        rewards = [-pt] * self.tot_players
        rewards[takes_player] = pt
        for player in range(self.tot_players):
            if self.table[player].suit == self.briscola.suit:
                rewards[player] -= 1

        # preparing next turn
        self.starting_player = takes_player
        self.current_player = takes_player
        self.turn += 1
        self.message = ""
        self.table = Table([Card(None, 0) for _ in range(self.tot_players)])

        self.turn_memory.reward = rewards[protagonist]
        self.game_memories.append(self.turn_memory)

        if len(self.deck) > self.tot_players - 2:  # usually, proceed to draw
            self.phase = "D"
        elif self.turn >= 10 * 4 // self.tot_players + 1:  # if very last turn, game ended
            self.phase = "test"
            if sum(self.points) != 120:
                print(f"{self.points}. Achtung score is not 120! turni {self.turn}")

            self.running = False
        else:  # if no more card but last 3 turns, don't draw but play
            self.phase = "P"

    def _draw_at_end_turn(self, turn_order):
        """Determine drawing order and implement it"""
        for player in turn_order:
            if len(self.deck) == 0:  # last round, last player draws briscola
                self.players_hands[player].draw_replacement(draw_briscola_last_round=self.briscola)
            else:
                self.players_hands[player].draw_replacement()
        self.phase = "P"

    def _who_takes(self, table: list[type(Card(0, 0))] | Table, starting_player: int) -> (int, int):
        commanding_suit = table[starting_player].suit \
            if self.briscola.suit not in [c.suit for c in table.cards] else self.briscola.suit
        takes_player = np.argmax([card.ia()[commanding_suit] for card in table.cards])
        pt = sum([carta.points() for carta in table.cards])
        self.message = f"Player {takes_player} takes"
        return pt, takes_player

    def run_env(self, agents: list):
        """Run a single game"""
        self.reset()
        self.running = True
        while self.running:
            self.game_engine(agents)

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


