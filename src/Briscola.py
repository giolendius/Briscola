import pygame
import pandas as pd
from tqdm import tqdm
from typing import Literal, Dict

from .Agents import *
from .Card import Card, Deck, Hand, BriscolaCard, SetOfCards

mode = Literal["pygame", "train"]
memories = Literal["observation", "action", "reward"]


class Briscola:
    deck: Deck
    table: SetOfCards
    briscola: BriscolaCard
    turn_memory: Dict[memories,float]

    def __init__(self, players: int = 2):
        self.tot_players = players
        # if self.tot_players == 4:
        #     self.teams = True
        # else:
        #     self.teams = False
        self.teams = True if self.tot_players == 4 else False
        # self.reset()

    def __repr__(self):
        return f"Briscola a {self.tot_players}"

    def reset(self):
        # self.card_values = [2, 4, 5, 6, 7, 8, 9, 10, 13, 15]
        # self.deck = [Card(v, s) for s in range(4) for v in self.card_values]
        self.deck = Deck()
        self.turn = 1
        self.done = False
        self.starting_player = 0
        self.current_player = self.starting_player
        self.points = [0 for _ in range(self.tot_players)]
        if self.teams:
            self.points = [0, 0]

        self.briscola = BriscolaCard(self.deck)
        self.hand = {}
        self.table = SetOfCards([Card(None, 0) for _ in range(self.tot_players)])
        self.phase = "P"  # "P" play, "C" Calculates points "D" Draw

        for player in range(self.tot_players):
            self.hand[player] = Hand(self.deck)

    def play(self, agents: list, render_mode="none", delay_play=1000, delay_end_round=2000):
        self.delay_play = delay_play
        self.delay_end_round = delay_end_round
        self.reset()
        assert self.tot_players == len(agents)
        if render_mode == "pygame":
            import pygame
            pygame.init()
            SCREEN_W = 1000
            SCREEN_H = 800
            pygame.display.set_caption("Love Briscola")
            self.screen = pygame.display.set_mode((SCREEN_W, SCREEN_H))
            self.tempo = [pygame.time.get_ticks()]
            self.message = ""

            def text(txt, posit, color=(0, 0, 0), size=40):
                pos = (SCREEN_W // 2 + posit[0], posit[1])
                font = pygame.font.Font("Pixeltype.ttf", size)
                txtsurf = font.render(txt, False, color)
                text_rect = txtsurf.get_rect(midtop=pos)
                self.screen.blit(txtsurf, text_rect)

            self.run = True
            while self.run:
                for evento in pygame.event.get():
                    if evento.type == pygame.QUIT:
                        self.run = False

                self.screen.fill((62, 184, 99))

                text(f"{agents[0]}:      score {self.points[0]}", (0, 100))
                text(f"{self._hand_to_string(self.hand[0])}", (00, 150))
                text(f"{self.table[0]}", (0, 300))
                text(f"{self.briscola}", (-400, 350))
                text(f"Remaining: {len(self.deck)}, t={self.turn}", (400, 350))
                text(f"{self.message}", (0, 350), size=20)
                text(f"{self.table[1]}", (0, 400))
                text(f"""{self._hand_to_string(self.hand[1])}""", (00, 600))
                text(f"{agents[1]}:      score {self.points[1]}", (0, 650))

                self.game_engine(agents, mode="pygame")

                pygame.display.update()
                pygame.time.Clock().tick(30)
            pygame.quit()

    def game_engine(self, agents, mode: str = "not"):

        turn_order = [pl % self.tot_players for pl in
                      range(self.starting_player, self.starting_player + self.tot_players)]

        if self.phase == "P":
            # for i in range(self.tot_players):
                # if it's the turn of current_player:
                # if turn_order[i] == self.current_player:
            self._play_a_card(mode=mode, agents=agents)
                        # break
        elif self.phase == "C":
            self._end_round_operations(mode=mode)

        elif self.phase == "D":
            self._draw_at_end_turn(turn_order=turn_order)

        elif self.phase == "test":
            print("we reached the test phase")

    def _play_a_card(self, mode: str, agents):
        if mode != "pygame" or pygame.time.get_ticks() - self.tempo[0] > self.delay_play:
            if mode == "pygame":
                self.tempo[0] = pygame.time.get_ticks()
            observation = Observation(self.briscola,
                                      self.hand[self.current_player],
                                      self.table[1:4])  # here we always exclude player 0, he IS playing
            azione, q_val = agents[self.current_player].action(observation)

            if mode == "train" and self.current_player == 0:
                if observation.table:
                    self.turn_memory = {f"table{i}": card.ia() for i, card in enumerate(observation.table.cards)}
                    self.turn_memory["briscola"] = observation.briscola.ia()
                    for i in range(3):
                        self.turn_memory[f"hand{i}"] = observation.hand[i].ia()
                    self.turn_memory["turn"]=self.turn
                else:
                    self.turn_memory = None
            self.table[self.current_player] = self.hand[self.current_player].play_this_card(azione)
            self.current_player = (self.current_player + 1) % self.tot_players
            if self.current_player == self.starting_player:
                self.phase = "C"

    def _end_round_operations(self, mode):
        # determine who takes

        pt, takes_player = self._who_takes(self.table, self.starting_player)
        pt = sum([carta.points() for carta in self.table])
        self.message = f"Player {takes_player} takes"

        if mode != "pygame" or pygame.time.get_ticks() - self.tempo[0] > self.delay_end_round:
            # Points and rewards
            if self.teams:
                print("teams?")
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
            self.table = SetOfCards([Card(None, 0) for _ in range(self.tot_players)])

            if mode == "pygame":
                self.tempo[0] = pygame.time.get_ticks()

            if mode == "train":
                if self.turn_memory:
                    self.turn_memory["reward"] = rewards[0]/10
                    self.memory.append(self.turn_memory)

            if len(self.deck) > self.tot_players - 2:  # usually, proceed to draw
                self.phase = "D"
            elif self.turn >= 10 * 4 // self.tot_players + 1:  # if very last turn, game ended
                self.phase = "test"
                if sum(self.points) != 120:
                    print(f"{self.points}. Achtung score is not 120! turni {self.turn}")

                self.done = True
                self.run = False
            else:  # if no more card but last 3 turns, don't draw but play
                self.phase = "P"

    def _draw_at_end_turn(self, turn_order):
        """Determine drawing order and implement it"""

        for player in turn_order:
            if len(self.deck) == 0:  # last round, last player draws briscola
                self.hand[player].draw_replacement(draw_briscola_last_round=self.briscola)
            else:
                self.hand[player].draw_replacement()
        self.phase = "P"

    def _who_takes(self, table: list[type(Card(0, 0))] | SetOfCards, starting_player: int) -> (int, int):
        commanding_suit = table[starting_player].suit \
            if self.briscola.suit not in [c.suit for c in table.cards] else self.briscola.suit
        takes_player = np.argmax([card.ia()[commanding_suit] for card in table.cards])
        pt = sum([carta.points() for carta in table.cards])
        self.message = f"Player {takes_player} takes"
        return pt, takes_player

    def simulate_games(self, agent, train_episodes, save_name=None, agent_position=0):
        f"""Train the agent '{agent}' (remember to call its class with brackets().
        The agents needs an action method and a self.model attribute"""
        self.df = pd.DataFrame()
        for _ in tqdm(range(train_episodes)):
            self.reset()
            self.memory = []
            self.done = False
            random_agent = RandomAgent()
            while not self.done:
                self.game_engine([agent, random_agent], mode="train")

            df = pd.DataFrame(self.memory)
            self.df = pd.concat([df, self.df], ignore_index = True)

        if save_name:
            self.df.to_csv(save_name)
        else:
            return self.df

            # Feature tensor
            # br = np.repeat([self.briscola.ia()], 20, axis=0)
            # table = np.array([np.concatenate([card.ia() for card in ob[3:7]]) for ob in self.observation_memory])
            # hand = [np.array([ob[i].ia() for ob in self.observation_memory]) for i in range(3)]

            # Target tensor
            # y = np.array(self.reward_memory).reshape(20, 1)
            # y = np.array([pair[1] for pair in self.action_memory])  # initialize y with the q_values
            # for i in range(20):  # for each round, the q_value of the taken action is replaced with the reward
            #     y[i][self.action_memory[i][0]] = self.reward_memory[i]
            # df_match = pd.concat([self.df_match,pd.DataFrame(y, columns=[f"q_val{i}" for i in range(3)])],axis=1)
            # self.df_whole = pd.concat([self.df_whole, df_match], axis=0, ignore_index=True)

    def train_model(self, agent, data, epochs=5, save_name = None):
        if isinstance(data, pd.DataFrame):
            df = data
        else:
            df = pd.read_csv(data)

        br = np.stack(df["briscola"])
        tb = np.stack(df["table0"])
        hand0 = np.stack(df["hand0"])
        hand1 = np.stack(df["hand1"])
        hand2 = np.stack(df["hand2"])
        reward = df["reward"].to_numpy().reshape(-1,1)
        hst = agent.model.fit([br, tb,hand0,hand1,hand2], reward, verbose=2, epochs=epochs)
        history = hst.history["loss"]

        if save_name:
            agent.model.save_weights(save_name)



    def _draw_from_deck(self) -> Card | None:
        from random import randint
        if not self.deck:
            return None  # or raise an exception if you prefer
        index = randint(0, len(self.deck) - 1)
        return self.deck.pop(index)

    def _hand_to_string(self, hand, spaces=10):
        sp = " " * spaces + "|" + " " * spaces
        return f"{hand[0]}" + sp + f"{hand[1]}" + sp + f"{hand[2]}"

    def _append_dict_observ_todf(self, dict_observation, df):
        df = pd.concat([df, pd.DataFrame([dict_observation])], ignore_index=True)
        return df
