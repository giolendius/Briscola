from Card import Card
import pygame
import numpy as np
from Agents import *

class Briscola:
    def __init__(self, players: int = 2):
        self.tot_players = players
        if self.tot_players == 4:
            self.teams = True
        else:
            self.teams = False
        self.reset()

    def __str__(self):
        return f"Briscola a {self.tot_players}"

    def reset(self):
        self.card_values = [2, 4, 5, 6, 7, 8, 9, 10, 13, 15]
        self.deck = [Card(v, s) for s in range(4) for v in self.card_values]
        self.turn = 1
        self.done = False
        self.starting_player = 0
        self.current_player = self.starting_player
        self.points = [0 for _ in range(self.tot_players)]
        if self.teams:
            self.points = [0, 0]

        self.briscola = self._draw_from_deck()
        self.hand = {}
        self.table = [Card(None, 0) for _ in range(self.tot_players)]
        self.phase = "P"  # "P" play, "C" Calculates points "D" Draw


        for player in range(self.tot_players):
            self.hand[player] = []
            for card in range(3):
                self.hand[player].append(self._draw_from_deck())

    def play(self, agents: list, render_mode="none", delay1=1000, delay2=2000):
        self.delay1 = delay1
        self.delay2 = delay2
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

                self._game_engine(agents, mode="pygame")

                pygame.display.update()
                pygame.time.Clock().tick(30)
            pygame.quit()

    def _game_engine(self, agents, mode: str = "not"):

        def _end_round_operations():
            # determine who takes

            pt, takes_player = self._who_takes(self.table, self.starting_player)
            pt = sum([carta.points() for carta in self.table])
            self.message = f"Player {takes_player} takes"

            if mode != "pygame" or pygame.time.get_ticks() - self.tempo[0] > self.delay2:
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
                self.table = [Card(None, 0) for _ in range(self.tot_players)]

                if mode == "pygame":
                    self.tempo[0] = pygame.time.get_ticks()

                if mode == "train":
                    self.reward_memory.append(rewards[0])

                if len(self.deck) > self.tot_players - 2:  # usually, proceed to draw
                    self.phase = "D"
                elif self.turn >= len(self.card_values) * 4 // self.tot_players + 1:  # if very last turn, game ended
                    self.phase = "test"
                    if sum(self.points) != 120:
                        print(f"{self.points}. Achtung score is not 120! turni {self.turn}")

                    self.done = True
                    self.run = False
                else:  # if no more card but last 3 turns, don't draw but play
                    self.phase = "P"

        def _draw_at_end_turn():
            # draw

            for player in turn_order:
                if len(self.deck) == 0:  # last round, last player draws briscola
                    for i, card in enumerate(self.hand[player]):
                        if not card.val:
                            self.hand[player][i] = self.briscola
                else:
                    for i, card in enumerate(self.hand[player]):
                        if not card.val:
                            self.hand[player][i] = self._draw_from_deck()
            self.phase = "P"

        turn_order = [pl % self.tot_players for pl in
                      range(self.starting_player, self.starting_player + self.tot_players)]

        if self.phase == "P":
            for i in range(self.tot_players):
                # if it's the turn of current_player:
                if turn_order[i] == self.current_player:
                    if mode != "pygame" or pygame.time.get_ticks() - self.tempo[0] > self.delay1:
                        if mode == "pygame":
                            self.tempo[0] = pygame.time.get_ticks()
                        observation = self.hand[self.current_player] + self.table[
                                                                       1:4]  # here we always exclude player 0, he IS playing
                        azione, q_val = agents[self.current_player].action([self.briscola] + observation)
                        if mode == "train" and self.current_player == 0:
                            self.observation_memory.append(observation)
                            # print(self.observation_memory)
                            self.action_memory.append([azione, q_val])
                        self.table[self.current_player] = self.hand[self.current_player][azione]  # [agents[cur_player]]
                        self.hand[self.current_player][azione] = Card(None, 0)
                        self.current_player = (self.current_player + 1) % self.tot_players
                        if self.current_player == self.starting_player:
                            self.phase = "C"
                        break
        elif self.phase == "C":
            _end_round_operations()

        elif self.phase == "D":
            _draw_at_end_turn()

        elif self.phase == "test":
            print("we reach the test phase")

    def _who_takes(self, table: list[type(Card(0, 0))], starting_player: int) -> (int, int):
        commanding_suit = table[starting_player].suit \
            if self.briscola.suit not in [c.suit for c in table] else self.briscola.suit
        takes_player = np.argmax([card.ia()[commanding_suit] for card in table])
        pt = sum([carta.points() for carta in table])
        self.message = f"Player {takes_player} takes"
        return pt, takes_player

    def train(self, agent, train_episodes, save_name=None, agent_position=0, epochs=5):
        f"""Train the agent '{agent}' (remember to call its class with brackets().
        The agents needs a action method and a self.model attribute"""

        from tqdm import tqdm

        self.history = []

        for episode in tqdm(range(train_episodes)):
            self.reset()
            self.observation_memory = []
            self.reward_memory = []
            self.action_memory = []
            self.done = False
            while not self.done:
                self._game_engine([agent, Agent_Random()], mode="train")

            # Feature tensor
            br = np.repeat([self.briscola.ia()], 20, axis=0)
            table = np.array([np.concatenate([card.ia() for card in ob[3:7]]) for ob in self.observation_memory])
            hand = [np.array([ob[i].ia() for ob in self.observation_memory]) for i in range(3)]
            #hand_played = np.array([ob[self.action_memory[i][0]].ia() for i, ob in enumerate(self.observation_memory)])

            # Target tensor
            #y = np.array(self.reward_memory).reshape(20, 1)
            y = np.array([pair[1] for pair in self.action_memory]) # initialize y with the q_values
            for i in range(20): # for each round, the q_value of the taken action is replaced with the reward
                y[i][self.action_memory[i][0]] = self.reward_memory[i]
            hst = agent.model.fit([br, table]+hand, y, verbose=0, epochs=epochs)
            self.history += hst.history["loss"]
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