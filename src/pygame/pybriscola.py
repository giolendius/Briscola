import pygame

from ..Card import SetOfCards, Card, Deck, BriscolaCard
from ..Briscola import BriscolaGame

SCREEN_W = 1000
SCREEN_H = 800


class PyBriscola(BriscolaGame):
    def __init__(self, n_players: int = 2):
        super().__init__(n_players)
        self.screen = pygame.display.set_mode((SCREEN_W, SCREEN_H))
        self.tempo = [pygame.time.get_ticks()]
        self.message = ""

    def play(self, agents: list, render_mode="none", delay_play=1000, delay_end_round=2000):
        self.delay_play = delay_play
        self.delay_end_round = delay_end_round

        assert self.tot_players == len(agents)

        pygame.init()

        pygame.display.set_caption("Love Briscola")

        self.run = True
        while self.run:
            for evento in pygame.event.get():
                if evento.type == pygame.QUIT:
                    self.run = False

            self.screen.fill((62, 184, 99))

            self.text(f"{agents[0]}:      score {self.points[0]}", (0, 100))
            # self.text(f"{self._hand_to_string(self.hand[0])}", (00, 150))
            self.text(f"{self.table[0]}", (0, 300))
            self.text(f"{self.briscola}", (-400, 350))
            self.text(f"Remaining: {len(self.deck)}, t={self.turn}", (400, 350))
            self.text(f"{self.message}", (0, 350), size=20)
            self.text(f"{self.table[1]}", (0, 400))
            # self.text(f"""{self._hand_to_string(self.hand[1])}""", (00, 600))
            self.text(f"{agents[1]}:      score {self.points[1]}", (0, 650))

            if pygame.time.get_ticks() - self.tempo[0] > self.delay_play:
                self.game_engine(agents, mode="amigos")
                self.tempo[0] = pygame.time.get_ticks()

            pygame.display.update()
            pygame.time.Clock().tick(30)
        pygame.quit()

    def text(self, txt, posit, color=(0, 0, 0), size=40):
        pos = (SCREEN_W // 2 + posit[0], posit[1])
        font = pygame.font.Font("Pixeltype.ttf", size)
        txtsurf = font.render(txt, False, color)
        text_rect = txtsurf.get_rect(midtop=pos)
        self.screen.blit(txtsurf, text_rect)