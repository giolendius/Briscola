import pygame
from typing import List

from ..Briscola import BriscolaEnv
from ..types.Agents import Agent, Human
from ..types.enums import Action

SCREEN_W = 1000
SCREEN_H = 800


class PyBriscolaEnv(BriscolaEnv):
    def __init__(self, n_players: int = 2, delay_play=1000, delay_end_round=2000):
        super().__init__(n_players)
        self.screen = pygame.display.set_mode((SCREEN_W, SCREEN_H))
        self.tempo = [pygame.time.get_ticks()]
        self.message = ""
        self.flg_pause = False
        self.awaiting_user_input: bool = False
        self.delay_play = delay_play
        self.delay_end_round = delay_end_round

    def initial_checks(self, agents):
        assert self.tot_players == len(agents)
        assert not any([isinstance(a, Human) for a in agents[1:]]), 'Human player first pos'

        if isinstance(agents[0], Human):
            pass

    def run_env(self, agents: List[Agent]):
        self.initial_checks(agents)
        pygame.init()
        pygame.display.set_caption("Love Briscola")

        self.running = True
        while self.running:
            for evento in pygame.event.get():
                if evento.type == pygame.QUIT:
                    self.running = False
                elif evento.type == pygame.KEYDOWN:
                    primo_giocatore = agents[0]
                    if evento.key == pygame.K_p:
                        # Azione da eseguire quando si preme 'N'
                        print("Hai premuto P, metto in pausa")
                        self.flg_pause = not self.flg_pause
                    elif isinstance(primo_giocatore, Human) and self.awaiting_user_input:
                        if evento.key == pygame.K_1:
                            self.awaiting_user_input = False
                            primo_giocatore.action_chosen = Action(0)
                        elif evento.key == pygame.K_2:
                            self.awaiting_user_input = False
                            primo_giocatore.action_chosen = Action(1)
                        elif evento.key == pygame.K_3:
                            self.awaiting_user_input = False
                            primo_giocatore.action_chosen = Action(2)

            self.pygame_play_time(agents)

            pygame.display.update()
            pygame.time.Clock().tick(30)
        pygame.quit()

    def pygame_play_time(self, agents):
        """Handles pygame while the actual game is going"""
        self.screen.fill((62, 184, 99))

        if self.flg_pause:
            text(self.screen, f"Game pause! Press P to resume", (50,50))

        if self.awaiting_user_input:
            text(self.screen, f"It's your turn!", (0, 700), (180, 20, 20), size=60)

        text(self.screen, f"{agents[0]}:      score {self.points[0]}", (0, 650))
        text(self.screen, self.players_hands[0].display(), (00, 600))
        text(self.screen, f"{self.table[0]}", (0, 400))

        text(self.screen, f"{self.briscola}", (-400, 350))
        text(self.screen, f"Remaining: {len(self.deck)}, t={self.turn}", (400, 350))
        text(self.screen, f"{self.message}", (0, 350), size=20)

        text(self.screen, f"{self.table[1]}", (0, 300))
        text(self.screen, self.players_hands[1].display(True), (00, 150))
        text(self.screen, f"{agents[1]}:      score {self.points[1]}", (0, 100))

        if ((pygame.time.get_ticks() - self.tempo[0] > self.delay_play)
                and not self.flg_pause
                and not self.awaiting_user_input):
            self.game_engine(agents)
            self.tempo[0] = pygame.time.get_ticks()


def text(screen, txt: str, posit, color=(0, 0, 0), size=40):
    """Writes text on the pygame screen"""
    pos = (SCREEN_W // 2 + posit[0], posit[1])
    font = pygame.font.Font("Pixeltype.ttf", size)
    txt_surf = font.render(txt, False, color)
    text_rect = txt_surf.get_rect(midtop=pos)
    screen.blit(txt_surf, text_rect)