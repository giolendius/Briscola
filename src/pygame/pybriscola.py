from typing import List, Tuple

from ..Briscola import BriscolaEnv
from ..types.Agents import Agent, Human
from ..types.types import Action, PygameState
from .pygame_utils import *
from .pygame_card import PyTable, PyHand, sprite_group, Container, PyBriscolaCard

SCREEN_W = 1000
SCREEN_H = 800


class PyBriscolaEnv(BriscolaEnv):
    def __init__(self, n_players: int = 2, delay_play=1000, agents: list = None, state=1):
        super().__init__(n_players)
        self.screen = pygame.display.set_mode((SCREEN_W, SCREEN_H))
        self.tempo = [pygame.time.get_ticks()]
        self.message = "uuuh"
        self.flg_pause = False
        self.awaiting_user_input: bool = False
        self.delay_play = delay_play
        self.agents = agents
        self.pygame_state = PygameState(state)
        # self.delay_end_round = delay_end_round

    def initial_draws(self):
        self.briscola = PyBriscolaCard(self.deck)
        self.table = PyTable(n_players=self.player.tot)
        for player in range(self.player.tot):
            self.players_hands[player] = PyHand(self.deck, player_num=player)

    @staticmethod
    def initial_checks(agents):
        assert not any([isinstance(a, Human) for a in agents[1:]]), 'Human player first pos'
        if isinstance(agents[0], Human):
            pass

    def run_env(self, agents: List[Agent]):
        self.reset(len(agents))
        pygame.init()
        pygame.display.set_caption("Love Briscola")
        Container.card_images_sheet = get_card_sheet()
        self.initial_draws()
        self.initial_checks(agents)

        self.running = True
        while self.running:
            for evento in pygame.event.get():
                if evento.type == pygame.QUIT:
                    self.running = False

            if self.pygame_state == PygameState.MainMenu:
                self.main_menu()
            elif self.pygame_state == PygameState.Playing:
                self.pygame_play_time(agents)

            pygame.display.update()
            pygame.time.Clock().tick(30)
        pygame.quit()

    def main_menu(self):
        self.screen.fill((20, 20, 99))
        tasto = pygame.key.get_pressed()
        if tasto[pygame.K_n]:
            self.reset(self.player.tot)
            self.initial_draws()
            #FIXME c'è uno strana carta che vine data quando si gioca partendo dal main menu
            self.pygame_state = PygameState.Playing

        text(self.screen, 'Press N to play', (0,SCREEN_H//2), 'red', 50)

    def pygame_play_time(self, agents):
        """Handles pygame while the actual game is going"""
        tasto = pygame.key.get_pressed()
        primo_giocatore = agents[0]
        if tasto[pygame.K_p]:
            print("Hai premuto P, metto in pausa")
            self.flg_pause = not self.flg_pause
        elif isinstance(primo_giocatore, Human) and self.awaiting_user_input:
            if tasto[pygame.K_1]:
                self.awaiting_user_input = False
                primo_giocatore.action_chosen = Action(0)
            elif tasto[pygame.K_2]:
                self.awaiting_user_input = False
                primo_giocatore.action_chosen = Action(1)
            elif tasto[pygame.K_3]:
                self.awaiting_user_input = False
                primo_giocatore.action_chosen = Action(2)
        elif tasto[pygame.K_PLUS]:
            self.delay_play = max(self.delay_play / 2, 250)
            print(f'+ Speed increased to {1000 / self.delay_play}')
        elif tasto[pygame.K_MINUS]:
            self.delay_play = min(self.delay_play * 2, 2000)
            print(f'- Speed decreased to {1000 / self.delay_play}')
        elif tasto[pygame.K_ESCAPE]:
            self.running = False

        self.screen.fill((62, 184, 99))

        if self.flg_pause:
            text(self.screen, f"Game pause! Press P to resume", (50, 50))

        if self.awaiting_user_input:
            text(self.screen, f"It's your turn!", (350, 650), (180, 20, 20), size=60)

        text(self.screen, f"{agents[0]}:      score {self.points[0]}", (0, 750))
        text(self.screen, self.players_hands[0].display(), (00, 570), size=26)
        text(self.screen, f"{self.table[0]}", (-120, 450))

        text(self.screen, f"{self.briscola}", (-SCREEN_W//8*3, 300))
        text(self.screen, f"Remaining: {len(self.deck)}, t={self.turn}", (400, 350))
        text(self.screen, f"{self.message}", (0, 350), size=20)

        text(self.screen, f"{self.table[1]}", (-120, 350))
        text(self.screen, self.players_hands[1].display(False), (00, 200), size=26)
        text(self.screen, f"{agents[1]}:      score {self.points[1]}", (0, 40))

        sprite_group.draw(self.screen)
        sprite_group.update()

        if ((pygame.time.get_ticks() - self.tempo[0] > self.delay_play)
                and not self.flg_pause
                and not self.awaiting_user_input):
            self.game_engine(agents)
            self.tempo[0] = pygame.time.get_ticks()


def text(screen, txt: str, posit: Tuple[int, int], color=(0, 0, 0), size=40):
    """Writes text on the pygame screen"""
    pos = (SCREEN_W // 2 + posit[0], posit[1])
    font = pygame.font.Font("src/asset/Pixeltype.ttf", size)
    txt_surf = font.render(txt, False, color)
    text_rect = txt_surf.get_rect(midtop=pos)
    screen.blit(txt_surf, text_rect)
