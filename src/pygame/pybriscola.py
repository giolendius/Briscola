from typing import List, Tuple
import pygame

from .pygame_utils import Button, main_menu_sprite_group
from ..Briscola import BriscolaEnv
from ..types.Agents import Agent, Human, agents_dict
from ..types.types import Action, PygameState
from .pygame_card import PyTable, PyHand, card_sprite_group, Container, PyBriscolaCard, get_card_sheet

SCREEN_W = 1000
SCREEN_H = 800


class PyBriscolaEnv(BriscolaEnv):
    def __init__(self, delay_play=1000, agents: list = None, state=1):
        self.screen = pygame.display.set_mode((SCREEN_W, SCREEN_H))
        self.tempo = [pygame.time.get_ticks()]
        self.flg_pause = False
        self.awaiting_user_input: bool = False
        self.delay_play = delay_play
        self.agents: List[Agent | None] = [None, None]
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

        pygame.init()
        pygame.display.set_caption("Love Briscola")

        Container.card_images_sheet = get_card_sheet()
        self.create_buttons()

        self.running = True
        while self.running:
            for evento in pygame.event.get():
                if evento.type == pygame.QUIT:
                    self.running = False

            if self.pygame_state == PygameState.MainMenu:
                self.main_menu()
            elif self.pygame_state == PygameState.Playing:
                self.pygame_play_time()

            pygame.display.update()
            pygame.time.Clock().tick(30)
        pygame.quit()

    def create_buttons(self):
        self.buttons = {(i, agent): Button(100, 60, 'red',
                                           (300 * (i + 1), 300 + 80 * j), 30, agent,
                                           'blue')
                        for i in range(2)
                        for j, agent in enumerate(agents_dict.keys())}

    def main_menu(self):

        self.screen.fill((20, 20, 99))
        tasto = pygame.key.get_pressed()
        if tasto[pygame.K_n]:
            self.initial_checks(self.agents)
            self.reset(self.agents)
            self.pygame_state = PygameState.Playing

        mouse_pos, mouse_pressed = pygame.mouse.get_pos(), pygame.mouse.get_pressed()

        for (i, agent_name), button in self.buttons.items():
            if button.is_pressed(mouse_pos, mouse_pressed):
                print(f'{i} is now {agent_name}')
                self.agents[i] = agents_dict[agent_name]()

        text(self.screen, 'Press N to play', (0, SCREEN_H // 2), 'red', 50)

        main_menu_sprite_group.draw(self.screen)

    def pygame_play_time(self):
        """Handles pygame while the actual game is going"""
        tasto = pygame.key.get_pressed()
        primo_giocatore = self.agents[0]
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

        text(self.screen, f"{self.agents[0]}:      score {self.points[0]}", (0, 750))
        text(self.screen, self.players_hands[0].display(), (00, 570), size=26)
        text(self.screen, f"{self.table[0]}", (-120, 450))

        text(self.screen, f"{self.briscola}", (-SCREEN_W // 8 * 3, 300))
        text(self.screen, f"Remaining: {len(self.deck)}, t={self.turn}", (400, 350))

        text(self.screen, f"{self.table[1]}", (-120, 350))
        text(self.screen, self.players_hands[1].display(False), (00, 200), size=26)
        text(self.screen, f"{self.agents[1]}:      score {self.points[1]}", (0, 40))

        card_sprite_group.draw(self.screen)
        card_sprite_group.update()

        if ((pygame.time.get_ticks() - self.tempo[0] > self.delay_play)
                and not self.flg_pause
                and not self.awaiting_user_input):
            self.game_engine()
            self.tempo[0] = pygame.time.get_ticks()


def text(screen, txt: str, posit: Tuple[int, int], color=(0, 0, 0), size=40):
    """Writes text on the pygame screen"""
    pos = (SCREEN_W // 2 + posit[0], posit[1])
    font = pygame.font.Font("src/asset/Pixel-type.ttf", size)
    font = pygame.font.Font("src/asset/Pixel-type.ttf", size)
    txt_surf = font.render(txt, False, color)
    text_rect = txt_surf.get_rect(midtop=pos)
    screen.blit(txt_surf, text_rect)
