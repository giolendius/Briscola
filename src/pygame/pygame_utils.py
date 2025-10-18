from dataclasses import dataclass
from typing import Tuple
import pygame
from .pygame_constants import *

main_menu_sprite_group = pygame.sprite.Group()


class Button(pygame.sprite.Sprite):
    def __init__(self, width, height, button_color1, button_color2, position, text_size, text_content, text_color):
        super().__init__(main_menu_sprite_group)
        self.image = pygame.Surface([width, height])
        self.button_color1 = button_color1
        self.button_color2 = button_color2
        self.image.fill(self.button_color1)
        self.rect = self.image.get_rect(center=(SCREEN_W//2+position[0], SCREEN_H//2+position[1]))
        self.text = pygame.font.Font("src/asset/Pixel-type.ttf", text_size).render(text_content, True, text_color)
        self.text_rect = self.text.get_rect(center=[width / 2, height / 2])
        self.image.blit(self.text, self.text_rect)

    def is_pressed(self, mouse_pos, pressed):
        if self.rect.collidepoint(mouse_pos) and pressed[0]:
            self.image.fill(self.button_color2)
            self.image.blit(self.text, self.text_rect)
            return True
        else:
            self.image.fill(self.button_color1)
            self.image.blit(self.text, self.text_rect)
            return False


@dataclass
class Id:
    player_id: int
    agent_id: int


class ChoosePlayerButton(Button):
    pressed = {}

    def __init__(self, id: Tuple[int, int], width, height, button_color1, button_color2, position, text_size, text_content, text_color):
        super().__init__(width, height, button_color1, button_color2, position, text_size, text_content, text_color)
        self.id = Id(*id)

    def is_pressed(self, mouse_pos, pressed):
        if self.rect.collidepoint(mouse_pos) and pressed[0]:
            self.pressed[self.id.player_id] = self.id.agent_id
            return True
        return False

    def update(self, *args, **kwargs):
        if self.pressed.get(self.id.player_id) == self.id.agent_id:
            self.image.fill(self.button_color2)
        else:
            self.image.fill(self.button_color1)
        self.image.blit(self.text, self.text_rect)
