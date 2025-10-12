import pygame
from pygame.sprite import Sprite
from typing import Tuple

SCREEN_W = 1000
SCREEN_H = 800

sprite_card_group = pygame.sprite.Group()

def get_card_sheet():
    image = pygame.image.load("src/asset/spade.png").convert_alpha()
    l=8
    image = pygame.transform.scale(image, (l * 10, l * 15))
    return image


class SpriteCard(Sprite):
    def __init__(self, position, game_env):
        super().__init__(sprite_card_group)
        self.game_env = game_env

        l = 32
        # image = pygame.image.load("src/asset/spade.png").convert_alpha()
        # self.image = pygame.transform.scale(image, (l*10, l*15))
        # cozza = pygame.transform.scale(image, (l * 10, l * 15))
        image = pygame.Surface([40, 120])
        image.blit(self.game_env.card_image_sheet, (0,0),(0,0, 40, 120))
        self.image = pygame.transform.scale(image, (l * 10, l * 15))
        self.rect = self.image.get_rect(center=position)

    def update(self):
        pass
        # self.rect.move_ip(20, 10)


def create_sprite():
    pass

