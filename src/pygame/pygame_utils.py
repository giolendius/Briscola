from numpy import array
import pygame
from pygame.sprite import Sprite


from ..types.Card import Card, Hand, Table, BriscolaCard, SetOfCards

SCREEN_W = 1000
SCREEN_H = 800
HAND_HEIGHT_POS = 260
CARD_DISTANCE = 100

l = 5
card_w, card_h = 15.5*5, 24.5*5
sprite_card_group = pygame.sprite.Group()

def get_card_sheet():
    #735, 502
    #ori, coppe, bastoni, spade
    image = pygame.image.load("src/asset/carte.png").convert_alpha()
    return image

def image_position_in_sheet(card: Card) -> [float, float, float, float]:
    """returns x,y,w,h"""
    w, h = 73.5, 125.5
    x,y = {15:0, 2:1, 13:2, 4:3, 5:4, 6:5,7:6, 8:7, 9:8, 10:9}[card.val]*w, card.suit * h
    return x, y, w ,h

def card_position_in_screen(position_dict) -> array:
    center = array([SCREEN_W//2, SCREEN_H//2])
    tipo = position_dict['type']
    if tipo == Hand:
        card_set_type = array([0, -(position_dict['hand_number']*2-1)*HAND_HEIGHT_POS])
    elif tipo == Table:
        card_set_type = array([0, 0])
    elif tipo == BriscolaCard:
        card_set_type = array([-SCREEN_W//2 +2*card_w, 50])
    else:
        raise Exception("Tipo di set di carte non valido")
    card_pos = array([CARD_DISTANCE * (position_dict['num_card']-1), 0])
    return center+card_set_type+card_pos



class SpriteCard(Sprite):
    def __init__(self, position_dict: dict, card: Card, all_card_sheet: pygame.image):
        super().__init__(sprite_card_group)

        x, y, w, h = image_position_in_sheet(card)
        self.image = pygame.Surface([w,h])
        self.image.blit(all_card_sheet, (0,0),(x,y, w, h))
        self.image = pygame.transform.scale(self.image, (card_w, card_h))

        position = card_position_in_screen(position_dict)
        self.rect = self.image.get_rect(center=position)

    def update(self):
        pass
        # self.rect.move_ip(20, 10)


