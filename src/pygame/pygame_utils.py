from numpy import array
import pygame
from pygame.sprite import Sprite


from ..types.briscola_cards import Card, Hand, Table, BriscolaCard, SetOfCards

SCREEN_W = 1000
SCREEN_H = 800
HAND_HEIGHT_POS = 260
TABLE_HEIGHT_POS = 60
CARD_DISTANCE = 100

l = 5
card_w, card_h = 15.5*5, 24.5*5
# sprite_card_group = pygame.sprite.Group()
counter = [0]

def get_card_sheet():
    #735, 502
    #ori, coppe, bastoni, spade
    counter[0] += 1
    print(f'carico immagine {counter[0]}')
    image = pygame.image.load("src/asset/carte.png").convert_alpha()
    return image

def image_position_in_sheet(card: Card) -> [float, float, float, float]:
    """returns x,y,w,h"""
    w, h = 73.5, 125.5
    x,y = {15:0, 2:1, 13:2, 4:3, 5:4, 6:5,7:6, 8:7, 9:8, 10:9, None:10}[card.val]*w, card.suit * h
    return x, y, w, h

def card_position_in_screen(position_dict) -> array:
    center = array([SCREEN_W//2, SCREEN_H//2])
    tipo = position_dict['type']
    if issubclass(tipo,Hand):
        card_set_type = array([0, -(position_dict['player_hand_number']*2-1)*HAND_HEIGHT_POS])
        card_pos = array([CARD_DISTANCE * (position_dict.get('card_num', 1) - 1), 0])
    elif issubclass(tipo,Table):
        card_set_type = array([0, -(position_dict['card_num']*2-1)*TABLE_HEIGHT_POS])
        card_pos = array([0, 0])
    elif issubclass(tipo,BriscolaCard):
        card_set_type = array([-SCREEN_W//2 +2*card_w, 50])
    else:
        raise Exception("Tipo di set di carte non valido")

    return center+card_set_type+card_pos


def assign_sprite(card_set: SetOfCards|Card, player_number: int = 1,):
    for i, card in enumerate(card_set):
        dizio = {'type': type(card_set), 'player_number': player_number, 'num_card': i}
        card.sprite = SpriteCard(dizio, card, get_card_sheet())


# class SpriteCard(Sprite):
#     def __init__(self, position_dict: dict, card: Card, all_card_sheet: pygame.image):





