from numpy import array
import pygame


from ..types.briscola_cards import Card, Hand, Table, BriscolaCard

SCREEN_W = 1000
SCREEN_H = 800
HAND_HEIGHT_POS = 260
TABLE_HEIGHT_POS = 60
CARD_DISTANCE = 100

# l = 5
card_w, card_h = 15.5*5, 24.5*5


def get_card_sheet():
    #735, 502
    #ori, coppe, bastoni, spade
    image = pygame.image.load("src/asset/carte.png").convert_alpha()
    return image


def image_position_in_sheet(card: Card) -> [float, float, float, float]:
    """returns x,y,w,h"""
    w, h = 73.5, 125.5
    x,y = {15:0, 2:1, 13:2, 4:3, 5:4, 6:5,7:6, 8:7, 9:8, 10:9, None:10}[card.val]*w, card.suit * h
    return x, y, w, h


def card_position_in_screen(pyset_type, position_dict) -> array:
    center = array([SCREEN_W//2, SCREEN_H//2])
    if issubclass(pyset_type,Hand):
        card_set_type = array([0, -(position_dict['player_hand_number']*2-1)*HAND_HEIGHT_POS])
        card_pos = array([CARD_DISTANCE * (position_dict.get('card_num', 1) - 1), 0])
    elif issubclass(pyset_type,Table):
        card_set_type = array([0, -(position_dict['card_num']*2-1)*TABLE_HEIGHT_POS])
        card_pos = array([0, 0])
    elif issubclass(pyset_type,BriscolaCard):
        card_set_type = array([-SCREEN_W//8*3, 0])
        card_pos = array([0, 0])
    else:
        raise Exception("Tipo di set di carte non valido")
    return center+card_set_type+card_pos

