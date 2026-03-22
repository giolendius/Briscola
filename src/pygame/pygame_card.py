from typing import List  # , Dict, Optional, cast, Literal, Iterator


import pygame
from numpy import array

from ..types import briscola_cards as bc
from ..types.types import Action, CurrentPlayer
from .pygame_constants import *

card_sprite_group = pygame.sprite.Group()





class Container:
    """A class to contain static objects"""
    card_images_sheet = None


def get_card_sheet():
    # 735, 502
    # ori, coppe, bastoni, spade
    image = pygame.image.load("asset/carte.png").convert_alpha()
    return image


def image_position_in_sheet(card: bc.Card) -> [float, float, float, float]:
    """returns x,y,w,h"""
    w, h = 73.5, 125.5
    x, y = {15: 0, 2: 1, 13: 2, 4: 3, 5: 4, 6: 5, 7: 6, 8: 7, 9: 8, 10: 9, None: 10}[card.val]*w, card.suit * h
    return x, y, w, h


def card_position_in_screen(pyset_type, position_dict) -> array:
    center = array([SCREEN_W//2, SCREEN_H//2])
    if issubclass(pyset_type, bc.Hand):
        card_set_type = array([0, -(position_dict['player_hand_number']*2-1)*HAND_HEIGHT_POS])
        card_pos = array([CARD_DISTANCE * (position_dict.get('card_num', 1) - 1), 0])
    elif issubclass(pyset_type, bc.Table):
        card_set_type = array([0, -(position_dict['card_num']*2-1)*TABLE_HEIGHT_POS])
        card_pos = array([0, 0])
    elif issubclass(pyset_type, bc.BriscolaCard):
        card_set_type = array([-SCREEN_W//8*3, 0])
        card_pos = array([0, 0])
    else:
        raise Exception("Tipo di set di carte non valido")
    return center+card_set_type+card_pos


class PySpriteCard(bc.Card, pygame.sprite.Sprite):
    def __init__(self, val: int | None,
                 suit: int,
                 pyset_type: type,
                 position_dict: dict = None,
                 visible: bool = True
                 ):
        bc.Card.__init__(self, val, suit)

        pygame.sprite.Sprite.__init__(self, card_sprite_group)
        if self:
            self.image, self.rect = self._get_image_and_rect(pyset_type, position_dict, visible)
        else:
            raise Exception("Non puoi creare una PySpriteCard vuota")

    def _get_image_and_rect(self, pyset_type, position_dict, visible: bool = True):
        x, y, w, h = image_position_in_sheet(self)
        image = pygame.Surface([w, h])
        if visible:
            image.blit(Container.card_images_sheet, (0, 0), (x, y, w, h))
        else:
            image.fill((120, 110, 120))
        # blit on the small rectangle the sub-image of card_sheet with dimensions(w,h)
        # whose top-left corner is on (x,y) of the sheet
        image = pygame.transform.scale(image, (card_w, card_h))

        position = card_position_in_screen(pyset_type, position_dict)
        rect = image.get_rect(center=position)
        return image, rect

    def __repr__(self):
        return f"PySpriteCard({self.val},{self.suit})"

    def update(self):
        pass
        # self.rect.move_ip(20, 10)


class PySet(bc.SetOfCards):
    cards: List[PySpriteCard]
    player_hand_number = 0
    visible = True

    def __setitem__(self, key, card):
        if isinstance(key, CurrentPlayer):
            key = int(key)
        if isinstance(self.cards[key], pygame.sprite.Sprite):
            self.cards[key].kill()
        if isinstance(card, bc.Card):
            if card:
                self.cards[key] = PySpriteCard(card.val,
                                               card.suit,
                                               type(self),
                                               position_dict={'card_num': key,
                                                              'player_hand_number': self.player_hand_number
                                                              },
                                               visible=self.visible
                                               )
            else:
                super().__setitem__(key, card)  # empty card
        else:
            raise f"In un {type(self).__name__} puoi assegnare solo oggetti 'carte' o 'PyCarte'"


class PyHand(bc.Hand, PySet):
    def __init__(self,
                 deck: bc.Deck,
                 player_num):
        super().__init__(deck)
        self.player_hand_number = player_num
        self.visible = self.player_hand_number == 0
        for card_index in range(len(self.cards)):  # this seems tautological, but actually convert card to pycard
            self[card_index] = self[card_index]

    def play_this_card(self, index: Action) -> PySpriteCard:
        played_card = super().play_this_card(index)
        if not isinstance(played_card, PySpriteCard):
            raise Exception("Errore, non è una PySpriteCard")
        played_card.kill()
        return played_card


class PyTable(bc.Table, PySet):
    def __init__(self, list_of_cards=None, n_players=None):
        super().__init__(list_of_cards=list_of_cards, n_players=n_players)
        self.visible = True

        for player_num in range(len(self.cards)):
            self[player_num] = self[player_num]
            # this seems tautological, but actually convert card to pycard


class PyBriscolaCard(bc.BriscolaCard, PySpriteCard):
    def __init__(self, deck: bc.SetOfCards):
        bc.BriscolaCard.__init__(self, deck)
        PySpriteCard.__init__(self, self.val,
                              self.suit,
                              type(self))
