from typing import List  # , Dict, Optional, cast, Literal, Iterator

from .pygame_utils import get_card_sheet
from ..types import briscola_cards as c
import pygame

from .pygame_utils import card_position_in_screen, image_position_in_sheet, card_h, card_w
from ..types.types import Action, CurrentPlayer

sprite_group = pygame.sprite.Group()


class Container:
    """A class to contain static objects"""
    card_images_sheet = None


class PySet(c.SetOfCards):
    player_hand_number = 0
    visible = True
    def __setitem__(self, key, card):
        if isinstance(key, CurrentPlayer):
            key = int(key)
        if isinstance(self.cards[key], pygame.sprite.Sprite):
            self.cards[key].kill()
        if isinstance(card, c.Card):
            if card:
                self.cards[key] = PySpriteCard(card.val,
                                               card.suit,
                                               position_dict={'type': type(self), 'card_num': key,
                                                              'player_hand_number': self.player_hand_number
                                                              },
                                               visible=self.visible
                                               )
            else:
                super().__setitem__(key, card)  # empty card
        else:
            raise f"In un {type(self).__name__} puoi assegnare solo oggetti 'carte' o 'PyCarte'"


class PySpriteCard(c.Card, pygame.sprite.Sprite):
    def __init__(self, val: int | None,
                 suit: int,
                 position_dict: dict,
                 visible: bool = True
                 ):
        c.Card.__init__(self, val, suit)

        pygame.sprite.Sprite.__init__(self, sprite_group)
        if self:
            self.image, self.rect = self._get_image_and_rect(position_dict, visible)
        else:
            raise Exception("Non puoi creare una PySpriteCard vuota")

    def _get_image_and_rect(self, position_dict, visible):
        x, y, w, h = image_position_in_sheet(self)
        image = pygame.Surface([w, h])
        if visible:
            image.blit(Container.card_images_sheet, (0, 0), (x, y, w, h))
        # blit on the small rectangle the sub-image of card_sheet with dimensions(w,h)
        # whose top-left corner is on (x,y) of the sheet
        image = pygame.transform.scale(image, (card_w, card_h))

        position = card_position_in_screen(position_dict)
        rect = image.get_rect(center=position)
        return image, rect

    def __repr__(self):
        return f"PySpriteCard({self.val},{self.suit})"

    def update(self):
        pass
        # self.rect.move_ip(20, 10)


class PyHand(c.Hand, PySet):
    def __init__(self,
                 deck: c.Deck,
                 player_num):
        super().__init__(deck)
        self.player_hand_number = player_num
        self.visible = self.player_hand_number == 0
        for card_index in range(len(self.cards)): # this seems tautological, but actually convert card to pycard
            self[card_index] = self[card_index]

    def play_this_card(self, index: Action) -> PySpriteCard:
        played_card = super().play_this_card(index)
        if not isinstance(played_card, PySpriteCard):
            raise Exception("Errore, non è una PySpriteCard")
        played_card.kill()
        return played_card


class PyTable(c.Table, PySet):
    def __init__(self, list_of_cards=None, n_players=None):
        super().__init__(list_of_cards=list_of_cards, n_players=n_players)
        self.visible = True

        for player_num in range(len(self.cards)):
            self[player_num] = self[player_num]
            # this seems tautological, but actually convert card to pycard

# class PyBriscolaCard(c.BriscolaCard, PySpriteCard):
#     def __init__(self, deck: c.SetOfCards):
#         super().__init__(deck)
#         dizio = {'type': c.BriscolaCard, 'player_number': 1, 'num_card': 1}
#         self.sprite = SpriteCard(dizio, self, get_card_sheet())
