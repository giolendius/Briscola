from typing import List  # , Dict, Optional, cast, Literal, Iterator

from .pygame_utils import get_card_sheet
from ..types import Card as c
import pygame

from .pygame_utils import card_position_in_screen, image_position_in_sheet, card_h, card_w
from ..types.enums import Action

print('un import random')
#sprite_card_group = pygame.sprite.Group()

class PySpriteCard(c.Card, pygame.sprite.Sprite):
    def __init__(self, val: int | None,
                 suit: int,
                 position_dict: dict,
                 sprite_card_group,
                 all_card_sheet: pygame.image,
                 ):
        c.Card.__init__(self, val, suit)
        pygame.sprite.Sprite.__init__(self, sprite_card_group)
        self.image, self.rect = self._get_image_and_rect(position_dict, all_card_sheet)

    def _get_image_and_rect(self, position_dict, all_card_sheet: pygame.image):
        x, y, w, h = image_position_in_sheet(self)
        image = pygame.Surface([w, h])
        image.blit(all_card_sheet, (0, 0), (x, y, w, h))
        image = pygame.transform.scale(image, (card_w, card_h))

        position = card_position_in_screen(position_dict)
        rect = image.get_rect(center=position)
        return image, rect

    def __repr__(self):
        return f"PySpriteCard({self.val},{self.suit})"

    def update(self):
        pass
        # self.rect.move_ip(20, 10)


class PyTable(c.Table):
    def __init__(self, sprite_card_group, list_of_cards=None, n_players=None):
        super().__init__(list_of_cards=list_of_cards, n_players=n_players)
        self.sprite_card_group = sprite_card_group

        self.cards = [PySpriteCard(card.val,
                                   card.suit,
                                   position_dict={'type': c.Table, 'player_number': card_num},
                                   all_card_sheet=get_card_sheet(),
                                   sprite_card_group=self.sprite_card_group) for card_num, card in
                      enumerate(self.cards)]

    def empty(self):
        self.cards = [PySpriteCard(None,
                                   0,
                                   position_dict={'type': c.Table, 'player_number': card_num},
                                   all_card_sheet=get_card_sheet(),
                                   sprite_card_group=self.sprite_card_group) for card_num, card in
                      enumerate(self.cards)]


    def __setitem__(self, key, card):
        if not isinstance(card, PySpriteCard):
            raise Exception("Puoi assegnare solo oggetti 'pycarte'")
        else:
            # card.kill()
            self.cards[key] = PySpriteCard(card.val,
                                           card.suit,
                                           position_dict={'type': c.Table, 'player_number': key},
                                           all_card_sheet=get_card_sheet(),
                                           sprite_card_group=self.cards[key].groups()[0])


# class PyBriscolaCard(c.BriscolaCard, PySpriteCard):
#     def __init__(self, deck: c.SetOfCards):
#         super().__init__(deck)
#         dizio = {'type': c.BriscolaCard, 'player_number': 1, 'num_card': 1}
#         self.sprite = SpriteCard(dizio, self, get_card_sheet())

class PyHand(c.Hand):
    def __init__(self,
                 deck: c.Deck,
                 sprite_card_group: pygame.sprite.Group,
                 player_num):
        super().__init__(deck)

        self.sprite_card_group=sprite_card_group
        self.player_num = player_num
        self.cards = [PySpriteCard(card.val,
                                   card.suit,
                                   position_dict={'type': c.Hand, 'player_number': self.player_num, 'card_num': card_num},
                                   all_card_sheet=get_card_sheet(),
                                   sprite_card_group=sprite_card_group) for card_num, card in enumerate(self.cards)]

    def __setitem__(self, key, value):
        if isinstance(value, c.Card):
            self.cards[key] = PySpriteCard(value.val,
                                           value.suit,
                                           position_dict={'type': c.Hand, 'player_number': self.player_num, 'card_num': key},
                                           all_card_sheet=get_card_sheet(),
                                           sprite_card_group=self.sprite_card_group)

    def play_this_card(self, index: Action) -> PySpriteCard:
        played_card = super().play_this_card(index)
        if not isinstance(played_card, PySpriteCard):
            raise Exception("Errore, non è una PySpriteCard")
        played_card.kill()
        return played_card

# class SetOfCards:
#     def __init__(self, list_of_cards: List[Card] = None):
#         self.cards: List[Card] = list_of_cards if list_of_cards else []
#         self.name = 'SetOfCards'
#
#     def draw_random(self) -> Card | None:
#         """Remove a random card from this set and returns it"""
#         from random import randint
#         if not self.cards:
#             return None  # or raise an exception if you prefer
#         index = randint(0, len(self.cards) - 1)
#         return self.cards.pop(index)
#
#     def __len__(self):
#         return len(self.cards)
#
#     def __iter__(self) -> Iterator[Card]:
#         return iter(self.cards)
#
#     def __add__(self, other):
#         if isinstance(other, Card):
#             self.cards.append(other)
#             return self
#         elif isinstance(other, SetOfCards):
#             return SetOfCards(self.cards+other.cards)
#         elif isinstance(other, list):
#             return SetOfCards(self.cards+other)
#         else:
#             raise Exception("Puo aggiungere solo una carta")
#
#     def __repr__(self):
#         return type(self).__name__+"-object with "+str(len(self))+" cards\n"+repr(self.cards)
#
#     def __getitem__(self, index: int | slice):
#         if isinstance(index, (int, np.int64)):
#             return self.cards[index]
#         elif isinstance(index, slice):
#             return type(self)(self.cards[index])
#
#     def __setitem__(self, key, card: Card):
#         if not isinstance(card, Card):
#             raise Exception("Puoi assegnare solo una carta")
#         self.cards[key] = card
#
#     def __bool__(self):
#         return bool(self.cards[0])
#
#     def ia(self):
#         """Returns a list of the cards.ia()"""
#         return [card.ia().reshape(1,4) for card in self.cards]
