import numpy as np
from typing import List
from dataclasses import dataclass
# from loguru import logger

_all_possible_val = [2, 4, 5, 6, 7, 8, 9, 10, 13, 15]


class Card:
    suit_dic = {0: "Bastoni", 1: "Coppe", 2: "Ori", 3: "Spade"}
    card_dic = {x: f"{x}" for x in range(4, 8)} | {2: "2", 8: "Fante", 9: "Cav", 10: "Re", 13: "3", 15: "Asso"}
    points_dic = {8: 2, 9: 3, 10: 4, 13: 10, 15: 11}

    def __init__(self, val: int | None, suit: int):
        if val in _all_possible_val+[0, None]:
            self.val = val
        else:
            raise Exception(f"Il valore {val} della carta non è valido")
        if suit in Card.suit_dic:
            self.suit: int = suit
        else:
            raise Exception("Il seme della carta non è valido. Dichiarare seme con intero 0-3")

    def ia(self):
        """Output the card as tensor of shape (4,)"""
        a = np.array([0, 0, 0, 0])
        if self.val:
            a[self.suit] = self.val
        return a

    def __str__(self):
        if not self.val:
            return "___"
        elif self.val == "Card":
            return "Card"
        return Card.card_dic[self.val] + " di " + Card.suit_dic[self.suit]

    def __repr__(self) -> str:
        repr = f"Card({self.val},{self.suit})" if self else "NoCard"
        return repr

    def __bool__(self):
        if self.val:
            return True
        else:
            return False

    def points(self):
        return Card.points_dic.get(self.val, 0)


class SetOfCards:
    def __init__(self, list_of_cards: List[Card] = None):
        self.cards = list_of_cards if list_of_cards else []

    def draw_random(self) -> Card | None:
        """Remove a random card from this set and returns it"""
        from random import randint
        if not self.cards:
            return None  # or raise an exception if you prefer
        index = randint(0, len(self.cards) - 1)
        return self.cards.pop(index)

    def __len__(self):
        return len(self.cards)

    def __add__(self, other):
        if isinstance(other, Card):
            self.cards.append(other)
            return self
        elif isinstance(other, SetOfCards):
            return SetOfCards(self.cards+other.cards)
        elif isinstance(other, list):
            return SetOfCards(self.cards+other)
        else:
            raise Exception("Puo aggiungere solo una carta")

    def __repr__(self):
        return type(self).__name__+"-object with "+str(len(self))+" cards\n"+repr(self.cards)

    def __getitem__(self, index: int | slice):
        if isinstance(index, (int, np.int64)):
            return self.cards[index]
        elif isinstance(index, slice):
            return SetOfCards(self.cards[index])

    def __setitem__(self, key, value):
        self.cards[key] = value

    def __bool__(self):
        return bool(self.cards[0])

    def ia(self):
        """Returns a list of the cards.ia()"""
        return [card.ia().reshape(1,4) for card in self.cards]



class Deck(SetOfCards):
    def __init__(self):
        super().__init__()
        self.cards = [Card(v, s) for s in range(4) for v in _all_possible_val]


class BriscolaCard(Card, SetOfCards):
    def __init__(self, deck):
        briscola_card = deck.draw_random()
        super().__init__(briscola_card.val, briscola_card.suit)
        self.cards = [briscola_card]

    def __repr__(self):
        return type(self).__name__+f"({self.val},{self.suit})"


class Hand(SetOfCards):
    def __init__(self, deck: Deck):
        super().__init__()
        self.deck = deck
        self.cards = [
            self.deck.draw_random(),
            self.deck.draw_random(),
            self.deck.draw_random()]

    def __str__(self):
        return "stampo"+str(self.cards)

    def play_this_card(self, index):
        played_card = self[index]
        self[index] = Card(None, 0)
        return played_card

    def indices_card_in_hand(self) -> List[int]:
        return [i for i in range(3) if self.cards[i]]

    def draw_replacement(self, draw_briscola_last_round: BriscolaCard = False):
        for i, position in enumerate(self.cards):
            if not position:
                if not draw_briscola_last_round:
                    self.cards[i] = self.deck.draw_random()
                else:
                    self.cards[i] = draw_briscola_last_round
                return
        print("no pescato")


@dataclass
class Observation:
    briscola: BriscolaCard
    hand: Hand
    table: SetOfCards

    def predict_form(self):
        return [self.briscola.ia().reshape(1,4)]+self.table.ia()+self.hand.ia()
    # FIX ME with proper inheritance from the briscola

if __name__ == '__main__':
    t = SetOfCards([Card(2,2), Card(10,1), Card(15,3), Card(6,0)])
    t[1]
    t[1:2]

    print("done")
