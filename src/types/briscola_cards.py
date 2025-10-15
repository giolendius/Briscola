import numpy as np
from typing import List, Iterator
from dataclasses import dataclass

from .types import Action


# from loguru import logger

_all_possible_val = [2, 4, 5, 6, 7, 8, 9, 10, 13, 15]
suit_dictionary = {2: "Bastoni", 1: "Coppe", 0: "Denari", 3: "Spade"}
suit_dic = {val: suit[0:2] for val, suit in suit_dictionary.items()}


class Card:
    card_dic = {x: f"{x}" for x in range(4, 8)} | {2: "2", 8: "Fante", 9: "Cav", 10: "Re", 13: "3", 15: "Asso"}
    points_dic = {8: 2, 9: 3, 10: 4, 13: 10, 15: 11}

    def __init__(self, val: int | None, suit: int):
        if val in _all_possible_val+[0]:
            self.val = val
            if suit in suit_dictionary:
                self.suit: int = suit
            else:
                raise Exception("Il seme della carta non è valido. Dichiarare seme con intero 0-3")
        elif not val:
            self.val = None
            self.suit = suit
        else:
            raise Exception(f"Il valore {val} della carta non è valido")


    def ia(self):
        """Output the card as tensor of shape (4,)"""
        a = np.array([0, 0, 0, 0])
        if self.val:
            a[self.suit] = self.val
        return a

    def card_to_dict(self, explicit: bool) -> dict:
        if explicit:
            d = {'val': self.val, 'suit': suit_dictionary[self.suit]}
        else:
            d = {"v":self.val, "s": self.val}
        return d

    def __str__(self):
        if not self:
            return " _ "
        elif self.val == "Card":
            print('sto cazzo')
            return "Card"
        return Card.card_dic[self.val] + " di " + suit_dictionary[self.suit]

    def __repr__(self) -> str:
        representation = f"Card({self.val},{suit_dic[self.suit]})" if self else "EmptyCard"
        return representation

    def __bool__(self):
        if self.val:
            return True
        else:
            return False

    def points(self):
        return Card.points_dic.get(self.val, 0)


class SetOfCards:
    def __init__(self, list_of_cards: List[Card] = None):
        self.cards: List[Card] = list_of_cards if list_of_cards else []

    def draw_random(self) -> Card | None:
        """Remove a random card from this set and returns it"""
        from random import randint
        if not self.cards:
            return None  # or raise an exception if you prefer
        index = randint(0, len(self.cards) - 1)
        return self.cards.pop(index)

    def __len__(self):
        return len(self.cards)

    def __iter__(self) -> Iterator[Card]:
        return iter(self.cards)

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
            return type(self)(list_of_cards = self.cards[index])
        else:
            raise Exception("Index must be int or slice")

    def __setitem__(self, key, card: Card):
        print(f'use setitem for {card}')
        if not isinstance(card, Card):
            raise Exception("Puoi assegnare solo oggetti 'carta'")
        self.cards[key] = card

    def __bool__(self):
        return bool(self.cards[0])

    def ia(self):
        """Returns a list of the cards.ia()"""
        return [card.ia().reshape(1,4) for card in self.cards]



class Table(SetOfCards):
    def __init__(self, list_of_cards: List[Card] = None, n_players: int = None):
        if not list_of_cards:
            list_of_cards = [Card(None, 0) for _ in range(n_players)]
        super().__init__(list_of_cards)

    def empty(self):
        for player_num in range(len(self.cards)):
            self[player_num] = Card(None, 0)


class Deck(SetOfCards):
    def __init__(self):
        super().__init__()
        self.cards = [Card(v, s) for s in range(4) for v in _all_possible_val]



class BriscolaCard(Card, SetOfCards):
    def __init__(self, deck):
        """Create an instance of BriscolaCard, which is both a Card and a SetOfCards with one card: itself"""
        briscola_card = deck.draw_random()
        super().__init__(briscola_card.val, briscola_card.suit) #call Card init
        self.cards = [self]

    def __repr__(self):
        return type(self).__name__+f"({self.val},{self.suit})"


class Hand(SetOfCards):
    def __init__(self, deck: Deck):
        super().__init__()
        self.name = 'Hand'
        self.deck = deck
        self.cards = [
            self.deck.draw_random(),
            self.deck.draw_random(),
            self.deck.draw_random()]

    def __str__(self, spaces=10):
        sp = " " * spaces + "|" + " " * spaces
        return f"{self[0]}" + sp + f"{self[1]}" + sp + f"{self[2]}"

    def display(self, visible: bool = True, spaces=5) -> str:
        sp = " " * spaces + "|" + " " * spaces
        if visible:
            show = f"{self[0]}" + sp + f"{self[1]}" + sp + f"{self[2]}"
        else:
            show = f"{Card(0, 0)}" + sp + f"{Card(0, 0)}" + sp + f"{Card(0, 0)}"
        return show

    def play_this_card(self, index: Action):
        """Returns the chosen card and removes it from the Hand"""
        played_card = self[index.value]
        self[index.value] = Card(None, 0)
        return played_card

    def indices_card_in_hand(self) -> List[int]:
        raise NotImplementedError
        return [i for i in range(3) if self.cards[i]]

    def draw_replacement(self, draw_briscola_last_round: BriscolaCard = False):
        for i, position in enumerate(self.cards):
            if not position:
                if not draw_briscola_last_round:
                    self[i] = self.deck.draw_random()
                else:
                    self[i] = draw_briscola_last_round
                return
        print("no pescato")


@dataclass
class Observation:
    briscola: BriscolaCard
    hand0: Card
    hand1: Card
    hand2: Card
    table0: Card = None
    table1: Card = None

    @classmethod
    def from_sets(cls, briscola: BriscolaCard, hand: Hand, table: Table):
        return cls(briscola=briscola,
                   hand0=hand[0],
                   hand1=hand[1],
                   hand2=hand[2],
                   table0=table[0] if len(table) > 0 else Card(None, 0),
                   table1=table[1] if len(table) > 1 else Card(None, 0))

    def indices_card_in_hand(self) -> List[int]:
        """returns list with 0,1,2 if index in hand"""
        return [i for i, card in enumerate([self.hand0, self.hand1, self.hand2]) if card]
    # def predict_form(self):
    #     return [self.briscola.ia().reshape(1,4)]+self.table.ia()+self.hand.ia()
    def to_dict(self, explicit: bool = True) -> dict:
        dict_card_suit_value = {f"{name}_{key}": val
                for name,carta in self.__dict__.items() if carta and isinstance(carta, Card)
                for key, val in carta.card_to_dict(explicit=explicit).items()}
        return dict_card_suit_value

@dataclass
class TurnMemory:
    turn: int
    action: int
    observation: Observation
    reward: int = None

    def to_dict(self, explicit: bool = True) -> dict:
        dict_int = {'turn' : self.turn,
                    'reward': self.reward,
                    'action': self.action}
        return dict_int | self.observation.to_dict(explicit)






if __name__ == '__main__':
    d = Deck()
    b = BriscolaCard(d)
    h = Hand(d)
    t = Table([Card(2,2)])
    o = Observation.from_sets(b, h, t)
    t = TurnMemory(1, o, reward=3, action=2)
    o.to_dict(True)
    t[1]
    t[1:2]

    print("done")
