import numpy as np


class Card:
    suit_dic = {0: "Bastoni", 1: "Coppe", 2: "Ori", 3: "Spade"}
    card_dic = {x: f"{x}" for x in range(4, 8)} | {2: "2", 8: "Fante", 9: "Cav", 10: "Re", 13: "3", 15: "Asso"}
    points_dic = {8: 2, 9: 3, 10: 4, 13: 10, 15: 11}

    def __init__(self, val: int, suit: int):
        self.val = val
        self.suit: int = suit

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

    def points(self):
        return Card.points_dic.get(self.val, 0)
