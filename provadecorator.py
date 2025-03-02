import typing


def panna(func, name):
    def wrap(name, args):
        print("Per favore", name)
        func(*args)
        print("Con panna")
    return wrap(name)

@panna("Gioele")
def gelato(gusto):
    print(f"Voglio un gelato al {gusto}")

class ciao:
    prova: typing.Dict[str,int]
gelato("pistacchio")