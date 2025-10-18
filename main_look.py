from src.pygame.pybriscola import PyBriscolaEnv
from src.types.Agents import RandomAgent, Human # DLAgent


def play():
    a1 = Human('Gioele')
    a1 = RandomAgent("Gioele")
    a2 = RandomAgent("Jhon")
    env = PyBriscolaEnv(state=0)
    env.run_env([a1, a2])


if __name__ == '__main__':
    play()