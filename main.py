from src.Briscola import BriscolaGame
from src.Agents import Agent, RandomAgent # CoolAgent
from src.pygame.pybriscola import PyBriscola


def play(agent1: Agent, agent2: Agent):
    env = BriscolaGame(2)
    env.play([agent1, agent2], render_mode="pygame", delay_play=500, delay_end_round=2000)


def play2(agents):
    env = PyBriscola(2)
    env.play(agents)


if __name__ == '__main__':
    # CoolAgent(model_path=None)
    a1 = RandomAgent()
    a2 = RandomAgent("Jhon")
    # play(a1, a2)
    play2([a1,a2])