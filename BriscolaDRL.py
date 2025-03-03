from src.Briscola import Briscola
from src.Agents import CoolAgent, RandomAgent, AgentOnlyFirst, Agentepercapire
from keras import Input, layers, models, optimizers
import numpy as np

def test_env(model_path):
    # check saved
    MA = CoolAgent(model_path=model_path)
    brisc = np.array([[0, 0, 8, 0]])
    table = np.array([[0, 0, 0, 15]])
    hand = [np.array([[10, 0, 0, 0]]), np.array([[0, 15, 0, 0]]), np.array([[0, 0, 2, 0]])]

    q_val = MA.model.predict([brisc, table] + hand, verbose=0)
    print("play briscola? 3", q_val)

    table = np.array([[0, 2, 0, 0]])
    q_val = MA.model.predict([brisc, table] + hand, verbose=0)
    print("play ace? 2", q_val)

    table = np.array([[4, 0, 0, 0]])
    q_val = MA.model.predict([brisc, table] + hand, verbose=0)
    print("play king? 1", q_val)


def play(model_path=None):
    env = Briscola(2)
    if model_path:
        MA = CoolAgent(model_path)
    env.play([RandomAgent(), CoolAgent("Jhon")], render_mode="pygame", delay_play=500, delay_end_round=2000)


def test_engine():
    env = Briscola(2)
    MA = CoolAgent()
    env.simulate_games(MA, 2)


def train_agents():
    env = Briscola(2)
    MA = CoolAgent()

    from src.Card import Card
    observation2 = [Card(2, 0),
                    Card(4, 1),
                    Card(13, 3),
                    Card(15, 1),
                    Card(13, 2)]
    brisc = observation2[0].ia().reshape(1, 4)
    table = observation2[1].ia().reshape(1, 4)
    hand0 = observation2[2].ia().reshape(1, 4)
    hand1 = observation2[3].ia().reshape(1, 4)
    hand2 = observation2[4].ia().reshape(1, 4)

    vec1 = MA.model.predict([brisc, table,hand0, hand1, hand2], verbose=0)

    df = env.simulate_games(MA, 20)
    env.train_model(MA, df, epochs=20, save_name="briscola_model.weights.h5")
    vec2 = MA.model.predict([brisc, table, hand0, hand1, hand2], verbose=0)
    print(vec1,vec2)


# play(model_path="briscola_model.weights.h5")
# test_env(model_path="briscola_model.weights.h5")
# play()
train_agents()
