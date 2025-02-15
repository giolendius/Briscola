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
    env.play([RandomAgent(),CoolAgent()], render_mode="pygame", delay1=1500, delay2=3500)


def test_engine():
    env=Briscola(2)
    env.simulate_games_and_train(RandomAgent("Giuanin"), 2)


def train_agents():
    # logger.info("Starting training")
    # from src.Card import Card
    # ag = agentepercapire()
    # modello = ag.build_model()
    # observation2 = [Card(2, 0),
    #                 Card(None, 1),
    #                 Card(None, 1),
    #                 Card(15, 1),
    #                 Card(12, 2)]
    # brisc = observation2[0].ia().reshape(1, 4)
    # table = observation2[0].ia().reshape(1, 4)
    # ag=CoolAgent()
    # modello.predict([brisc, table], verbose=0)
    # ag.action(observation2)
    env = Briscola(2)
    MA = CoolAgent()
    env.simulate_games_and_train(MA, 2, save_name="briscola_model.weights.h5")


# play(model_path="briscola_model.weights.h5")
# test_env(model_path="briscola_model.weights.h5")
play()
# h = env.history
# plt.plot(range(1,len(h)+1), h)
# plt.show()
