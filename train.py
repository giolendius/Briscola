import numpy as np
import pandas as pd
from tqdm import tqdm

from src.Briscola import BriscolaEnv
from src.types.Agents import RandomAgent


from src.types.DLAgents import DLAgent


# def test_env(model_path):
#     # check saved
#     MA = DLAgent(model_path=model_path)
#     brisc = np.array([[0, 0, 8, 0]])
#     table = np.array([[0, 0, 0, 15]])
#     hand = [np.array([[10, 0, 0, 0]]), np.array([[0, 15, 0, 0]]), np.array([[0, 0, 2, 0]])]
#
#     q_val = MA.model.predict([brisc, table] + hand, verbose=0)
#     print("play briscola? 3", q_val)
#
#     table = np.array([[0, 2, 0, 0]])
#     q_val = MA.model.predict([brisc, table] + hand, verbose=0)
#     print("play ace? 2", q_val)
#
#     table = np.array([[4, 0, 0, 0]])
#     q_val = MA.model.predict([brisc, table] + hand, verbose=0)
#     print("play king? 1", q_val)

def simulate_games(train_episodes=10, save_name=None):
    env = BriscolaEnv(2)
    agents = [RandomAgent("Gioele"), RandomAgent("Luca")]
    full_df = pd.DataFrame()
    for i in tqdm(range(train_episodes)):
        env.run_env(agents)
        dict_game_memory = [turn_memory.to_dict() for turn_memory in env.game_memories]
        df = pd.DataFrame(dict_game_memory, dtype='object')
        df['game'] = i
        full_df = pd.concat([full_df, df], ignore_index=True)

    if save_name:
        full_df.to_csv(save_name)







def train_agents():
    env = BriscolaEnv(2)
    MA = DLAgent()

    from src.types.Card import Card
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

    # df = env.simulate_games(MA, 2)
    env.train_model(MA, df, epochs=20, save_name="briscola_model.weights.h5")
    vec2 = MA.model.predict([brisc, table, hand0, hand1, hand2], verbose=0)
    print(vec1,vec2)


# play(model_path="briscola_model.weights.h5")
# test_env(model_path="briscola_model.weights.h5")
# play()
# train_agents()

# train_agents("briscola.weights.h5")
# play(model_path="briscola.weights.h5")

if __name__ == '__main__':
    simulate_games()
