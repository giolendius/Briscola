import numpy as np
import pandas as pd
from tqdm import tqdm

from src.Briscola import BriscolaEnv
from src.types.Agents import RandomAgent
from src.types.DLAgents import DLAgent


def simulate_games(train_episodes=2, save_name=None):
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
    dl_agent = DLAgent()

    from src.types.briscola_cards import Card
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

    vec1 = dl_agent.model.predict([brisc, table, hand0, hand1, hand2], verbose=0)

    # df = env.simulate_games(MA, 2)
    env.train_model(dl_agent, df, epochs=20, save_name="briscola_model.weights.h5")
    vec2 = dl_agent.model.predict([brisc, table, hand0, hand1, hand2], verbose=0)
    print(vec1, vec2)


# play(model_path="briscola_model.weights.h5")
# test_env(model_path="briscola_model.weights.h5")
# play()
# train_agents()

# train_agents("briscola.weights.h5")
# play(model_path="briscola.weights.h5")

if __name__ == '__main__':
    simulate_games()
