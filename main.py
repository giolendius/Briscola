import os

import click
import pandas as pd
from tqdm import tqdm

from src.Briscola import BriscolaEnv
from src.pygame.pybriscola import PyBriscolaEnv
from src.types.Agents import RandomAgent, Human  # DLAgent
from src.utils.utils import DATA_FOLDER, dataset_file_name, setup_logger


@click.command()
@click.option('--mode', )
@click.option('--simulate', is_flag=True)
def main(mode: str, simulate: bool = False):
    logger = setup_logger('main')
    if mode == 'train':
        if simulate:
            simulate_games(20)
    elif mode =='play':
        a1 = Human('Gioele')
        a2 = RandomAgent("Jhon")

        env = PyBriscolaEnv(state=0)  # [a1,a2])
        env.run_env([a1, a2])
    elif mode == 'look':
        a1 = RandomAgent("Gioele")
        a2 = RandomAgent("Jhon")

        env = PyBriscolaEnv(state=0)  # [a1,a2])
        env.run_env([a1, a2])
    else:
        raise ValueError('mode not recognised')



def simulate_games(train_episodes=2):
    logger = setup_logger('simulation')
    logger.info('Start simulation...')
    env = BriscolaEnv()
    agents = [RandomAgent("Gioele"), RandomAgent("Luca")]
    full_df = pd.DataFrame()
    for i in tqdm(range(train_episodes)):
        env.run_env(agents)
        dict_game_memory = [turn_memory.to_dict(False) for turn_memory in env.game_memories]
        df = pd.DataFrame(dict_game_memory, dtype='object')
        df['game'] = i
        full_df = pd.concat([full_df, df], ignore_index=True)

    logger.info('Saving df....')
    os.makedirs(DATA_FOLDER, exist_ok=True)
    full_df.to_csv(DATA_FOLDER / dataset_file_name)


if __name__ == '__main__':
    main()


