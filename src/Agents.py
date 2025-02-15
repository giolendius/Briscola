import numpy as np
from keras import Input, layers, models, optimizers
from abc import ABC, abstractmethod

from src.Card import SetOfCards

namelist = ["Pieruc", "Iuanin", "Barbacec", "Vecia"]


def boltzmann(q_val, gamma=12):
    num_actions = q_val.shape[0]
    a = np.exp(np.clip(q_val / gamma, -45., 45.))
    a = a / np.sum(a)
    return np.random.choice(range(num_actions), p=a)


class Agent(ABC):
    namecounter = 0

    def __init__(self, name=None):
        if name:
            self.name = name
        else:
            self.name = namelist[Agent.namecounter]
            Agent.namecounter += 1

    @abstractmethod
    def action(self, observation: SetOfCards):
        pass

    def __str__(self):
        return self.name

    def __repr__(self):
        return type(self).__name__ + ": "+self.name


class RandomAgent(Agent):
    """An agent who plays a random card of the available ones"""
    def action(self, observation):
        from random import choice
        poss = [i - 1 for i in range(1, 4) if observation[i].val]
        return choice(poss), 0


class AgentOnlyFirst(Agent):
    def action(self, observation):
        return min([i - 1 for i in range(1, 4) if observation[i].val]), 0


class IsBriscola(layers.Layer):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        # self.scale = tf.Variable(1.)

    def compute_output_shape(self, input_shape):
        return (input_shape[0], 1)

    def call(self, array, brisc):
        import tensorflow as tf
        # brisc=tf.constant(brisc, dtype=tf.float32)
        # assert array.shape==(4,)
        if array.ndim == 2 == brisc.ndim:
            batch_size = array.shape[0]
            c = tf.reduce_max(array * brisc, axis=1)
            c = tf.reshape(c, (-1, 1))
        elif array.ndim == 1 == brisc.ndim:
            c = tf.reduce_max(array * brisc, axis=0)
        else:
            raise "Shape non chiara sinceramente"
        w = tf.cast(tf.not_equal(c, 0), dtype=tf.float32)
        return w

    def __str__(self):
        return "Is Briscola L"


class CoolAgent(Agent):
    """Create a new DL agent. If model is not provided, a new model is created.
    Else, it is loaded from the specified path"""

    def __init__(self, name=None, model_path=None):
        super().__init__(name)
        self.build_model()
        if model_path:
            self.model.load_weights(model_path)
        self.model.compile(optimizer=optimizers.Adam(learning_rate=0.05),
                           loss="mse",
                           metrics=["mse"])

    def build_model(self):
        from keras import Input, layers, models, optimizers

        brisc = Input(shape=(4,), name="brisc")
        table = Input(shape=(4,), name="table")
        table_ren = table / 15
        tb = IsBriscola()(table_ren, brisc)

        core1 = layers.Dense(10, activation="relu", name="core1")
        core2 = layers.Dense(10, activation="relu", name="core2")
        core3 = layers.Dense(1)

        ini_hand = [0, 0, 0]
        outp = [0, 0, 0]
        for i in range(3):
            ini_hand[i] = Input(shape=(4,), name=f"hand{i}")
            played_ren = ini_hand[i] / 15
            pb = IsBriscola()(played_ren, brisc)
            z = layers.Concatenate(axis=1, name=f"ConcHand{i}")([table_ren, tb, played_ren, pb])
            z1 = core1(z)
            z2 = core2(z1)
            z3 = core3(z2)
            outp[i] = z3
        out = layers.concatenate(outp)
        # whole_hand=tf.constant(whole_hand)
        # out=layers.Concatenate()([whole_hand])
        modellobello = models.Model(inputs=[brisc, table] + ini_hand, outputs=out)
        self.model = modellobello

    def action(self, observation, policy="Boltzmann"):  # observation : list[type(Card(0,0))]

        brisc = observation[0].ia().reshape(1, 4)
        table = np.array([card.ia() for card in observation[4:8]]).reshape(1, 4)
        poss = [i - 1 for i in range(1, 4) if observation[i].val]
        # hand = np.concatenate([[observation[i + 1].ia() for i in range(3)]])
        hand = [observation[i + 1].ia().reshape(1, 4) for i in range(3)]

        q_val = self.model.predict([brisc, table] + hand, verbose=0)
        if np.isnan(q_val).any() or np.isinf(q_val).any():
            error = "Got a non number!!!!!"
            self.model.save_weights("prova.weights.h5")
        if policy == "Boltzmann":
            if q_val.ndim == 2:
                q_val = q_val.reshape(3)
            action = boltzmann(q_val)
            if action not in poss:
                new_q = [q_val[i] if i in poss else -55 for i in range(3)]
                q_val[action] = -30
                action = np.argmax(new_q)
        return action, q_val

class Agentepercapire:
    def __init__(self, name="Good Player", model_path=None):
        self.name = name
        # self.model = self.build_model()
        if model_path:
            self.model.load_weights(model_path)

    def build_model(self):
        brisc = Input(shape=(4,), name="brisc")
        table = Input(shape=(4,), name="table")
        z = layers.Concatenate()([brisc, table])
        z1 = layers.Dense(10, activation="relu")(z)
        modellobello = models.Model(inputs=[brisc, table], outputs=z1)
        modellobello.compile(optimizer=optimizers.Adam(),
                             loss="mse",
                             metrics=["mse"])
        return modellobello

    def action(self, observation, policy="Boltzmann"):  # observation : list[type(Card(0,0))]

        brisc = observation[0].ia().reshape(1, 4)
        table = observation[0].ia().reshape(1, 4)

        q_val = self.model.predict([brisc, table], verbose=0)
        return q_val


if __name__ == '__main__':
    from src.Card import Card
    r = RandomAgent()

    ag = Agentepercapire()
    modello = ag.build_model()
    observation2 = [Card(2, 0),
                    Card(None, 1)]
    brisc = observation2[0].ia().reshape(1, 4)
    table = observation2[0].ia().reshape(1, 4)

    modello.predict([brisc, table], verbose=0)
    ag.action(observation2)
