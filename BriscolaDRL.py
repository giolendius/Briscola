from src.Briscola import Briscola
from src.Agents import CoolAgent, Agent_Random, AgentOnlyFirst
import numpy as np

env = Briscola(2)

MA = CoolAgent()
env.train(MA, 2000, save_name="briscola_model.weights.h5")
#
#
MA = CoolAgent(model_path="briscola_model.weights.h5")
env.play([Agent_Random(), MA], render_mode="pygame", delay1=1500, delay2=3500)

# check saved
MA = CoolAgent(model_path="briscola_model.weights.h5")
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

# h = env.history
# plt.plot(range(1,len(h)+1), h)
# plt.show()
