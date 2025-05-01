import tomllib

from src.simulator import simulator_sofa
from src.nn import simple_agent

def setup_agent(config):
    return simple_agent.Agent(config)

def train_episode(agent, env):
    # training for an episode goes here
    pass

def main():

    config_agent = tomllib.load(open("config/agent.toml", "rb"))
    config_simulator = tomllib.load(open("config/simulator.toml", "rb"))
    agent = setup_agent(config_agent)
    env = simulator_sofa.SofaColonEndoscopeEnv(config_simulator)

    state = env.reset()
    print(state)
    action = agent.get_action(state)
    state = env.step(action)
    print(state)
    state = env.reset()
    state = env.step(action)
    #env.visualize_environment()
    #train_episode(agent, env)
    import numpy as np

    n_steps = 10

    obs, info = env.reset()
    for step in range(n_steps):
        print(step)
        action = np.random.uniform(-1, 1, size=env.action_space.shape)
        obs, reward, terminated, truncated, info = env.step(action)
        if terminated or truncated:
            obs, info = env.reset()
        if step == n_steps - 1:
            env.visualize_environment()
            
    env.close()

if __name__ == "__main__":
    main()
