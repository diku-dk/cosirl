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
    env.visualize_environment()
    #train_episode(agent, env)

if __name__ == "__main__":
    main()
