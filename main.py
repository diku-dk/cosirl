import tomllib

from src.simulator import simulator_sofa
from src.nn import simple_agent

def setup_agent(config):
    return simple_agent.Agent(config)

def setup_simulator(config):
    return simulator_sofa.Simulator(config)

def train_episode(agent, env):
    # training for an episode goes here
    pass

def main():

    config_agent = tomllib.load(open("config/agent.toml", "rb"))
    config_simulator = tomllib.load(open("config/simulator.toml", "rb"))
    agent = setup_agent(config_agent)
    simulator = setup_simulator(config_simulator)
    env = simulator.env

    train_episode(agent, env)

if __name__ == "__main__":
    main()
