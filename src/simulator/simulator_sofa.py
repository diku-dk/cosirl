from src.simulator import setup_sofa

class Simulator:
    def __init__(self, config):
        setup_sofa.setup_sofa_environment(config)
        
        import Sofa
        import SofaRuntime

        self.config = config

    @property
    def env(self):
        return None
