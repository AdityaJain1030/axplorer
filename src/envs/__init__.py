from src.envs.cycle import SquareEnvironment
from src.envs.isosceles import IsoscelesEnvironment
from src.envs.sphere import SphereEnvironment

ENVS = {"square": SquareEnvironment, "isosceles": IsoscelesEnvironment, "sphere": SphereEnvironment}

from src.envs.k4free import K4FreeEnvironment
ENVS["k4free"] = K4FreeEnvironment

def build_env(params):
    """
    Build environment.
    """
    env = ENVS[params.env_name](params)
    return env
