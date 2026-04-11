from src.envs.cycle import SquareEnvironment
from src.envs.isosceles import IsoscelesEnvironment
from src.envs.sphere import SphereEnvironment

ENVS = {"square": SquareEnvironment, "isosceles": IsoscelesEnvironment, "sphere": SphereEnvironment}

# from src.envs.k4free import K4FreeEnvironment
from src.envs.ramsey_4_t import K4FreeEnvironment
ENVS["ramsey_4_t"] = K4FreeEnvironment

from src.envs.kfour import KFourEnvironment
ENVS["kfour"] = KFourEnvironment

def build_env(params):
    """
    Build environment.
    """
    env = ENVS[params.env_name](params)
    return env
