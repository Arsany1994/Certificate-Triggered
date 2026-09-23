import os

from utilities.mappo_cavs import mappo_cavs
from utilities.helper_training import Parameters

config_file = os.environ.get("SIGMARL_CONFIG", "config.json")  # Adjust parameters therein
parameters = Parameters.from_json(config_file)
mappo_cavs(parameters=parameters)
