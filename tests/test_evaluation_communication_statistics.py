import json
import tempfile
from pathlib import Path

from scenarios.road_traffic import ScenarioRoadTraffic
from torchrl.envs.libs.vmas import VmasEnv
from utilities.evaluation_base import Evaluation
from utilities.helper_training import Parameters


parameters = Parameters(
    scenario_type="CPM_mixed",
    n_agents=4,
    device="cpu",
    max_steps=8,
    is_partial_observation=True,
    n_nearing_agents_observed=2,
    is_add_noise=False,
    communication_mode="periodic",
    communication_period_steps=2,
)

scenario = ScenarioRoadTraffic()
scenario.parameters = parameters

env = VmasEnv(
    scenario=scenario,
    num_envs=2,
    continuous_actions=True,
    max_steps=parameters.max_steps,
    device=parameters.device,
    n_agents=parameters.n_agents,
)

env.reset()

evaluation = object.__new__(Evaluation)

with tempfile.TemporaryDirectory() as temporary_directory:
    rollout_path = str(Path(temporary_directory) / "rollout.pth")
    statistics_path = evaluation._save_communication_statistics(
        env=env,
        path_eval_out_td=rollout_path,
    )

    statistics_file = Path(statistics_path)
    assert statistics_file.exists()

    with statistics_file.open("r") as file:
        saved_statistics = json.load(file)

    assert saved_statistics["mode"] == "periodic"
    assert saved_statistics["payload_dim"] == 11
    assert saved_statistics["messages_sent"] > 0
    assert saved_statistics["bytes_sent"] > 0
    assert evaluation.communication_statistics_path == statistics_path

    print(f"STATISTICS FILE: {statistics_path}")
    print(f"SAVED STATISTICS: {saved_statistics}")

env.close()

print("PASS: evaluator communication sidecar saved and reloaded")
