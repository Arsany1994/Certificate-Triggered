
from types import SimpleNamespace

import torch

from utilities.constants import AGENTS, SCENARIOS
from utilities.helper_scenario import Normalizers
from utilities.communication_cache import CommunicationCache
from utilities.communication_adapter import CommunicationAdapter
from scenarios.road_traffic import ScenarioRoadTraffic


PAYLOAD_DIM = 11
DISTANCE_INDEX = 10
DT = 0.05
RELATIVE_SPEED_BOUND = 2.0


def run_case(scenario_name, physical_distance_m, age_steps):
    scenario_config = SCENARIOS[scenario_name]

    lane_normalizer_m = scenario_config["lane_width"] * 3.0
    agent_normalizer_m = AGENTS["length"] * 10.0

    normalizers = Normalizers(
        distance_lanelet=torch.tensor(
            lane_normalizer_m,
            dtype=torch.float32,
        ),
        distance_agent=torch.tensor(
            agent_normalizer_m,
            dtype=torch.float32,
        ),
    )

    parameters = SimpleNamespace(
        is_observe_distance_to_agents=True,
        is_observe_vertices=True,
        certificate_relative_speed_bound_mps=RELATIVE_SPEED_BOUND,
        certificate_base_position_error_m=0.0,
        certificate_required_clearance_m=0.0,
        certificate_trigger_threshold_m=0.0,
    )

    scenario = SimpleNamespace(
        parameters=parameters,
        normalizers=normalizers,
        world=SimpleNamespace(dt=DT),
        agent_length=AGENTS["length"],
        agent_width=AGENTS["width"],
        last_certificate=None,
    )

    normalized_distance = physical_distance_m / lane_normalizer_m

    payload = torch.zeros(
        (1, 1, PAYLOAD_DIM),
        dtype=torch.float32,
    )
    payload[0, 0, DISTANCE_INDEX] = normalized_distance

    cache = CommunicationCache(
        batch_size=1,
        n_agents=2,
        payload_dim=PAYLOAD_DIM,
        device="cpu",
        dtype=torch.float32,
    )

    cache.update(
        env_indices=[0],
        ego_indices=[0],
        source_indices=[1],
        payload=payload[0, 0],
        step=0,
    )

    cached_payload, cached_valid, cached_age = cache.read(
        env_indices=[0],
        ego_indices=[0],
        source_indices=[1],
        current_step=age_steps,
    )

    assert torch.equal(
        cached_payload[0],
        payload[0, 0],
    ), "Cache changed the payload"

    assert bool(cached_valid[0].item())
    assert int(cached_age[0].item()) == age_steps

    adapter = CommunicationAdapter(
        batch_size=1,
        n_agents=2,
        mode="certificate",
    )

    adapter.prepare_cache(
        payload_dim=PAYLOAD_DIM,
        device=payload.device,
        dtype=payload.dtype,
    )

    adapter.cache.update(
        env_indices=[0],
        ego_indices=[0],
        source_indices=[1],
        payload=payload[0, 0],
        step=0,
    )

    source_indices = torch.tensor(
        [[1]],
        dtype=torch.long,
    )

    eligible_mask = torch.ones(
        (1, 1),
        dtype=torch.bool,
    )

    trigger_mask = ScenarioRoadTraffic._certificate_trigger_mask(
        scenario,
        adapter,
        payload,
        source_indices,
        eligible_mask,
        age_steps,
        0,
    )

    certificate = scenario.last_certificate

    return {
        "decoded_distance_m": float(
            certificate.center_distance_m.reshape(-1)[0].item()
        ),
        "slack_m": float(
            certificate.clearance_slack_m.reshape(-1)[0].item()
        ),
        "valid": bool(
            certificate.valid.reshape(-1)[0].item()
        ),
        "refresh": bool(
            certificate.request_refresh.reshape(-1)[0].item()
        ),
        "trigger": bool(
            trigger_mask.reshape(-1)[0].item()
        ),
    }


def main():
    scenarios = (
        "CPM_mixed",
        "intersection_2",
        "on_ramp_1",
        "roundabout_1",
    )

    distances_m = (0.2, 0.5, 1.0, 2.0, 5.0)

    for scenario_name in scenarios:
        lane_normalizer_m = SCENARIOS[scenario_name]["lane_width"] * 3.0
        agent_normalizer_m = AGENTS["length"] * 10.0
        expected_factor = agent_normalizer_m / lane_normalizer_m

        print(
            f"{scenario_name}: "
            f"lane_normalizer={lane_normalizer_m:.4f}, "
            f"agent_normalizer={agent_normalizer_m:.4f}, "
            f"historical_factor={expected_factor:.6f}"
        )

        for distance_m in distances_m:
            result = run_case(
                scenario_name,
                distance_m,
                age_steps=1,
            )

            assert abs(
                result["decoded_distance_m"] - distance_m
            ) < 1e-5, (
                f"{scenario_name}: decoded distance "
                f"{result['decoded_distance_m']} differs from "
                f"physical distance {distance_m}"
            )

            if distance_m == 0.2:
                assert result["trigger"] is True
                assert result["refresh"] is True

            if distance_m >= 0.5:
                assert result["trigger"] is False

            print(
                f"  d={distance_m:.3f} m: "
                f"decoded={result['decoded_distance_m']:.6f}, "
                f"slack={result['slack_m']:.6f}, "
                f"valid={int(result['valid'])}, "
                f"refresh={int(result['refresh'])}, "
                f"trigger={int(result['trigger'])}"
            )

    radius_m = 0.5 * (
        AGENTS["length"] ** 2 + AGENTS["width"] ** 2
    ) ** 0.5

    age_steps = 1
    uncertainty_m = RELATIVE_SPEED_BOUND * age_steps * DT
    unsafe_distance_m = (
        2.0 * radius_m + uncertainty_m - 0.01
    )

    unsafe = run_case(
        "intersection_2",
        unsafe_distance_m,
        age_steps,
    )

    assert unsafe["valid"] is False
    assert unsafe["refresh"] is True
    assert unsafe["trigger"] is True

    print(
        "unsafe boundary case: "
        f"d={unsafe_distance_m:.6f}, "
        f"decoded={unsafe['decoded_distance_m']:.6f}, "
        f"slack={unsafe['slack_m']:.6f}, "
        f"trigger={int(unsafe['trigger'])}"
    )

    print("PASS: corrected certificate normalization tests")


if __name__ == "__main__":
    main()
