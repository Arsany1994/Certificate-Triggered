import torch

from utilities.safety_certificate import (
    evaluate_safety_certificate,
    rectangle_circumradius,
)


def main() -> None:
    radius = rectangle_circumradius(
        length_m=0.16,
        width_m=0.08,
    )

    assert torch.isclose(
        radius,
        torch.tensor(0.5 * (0.16**2 + 0.08**2) ** 0.5),
    )

    safe = evaluate_safety_certificate(
        ego_position_m=torch.tensor([0.0, 0.0]),
        peer_position_m=torch.tensor([1.0, 0.0]),
        age_steps=0,
        dt_s=0.05,
        ego_radius_m=radius,
        peer_radius_m=radius,
        relative_speed_bound_mps=0.0,
        cache_valid=True,
    )
    assert bool(safe.valid.item())
    assert not bool(safe.request_refresh.item())
    print(
        "CERTIFICATE safe: "
        f"slack={safe.clearance_slack_m.item():.6f} m"
    )

    near_threshold = evaluate_safety_certificate(
        ego_position_m=torch.tensor([0.0, 0.0]),
        peer_position_m=torch.tensor([0.2, 0.0]),
        age_steps=0,
        dt_s=0.05,
        ego_radius_m=radius,
        peer_radius_m=radius,
        relative_speed_bound_mps=0.0,
        trigger_threshold_m=0.03,
        cache_valid=True,
    )
    assert bool(near_threshold.valid.item())
    assert bool(near_threshold.request_refresh.item())
    print(
        "CERTIFICATE near threshold: "
        f"slack={near_threshold.clearance_slack_m.item():.6f} m, "
        f"refresh={bool(near_threshold.request_refresh.item())}"
    )

    unsafe = evaluate_safety_certificate(
        ego_position_m=torch.tensor([0.0, 0.0]),
        peer_position_m=torch.tensor([0.1, 0.0]),
        age_steps=0,
        dt_s=0.05,
        ego_radius_m=radius,
        peer_radius_m=radius,
        relative_speed_bound_mps=0.0,
        cache_valid=True,
    )
    assert not bool(unsafe.valid.item())
    assert bool(unsafe.request_refresh.item())
    print(
        "CERTIFICATE unsafe: "
        f"slack={unsafe.clearance_slack_m.item():.6f} m"
    )

    fresh = evaluate_safety_certificate(
        ego_position_m=torch.tensor([0.0, 0.0]),
        peer_position_m=torch.tensor([0.4, 0.0]),
        age_steps=0,
        dt_s=0.1,
        ego_radius_m=radius,
        peer_radius_m=radius,
        relative_speed_bound_mps=1.0,
        cache_valid=True,
    )

    stale = evaluate_safety_certificate(
        ego_position_m=torch.tensor([0.0, 0.0]),
        peer_position_m=torch.tensor([0.4, 0.0]),
        age_steps=3,
        dt_s=0.1,
        ego_radius_m=radius,
        peer_radius_m=radius,
        relative_speed_bound_mps=1.0,
        cache_valid=True,
    )

    assert stale.clearance_slack_m.item() < fresh.clearance_slack_m.item()
    assert stale.uncertainty_radius_m.item() > fresh.uncertainty_radius_m.item()
    print(
        "CERTIFICATE aging: "
        f"fresh_slack={fresh.clearance_slack_m.item():.6f} m, "
        f"stale_slack={stale.clearance_slack_m.item():.6f} m, "
        f"stale_uncertainty={stale.uncertainty_radius_m.item():.6f} m"
    )

    invalid_cache = evaluate_safety_certificate(
        ego_position_m=torch.tensor([0.0, 0.0]),
        peer_position_m=torch.tensor([1.0, 0.0]),
        age_steps=0,
        dt_s=0.05,
        ego_radius_m=radius,
        peer_radius_m=radius,
        relative_speed_bound_mps=0.0,
        cache_valid=False,
    )
    assert not bool(invalid_cache.valid.item())
    assert bool(invalid_cache.request_refresh.item())
    print("CERTIFICATE invalid cache: refresh=True")

    vectorized = evaluate_safety_certificate(
        ego_position_m=torch.tensor(
            [[0.0, 0.0], [0.0, 0.0]]
        ),
        peer_position_m=torch.tensor(
            [[1.0, 0.0], [0.1, 0.0]]
        ),
        age_steps=torch.tensor([0, 0]),
        dt_s=0.05,
        ego_radius_m=radius,
        peer_radius_m=radius,
        relative_speed_bound_mps=0.0,
        cache_valid=torch.tensor([True, True]),
    )
    assert vectorized.valid.tolist() == [True, False]
    print(
        "CERTIFICATE vectorized: "
        f"valid={vectorized.valid.tolist()}, "
        f"refresh={vectorized.request_refresh.tolist()}"
    )

    print("PASS: isolated safety certificate tests")


if __name__ == "__main__":
    main()
