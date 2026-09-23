from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Union

import torch


TensorLike = Union[torch.Tensor, float, int]


@dataclass
class SafetyCertificateResult:
    """Result of a conservative stale-information safety certificate."""

    clearance_slack_m: torch.Tensor
    valid: torch.Tensor
    request_refresh: torch.Tensor
    uncertainty_radius_m: torch.Tensor
    center_distance_m: torch.Tensor
    age_seconds: torch.Tensor


def rectangle_circumradius(
    length_m: TensorLike,
    width_m: TensorLike,
) -> torch.Tensor:
    """Return the circumradius of a rectangular vehicle.

    The circumradius is half of the rectangle diagonal. Using it gives a
    conservative disk enclosure of the vehicle footprint.
    """
    length = torch.as_tensor(length_m, dtype=torch.float32)
    width = torch.as_tensor(width_m, dtype=torch.float32)

    if bool(torch.any(length < 0).item()):
        raise ValueError("length_m must be non-negative")
    if bool(torch.any(width < 0).item()):
        raise ValueError("width_m must be non-negative")

    return 0.5 * torch.sqrt(length.square() + width.square())


def _broadcast_parameter(
    value: TensorLike,
    shape: torch.Size,
    *,
    reference: torch.Tensor,
    name: str,
) -> torch.Tensor:
    """Convert and broadcast a scalar or tensor parameter."""
    tensor = torch.as_tensor(
        value,
        device=reference.device,
        dtype=reference.dtype,
    )

    try:
        return torch.broadcast_to(tensor, shape)
    except RuntimeError as exc:
        raise ValueError(
            f"{name} with shape {tuple(tensor.shape)} cannot be broadcast "
            f"to certificate shape {tuple(shape)}"
        ) from exc


def evaluate_safety_certificate(
    ego_position_m: torch.Tensor,
    peer_position_m: torch.Tensor,
    age_steps: TensorLike,
    *,
    dt_s: float,
    ego_radius_m: TensorLike,
    peer_radius_m: TensorLike,
    relative_speed_bound_mps: TensorLike,
    base_position_error_m: TensorLike = 0.0,
    required_clearance_m: TensorLike = 0.0,
    trigger_threshold_m: TensorLike = 0.0,
    cache_valid: Optional[TensorLike] = None,
) -> SafetyCertificateResult:
    """Evaluate a conservative pairwise certificate.

    Parameters
    ----------
    ego_position_m:
        Ego position in metres with final dimension of size 2.
    peer_position_m:
        Cached peer position in metres with final dimension of size 2.
    age_steps:
        Number of decision steps since the cached peer packet was received.
    dt_s:
        Simulator decision interval in seconds.
    ego_radius_m, peer_radius_m:
        Circumradii of the ego and peer rectangular footprints.
    relative_speed_bound_mps:
        Conservative upper bound on future relative speed.
    base_position_error_m:
        Initial position uncertainty in metres.
    required_clearance_m:
        Minimum certificate slack required to declare the packet safe.
    trigger_threshold_m:
        Refresh is requested when slack is at or below this value.
    cache_valid:
        Boolean validity mask for the cached packet.

    Notes
    -----
    The uncertainty set is a disk around the cached peer position with radius

        base_position_error_m
        + relative_speed_bound_mps * age_steps * dt_s.

    This speed-bounded formulation is intentional because SigmaRL commands
    speed directly and does not expose a validated acceleration bound.
    """
    ego = torch.as_tensor(ego_position_m, dtype=torch.float32)
    peer = torch.as_tensor(
        peer_position_m,
        device=ego.device,
        dtype=ego.dtype,
    )

    if ego.ndim == 0 or ego.shape[-1] != 2:
        raise ValueError("ego_position_m must have final dimension of size 2")
    if peer.ndim == 0 or peer.shape[-1] != 2:
        raise ValueError("peer_position_m must have final dimension of size 2")

    try:
        ego, peer = torch.broadcast_tensors(ego, peer)
    except RuntimeError as exc:
        raise ValueError(
            "ego_position_m and peer_position_m have incompatible shapes"
        ) from exc

    certificate_shape = ego.shape[:-1]

    if dt_s <= 0:
        raise ValueError("dt_s must be positive")

    age = torch.as_tensor(
        age_steps,
        device=ego.device,
        dtype=torch.long,
    )
    age = _broadcast_parameter(
        age,
        certificate_shape,
        reference=ego,
        name="age_steps",
    ).to(dtype=torch.long)

    ego_radius = _broadcast_parameter(
        ego_radius_m,
        certificate_shape,
        reference=ego,
        name="ego_radius_m",
    )
    peer_radius = _broadcast_parameter(
        peer_radius_m,
        certificate_shape,
        reference=ego,
        name="peer_radius_m",
    )
    relative_speed = _broadcast_parameter(
        relative_speed_bound_mps,
        certificate_shape,
        reference=ego,
        name="relative_speed_bound_mps",
    )
    base_error = _broadcast_parameter(
        base_position_error_m,
        certificate_shape,
        reference=ego,
        name="base_position_error_m",
    )
    required_clearance = _broadcast_parameter(
        required_clearance_m,
        certificate_shape,
        reference=ego,
        name="required_clearance_m",
    )
    trigger_threshold = _broadcast_parameter(
        trigger_threshold_m,
        certificate_shape,
        reference=ego,
        name="trigger_threshold_m",
    )

    for name, value in (
        ("ego_radius_m", ego_radius),
        ("peer_radius_m", peer_radius),
        ("relative_speed_bound_mps", relative_speed),
        ("base_position_error_m", base_error),
        ("required_clearance_m", required_clearance),
        ("trigger_threshold_m", trigger_threshold),
    ):
        if bool(torch.any(value < 0).item()):
            raise ValueError(f"{name} must be non-negative")

    age_seconds = age.to(dtype=ego.dtype) * float(dt_s)
    nonnegative_age = age >= 0
    effective_age_seconds = torch.clamp(age_seconds, min=0.0)

    uncertainty_radius = (
        base_error + relative_speed * effective_age_seconds
    )

    center_distance = torch.linalg.vector_norm(
        ego - peer,
        dim=-1,
    )

    clearance_slack = (
        center_distance
        - ego_radius
        - peer_radius
        - uncertainty_radius
    )

    finite_inputs = (
        torch.isfinite(ego).all(dim=-1)
        & torch.isfinite(peer).all(dim=-1)
        & torch.isfinite(uncertainty_radius)
        & torch.isfinite(clearance_slack)
    )

    if cache_valid is None:
        cache_is_valid = torch.ones(
            certificate_shape,
            device=ego.device,
            dtype=torch.bool,
        )
    else:
        cache_is_valid = _broadcast_parameter(
            cache_valid,
            certificate_shape,
            reference=ego,
            name="cache_valid",
        ).to(dtype=torch.bool)

    valid = (
        finite_inputs
        & nonnegative_age
        & cache_is_valid
        & (clearance_slack >= required_clearance)
    )

    request_refresh = (
        (~finite_inputs)
        | (~nonnegative_age)
        | (~cache_is_valid)
        | (clearance_slack <= trigger_threshold)
    )

    return SafetyCertificateResult(
        clearance_slack_m=clearance_slack,
        valid=valid,
        request_refresh=request_refresh,
        uncertainty_radius_m=uncertainty_radius,
        center_distance_m=center_distance,
        age_seconds=effective_age_seconds,
    )
