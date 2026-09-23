from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import torch

from utilities.communication_cache import CommunicationCache


@dataclass
class CommunicationResult:
    payload: torch.Tensor
    send_mask: torch.Tensor
    valid_mask: torch.Tensor
    age: torch.Tensor
    messages_sent: int
    messages_received: int
    bytes_sent: int
    bytes_received: int


class CommunicationAdapter:
    """Common identity-based adapter for all communication strategies."""

    def __init__(
        self,
        batch_size: int,
        n_agents: int,
        mode: str = "perfect",
        period_steps: Optional[int] = None,
        max_age_steps: Optional[int] = None,
    ) -> None:
        self.batch_size = int(batch_size)
        self.n_agents = int(n_agents)
        self.mode = str(mode).lower()
        self.period_steps = period_steps
        self.max_age_steps = max_age_steps
        self.cache = None
        self.payload_dim = None

        if self.mode not in {"perfect", "periodic", "age", "certificate"}:
            raise ValueError(
                "mode must be one of: perfect, periodic, age, certificate"
            )

        if self.mode == "periodic":
            if period_steps is None or int(period_steps) <= 0:
                raise ValueError(
                    "period_steps must be positive for periodic communication"
                )

        if self.mode == "age":
            if max_age_steps is None or int(max_age_steps) < 0:
                raise ValueError(
                    "max_age_steps must be non-negative for age communication"
                )

    def _normalise_steps(
        self,
        current_step,
        device: torch.device,
    ) -> torch.Tensor:
        steps = torch.as_tensor(
            current_step,
            device=device,
            dtype=torch.long,
        ).reshape(-1)

        if steps.numel() == 1:
            steps = steps.expand(self.batch_size)
        elif steps.numel() != self.batch_size:
            raise ValueError(
                "current_step must be scalar or contain one value per environment"
            )

        if bool(torch.any(steps < 0).item()):
            raise ValueError("current_step values must be non-negative")

        return steps

    def _ensure_cache(
        self,
        payload_dim: int,
        device: torch.device,
        dtype: torch.dtype,
    ) -> None:
        if self.cache is None:
            self.cache = CommunicationCache(
                batch_size=self.batch_size,
                n_agents=self.n_agents,
                payload_dim=payload_dim,
                device=device,
                dtype=dtype,
            )
            self.payload_dim = int(payload_dim)
            return

        if self.payload_dim != int(payload_dim):
            raise ValueError(
                "Communication payload dimension changed after cache creation"
            )

    def reset(self, env_mask=None) -> None:
        if self.cache is not None:
            self.cache.reset(env_mask)

    def apply(
        self,
        current_payload: torch.Tensor,
        source_indices: torch.Tensor,
        current_step,
        fallback_payload: torch.Tensor,
        eligible_mask: Optional[torch.Tensor] = None,
        trigger_mask: Optional[torch.Tensor] = None,
    ) -> CommunicationResult:
        """Apply one common communication decision.

        Shapes:
            current_payload: [batch, ego, neighbor_slot, payload_dim]
            source_indices:  [batch, ego, neighbor_slot]
            fallback_payload: same shape as current_payload
            eligible_mask:   [batch, ego, neighbor_slot]
            trigger_mask:    required only for certificate mode
        """
        if current_payload.ndim != 4:
            raise ValueError("current_payload must have four dimensions")

        batch, n_agents, n_slots, payload_dim = current_payload.shape

        if batch != self.batch_size or n_agents != self.n_agents:
            raise ValueError("current_payload batch or agent dimension mismatch")

        if tuple(source_indices.shape) != (batch, n_agents, n_slots):
            raise ValueError("source_indices shape mismatch")

        if tuple(fallback_payload.shape) != tuple(current_payload.shape):
            raise ValueError("fallback_payload shape mismatch")

        device = current_payload.device
        source_indices = source_indices.to(device=device, dtype=torch.long)
        steps = self._normalise_steps(current_step, device)

        if eligible_mask is None:
            eligible_mask = torch.ones(
                batch,
                n_agents,
                n_slots,
                device=device,
                dtype=torch.bool,
            )
        else:
            eligible_mask = eligible_mask.to(device=device, dtype=torch.bool)

        if tuple(eligible_mask.shape) != (batch, n_agents, n_slots):
            raise ValueError("eligible_mask shape mismatch")

        if self.mode == "perfect":
            send_mask = eligible_mask.clone()
            age = torch.zeros_like(source_indices)
            messages = int(send_mask.sum().item())
            bytes_sent = messages * payload_dim * current_payload.element_size()

            return CommunicationResult(
                payload=current_payload,
                send_mask=send_mask,
                valid_mask=eligible_mask.clone(),
                age=age,
                messages_sent=messages,
                messages_received=messages,
                bytes_sent=bytes_sent,
                bytes_received=bytes_sent,
            )

        self._ensure_cache(
            payload_dim=payload_dim,
            device=device,
            dtype=current_payload.dtype,
        )

        env_grid = torch.arange(
            batch, device=device, dtype=torch.long
        ).reshape(batch, 1, 1).expand(batch, n_agents, n_slots)

        ego_grid = torch.arange(
            n_agents, device=device, dtype=torch.long
        ).reshape(1, n_agents, 1).expand(batch, n_agents, n_slots)

        step_grid = steps.reshape(batch, 1, 1).expand(
            batch, n_agents, n_slots
        )

        cached_payload = self.cache.messages[
            env_grid, ego_grid, source_indices
        ]
        cached_valid = self.cache.valid[
            env_grid, ego_grid, source_indices
        ]
        cached_age = self.cache.age(steps)[
            env_grid, ego_grid, source_indices
        ]

        if self.mode == "periodic":
            send_mask = (
                (steps % int(self.period_steps))
                .reshape(batch, 1, 1)
                .expand(batch, n_agents, n_slots)
                == 0
            )
        elif self.mode == "age":
            send_mask = (
                (~cached_valid)
                | (cached_age >= int(self.max_age_steps))
            )
        else:
            if trigger_mask is None:
                raise ValueError(
                    "trigger_mask is required for certificate communication"
                )
            send_mask = trigger_mask.to(device=device, dtype=torch.bool)

        if tuple(send_mask.shape) != (batch, n_agents, n_slots):
            raise ValueError("trigger_mask shape mismatch")

        send_mask = send_mask & eligible_mask
        send_mask = send_mask | ((~cached_valid) & eligible_mask)

        flat_send = send_mask.reshape(-1)

        if bool(flat_send.any().item()):
            self.cache.update(
                env_indices=env_grid.reshape(-1)[flat_send],
                ego_indices=ego_grid.reshape(-1)[flat_send],
                source_indices=source_indices.reshape(-1)[flat_send],
                payload=current_payload.detach()
                .reshape(-1, payload_dim)[flat_send],
                step=step_grid.reshape(-1)[flat_send],
            )

        cached_payload = self.cache.messages[
            env_grid, ego_grid, source_indices
        ]
        cached_valid = self.cache.valid[
            env_grid, ego_grid, source_indices
        ]
        cached_age = self.cache.age(steps)[
            env_grid, ego_grid, source_indices
        ]

        usable = cached_valid & eligible_mask
        output_payload = torch.where(
            usable.unsqueeze(-1),
            cached_payload,
            fallback_payload,
        )

        messages = int(send_mask.sum().item())
        bytes_sent = messages * payload_dim * current_payload.element_size()

        return CommunicationResult(
            payload=output_payload,
            send_mask=send_mask,
            valid_mask=usable,
            age=cached_age,
            messages_sent=messages,
            messages_received=messages,
            bytes_sent=bytes_sent,
            bytes_received=bytes_sent,
        )
