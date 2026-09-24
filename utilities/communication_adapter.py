"""Communication adapter with deterministic delay/loss/outage impairments.

This file is a drop-in replacement/reference for
``utilities/communication_adapter.py``.  The normal configuration is
backwards compatible:

    delay_steps=0
    loss_probability=0.0
    outage_windows=()

With those defaults, the existing perfect/periodic/age/certificate behavior
is unchanged.  ``perfect`` deliberately remains an ideal oracle and rejects
non-default impairment settings; use periodic, age, or certificate when
measuring network robustness.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, Optional, Sequence

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
    delivered_mask: Optional[torch.Tensor] = None
    loss_mask: Optional[torch.Tensor] = None
    outage_mask: Optional[torch.Tensor] = None


class CommunicationAdapter:
    """Identity-based communication with optional channel impairments.

    ``messages_sent`` counts transmission attempts selected by the schedule.
    ``messages_received`` counts packets that arrive at this call (including
    delayed packets whose delay has elapsed).  Loss and outage packets are
    therefore visible as sent-but-not-received traffic.  Information age is
    measured from the original send step, so a packet arriving after a delay
    still carries the correct staleness.
    """

    def __init__(
        self,
        batch_size: int,
        n_agents: int,
        mode: str = "perfect",
        period_steps: Optional[int] = None,
        max_age_steps: Optional[int] = None,
        delay_steps: int = 0,
        loss_probability: float = 0.0,
        outage_windows: Optional[Sequence[Sequence[int]]] = None,
        random_seed: int = 0,
    ) -> None:
        self.batch_size = int(batch_size)
        self.n_agents = int(n_agents)
        self.mode = str(mode).lower()
        self.period_steps = period_steps
        self.max_age_steps = max_age_steps
        self.delay_steps = int(delay_steps)
        self.loss_probability = float(loss_probability)
        self.outage_windows = self._normalise_outage_windows(outage_windows)
        self.random_seed = int(random_seed)

        self.cache: Optional[CommunicationCache] = None
        self.payload_dim: Optional[int] = None
        self.payload_element_size: Optional[int] = None

        # Pending deliveries are intentionally kept separate from the cache:
        # a sent packet is not usable until its arrival step.
        self._pending = []
        self._rng: Optional[torch.Generator] = None
        self._rng_device: Optional[torch.device] = None

        self.total_apply_calls = 0
        self.total_slots = 0
        self.total_eligible_slots = 0
        self.total_messages_sent = 0
        self.total_messages_received = 0
        self.total_bytes_sent = 0
        self.total_bytes_received = 0
        self.total_valid_slots = 0
        self.total_fallback_events = 0
        self.total_age_sum = 0
        self.total_age_samples = 0
        self.max_age_seen = -1
        self.total_certificate_evaluations = 0
        self.total_certificate_refresh_requests = 0

        self.total_dropped_messages = 0
        self.total_delayed_messages = 0
        self.total_outage_blocked_messages = 0

        if self.mode not in {"perfect", "periodic", "age", "certificate"}:
            raise ValueError(
                "mode must be one of: perfect, periodic, age, certificate"
            )
        if self.mode == "periodic" and (
            period_steps is None or int(period_steps) <= 0
        ):
            raise ValueError(
                "period_steps must be positive for periodic communication"
            )
        if self.mode == "age" and (
            max_age_steps is None or int(max_age_steps) < 0
        ):
            raise ValueError(
                "max_age_steps must be non-negative for age communication"
            )
        if self.delay_steps < 0:
            raise ValueError("delay_steps must be non-negative")
        if not 0.0 <= self.loss_probability <= 1.0:
            raise ValueError("loss_probability must be in [0, 1]")
        if self.mode == "perfect" and (
            self.delay_steps != 0
            or self.loss_probability != 0.0
            or self.outage_windows
        ):
            raise ValueError(
                "perfect mode is an ideal oracle; use periodic, age, or "
                "certificate for impaired-channel experiments"
            )

    @staticmethod
    def _normalise_outage_windows(
        windows: Optional[Sequence[Sequence[int]]],
    ) -> tuple[tuple[int, int], ...]:
        if windows is None:
            return ()
        normalised = []
        for window in windows:
            if len(window) != 2:
                raise ValueError(
                    "each outage window must be [start_step, end_step]"
                )
            start, end = (int(window[0]), int(window[1]))
            if start < 0 or end <= start:
                raise ValueError(
                    "outage windows require 0 <= start_step < end_step"
                )
            normalised.append((start, end))
        return tuple(normalised)

    def reset_statistics(self) -> None:
        """Reset counters while preserving cache and pending state."""
        self.total_apply_calls = 0
        self.total_slots = 0
        self.total_eligible_slots = 0
        self.total_messages_sent = 0
        self.total_messages_received = 0
        self.total_bytes_sent = 0
        self.total_bytes_received = 0
        self.total_valid_slots = 0
        self.total_fallback_events = 0
        self.total_age_sum = 0
        self.total_age_samples = 0
        self.max_age_seen = -1
        self.total_certificate_evaluations = 0
        self.total_certificate_refresh_requests = 0
        self.total_dropped_messages = 0
        self.total_delayed_messages = 0
        self.total_outage_blocked_messages = 0

    def _record_statistics(
        self,
        *,
        valid_mask: torch.Tensor,
        age: torch.Tensor,
        eligible_mask: torch.Tensor,
        messages_sent: int,
        messages_received: int,
        bytes_sent: int,
        bytes_received: int,
    ) -> None:
        self.total_apply_calls += 1
        self.total_slots += int(eligible_mask.numel())
        self.total_eligible_slots += int(eligible_mask.sum().item())
        self.total_messages_sent += int(messages_sent)
        self.total_messages_received += int(messages_received)
        self.total_bytes_sent += int(bytes_sent)
        self.total_bytes_received += int(bytes_received)

        valid_eligible = valid_mask & eligible_mask
        self.total_valid_slots += int(valid_eligible.sum().item())
        self.total_fallback_events += int(
            (eligible_mask & (~valid_eligible)).sum().item()
        )

        valid_ages = age[valid_eligible]
        if valid_ages.numel() > 0:
            self.total_age_sum += int(valid_ages.sum().item())
            self.total_age_samples += int(valid_ages.numel())
            self.max_age_seen = max(
                self.max_age_seen,
                int(valid_ages.max().item()),
            )

    def statistics(self) -> dict:
        mean_age = (
            self.total_age_sum / self.total_age_samples
            if self.total_age_samples > 0
            else None
        )
        return {
            "mode": self.mode,
            "payload_dim": self.payload_dim,
            "payload_element_size_bytes": self.payload_element_size,
            "delay_steps": self.delay_steps,
            "loss_probability": self.loss_probability,
            "outage_windows": [list(w) for w in self.outage_windows],
            "random_seed": self.random_seed,
            "apply_calls": self.total_apply_calls,
            "slots": self.total_slots,
            "eligible_slots": self.total_eligible_slots,
            "messages_sent": self.total_messages_sent,
            "messages_received": self.total_messages_received,
            "bytes_sent": self.total_bytes_sent,
            "bytes_received": self.total_bytes_received,
            "valid_slots": self.total_valid_slots,
            "fallback_events": self.total_fallback_events,
            "mean_valid_information_age": mean_age,
            "max_valid_information_age": self.max_age_seen,
            "dropped_messages": self.total_dropped_messages,
            "delayed_messages": self.total_delayed_messages,
            "outage_blocked_messages": self.total_outage_blocked_messages,
            "certificate_evaluations": self.total_certificate_evaluations,
            "certificate_refresh_requests": (
                self.total_certificate_refresh_requests
            ),
        }

    def _normalise_steps(
        self,
        current_step,
        device: torch.device,
    ) -> torch.Tensor:
        steps = torch.as_tensor(
            current_step, device=device, dtype=torch.long
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

    def _ensure_rng(self, device: torch.device) -> None:
        if self._rng is not None and self._rng_device == device:
            return
        try:
            self._rng = torch.Generator(device=device)
        except (RuntimeError, TypeError):
            self._rng = torch.Generator(device="cpu")
        self._rng.manual_seed(self.random_seed)
        self._rng_device = device

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

    def prepare_cache(
        self,
        payload_dim: int,
        device: torch.device,
        dtype: torch.dtype,
    ) -> None:
        self._ensure_cache(payload_dim, device, dtype)

    def reset(self, env_mask=None) -> None:
        """Reset cache and pending packets for all/selected environments."""
        if self.cache is not None:
            self.cache.reset(env_mask)
        if env_mask is None:
            self._pending.clear()
            return
        mask = torch.as_tensor(env_mask, dtype=torch.bool).reshape(-1)
        if mask.numel() != self.batch_size:
            raise ValueError(
                "env_mask must have one Boolean value per environment"
            )
        blocked = {i for i, value in enumerate(mask.tolist()) if value}
        self._pending = [
            packet for packet in self._pending if packet["env"] not in blocked
        ]

    def reset_agents(self, env_indices, agent_indices) -> None:
        """Reset cache and pending packets touching selected agents."""
        if self.cache is not None:
            self.cache.reset_agents(env_indices, agent_indices)
        env_values = torch.as_tensor(env_indices).reshape(-1).tolist()
        agent_values = torch.as_tensor(agent_indices).reshape(-1).tolist()
        blocked = set(zip(map(int, env_values), map(int, agent_values)))
        self._pending = [
            packet
            for packet in self._pending
            if (packet["env"], packet["ego"]) not in blocked
            and (packet["env"], packet["source"]) not in blocked
        ]

    def cached_view(self, source_indices, current_step, ego_indices=None):
        if self.cache is None:
            raise RuntimeError("communication cache has not been initialized")
        source_indices = torch.as_tensor(
            source_indices, device=self.cache.device, dtype=torch.long
        )
        if source_indices.ndim != 3:
            raise ValueError("source_indices must have shape [batch, ego, slot]")
        batch, n_ego, n_slots = source_indices.shape
        if batch != self.batch_size:
            raise ValueError("source_indices batch dimension mismatch")
        if ego_indices is None:
            if n_ego != self.n_agents:
                raise ValueError(
                    "ego_indices is required when processing an ego subset"
                )
            ego_ids = torch.arange(
                self.n_agents, device=self.cache.device, dtype=torch.long
            )
        else:
            ego_ids = torch.as_tensor(
                ego_indices, device=self.cache.device, dtype=torch.long
            ).reshape(-1)
            if ego_ids.numel() != n_ego:
                raise ValueError("ego_indices must contain one ID per ego")
        steps = self._normalise_steps(current_step, self.cache.device)
        env_grid = torch.arange(
            self.batch_size, device=self.cache.device, dtype=torch.long
        ).reshape(batch, 1, 1).expand(batch, n_ego, n_slots)
        ego_grid = ego_ids.reshape(1, n_ego, 1).expand(
            batch, n_ego, n_slots
        )
        return (
            self.cache.messages[env_grid, ego_grid, source_indices],
            self.cache.valid[env_grid, ego_grid, source_indices],
            self.cache.age(steps)[env_grid, ego_grid, source_indices],
        )

    def _outage_mask(self, step_grid: torch.Tensor) -> torch.Tensor:
        blocked = torch.zeros_like(step_grid, dtype=torch.bool)
        for start, end in self.outage_windows:
            blocked |= (step_grid >= start) & (step_grid < end)
        return blocked

    def _deliver_pending(self, steps: torch.Tensor) -> int:
        if not self._pending or self.cache is None:
            return 0
        remaining = []
        delivered = 0
        for packet in self._pending:
            if int(steps[packet["env"]].item()) >= packet["arrival_step"]:
                self.cache.update(
                    env_indices=[packet["env"]],
                    ego_indices=[packet["ego"]],
                    source_indices=[packet["source"]],
                    payload=packet["payload"].unsqueeze(0),
                    # Keep the original measurement/send step so delay
                    # contributes to information age.
                    step=[packet["send_step"]],
                )
                delivered += 1
            else:
                remaining.append(packet)
        self._pending = remaining
        return delivered

    def _queue_or_deliver(
        self,
        *,
        env_indices: torch.Tensor,
        ego_indices: torch.Tensor,
        source_indices: torch.Tensor,
        payload: torch.Tensor,
        send_steps: torch.Tensor,
    ) -> int:
        if payload.numel() == 0:
            return 0
        if self.delay_steps == 0:
            self.cache.update(
                env_indices=env_indices,
                ego_indices=ego_indices,
                source_indices=source_indices,
                payload=payload,
                step=send_steps,
            )
            return int(payload.shape[0])

        for i in range(payload.shape[0]):
            self._pending.append(
                {
                    "env": int(env_indices[i].item()),
                    "ego": int(ego_indices[i].item()),
                    "source": int(source_indices[i].item()),
                    "payload": payload[i].detach().clone(),
                    "send_step": int(send_steps[i].item()),
                    "arrival_step": int(send_steps[i].item())
                    + self.delay_steps,
                }
            )
        return 0

    def apply(
        self,
        current_payload: torch.Tensor,
        source_indices: torch.Tensor,
        current_step,
        fallback_payload: torch.Tensor,
        eligible_mask: Optional[torch.Tensor] = None,
        trigger_mask: Optional[torch.Tensor] = None,
        ego_indices: Optional[torch.Tensor] = None,
    ) -> CommunicationResult:
        if current_payload.ndim != 4:
            raise ValueError("current_payload must have four dimensions")
        batch, n_ego, n_slots, payload_dim = current_payload.shape
        if batch != self.batch_size:
            raise ValueError("current_payload batch dimension mismatch")
        device = current_payload.device
        if self.payload_dim is None:
            self.payload_dim = int(payload_dim)
        elif self.payload_dim != int(payload_dim):
            raise ValueError(
                "Communication payload dimension changed after first use"
            )
        element_size = current_payload.element_size()
        if self.payload_element_size is None:
            self.payload_element_size = int(element_size)
        elif self.payload_element_size != int(element_size):
            raise ValueError(
                "Communication payload element size changed after first use"
            )

        if ego_indices is None:
            if n_ego != self.n_agents:
                raise ValueError(
                    "ego_indices is required when processing an ego subset"
                )
            ego_ids = torch.arange(
                self.n_agents, device=device, dtype=torch.long
            )
        else:
            ego_ids = torch.as_tensor(
                ego_indices, device=device, dtype=torch.long
            ).reshape(-1)
            if ego_ids.numel() != n_ego:
                raise ValueError(
                    "ego_indices must contain one ID per ego dimension"
                )
            if bool(torch.any(ego_ids < 0).item()) or bool(
                torch.any(ego_ids >= self.n_agents).item()
            ):
                raise IndexError("ego_indices contains an invalid agent ID")

        if tuple(source_indices.shape) != (batch, n_ego, n_slots):
            raise ValueError("source_indices shape mismatch")
        if tuple(fallback_payload.shape) != tuple(current_payload.shape):
            raise ValueError("fallback_payload shape mismatch")

        source_indices = source_indices.to(device=device, dtype=torch.long)
        fallback_payload = fallback_payload.to(
            device=device, dtype=current_payload.dtype
        )
        steps = self._normalise_steps(current_step, device)
        if eligible_mask is None:
            eligible_mask = torch.ones(
                batch, n_ego, n_slots, device=device, dtype=torch.bool
            )
        else:
            eligible_mask = eligible_mask.to(device=device, dtype=torch.bool)
        if tuple(eligible_mask.shape) != (batch, n_ego, n_slots):
            raise ValueError("eligible_mask shape mismatch")

        if self.mode == "perfect":
            send_mask = eligible_mask.clone()
            age = torch.zeros_like(source_indices)
            messages = int(send_mask.sum().item())
            bytes_sent = messages * payload_dim * element_size
            self._record_statistics(
                valid_mask=eligible_mask,
                age=age,
                eligible_mask=eligible_mask,
                messages_sent=messages,
                messages_received=messages,
                bytes_sent=bytes_sent,
                bytes_received=bytes_sent,
            )
            return CommunicationResult(
                payload=current_payload,
                send_mask=send_mask,
                valid_mask=eligible_mask.clone(),
                age=age,
                messages_sent=messages,
                messages_received=messages,
                bytes_sent=bytes_sent,
                bytes_received=bytes_sent,
                delivered_mask=send_mask.clone(),
                loss_mask=torch.zeros_like(send_mask),
                outage_mask=torch.zeros_like(send_mask),
            )

        self._ensure_cache(payload_dim, device, current_payload.dtype)
        self._ensure_rng(device)
        arrivals = self._deliver_pending(steps)

        env_grid = torch.arange(
            batch, device=device, dtype=torch.long
        ).reshape(batch, 1, 1).expand(batch, n_ego, n_slots)
        ego_grid = ego_ids.reshape(1, n_ego, 1).expand(
            batch, n_ego, n_slots
        )
        step_grid = steps.reshape(batch, 1, 1).expand(
            batch, n_ego, n_slots
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
                .expand(batch, n_ego, n_slots)
                == 0
            )
        elif self.mode == "age":
            send_mask = (~cached_valid) | (
                cached_age >= int(self.max_age_steps)
            )
        else:
            if trigger_mask is None:
                raise ValueError(
                    "trigger_mask is required for certificate communication"
                )
            certificate_trigger = trigger_mask.to(
                device=device, dtype=torch.bool
            )
            if tuple(certificate_trigger.shape) != (
                batch,
                n_ego,
                n_slots,
            ):
                raise ValueError("trigger_mask shape mismatch")
            self.total_certificate_evaluations += int(
                eligible_mask.sum().item()
            )
            self.total_certificate_refresh_requests += int(
                (certificate_trigger & eligible_mask).sum().item()
            )
            send_mask = certificate_trigger

        send_mask = send_mask & eligible_mask
        send_mask = send_mask | ((~cached_valid) & eligible_mask)

        outage_mask = send_mask & self._outage_mask(step_grid)
        if self.loss_probability == 0.0:
            loss_mask = torch.zeros_like(send_mask)
        elif self.loss_probability == 1.0:
            loss_mask = send_mask & (~outage_mask)
        else:
            random_values = torch.rand(
                send_mask.shape,
                device=device,
                generator=self._rng,
            )
            loss_mask = send_mask & (~outage_mask) & (
                random_values < self.loss_probability
            )

        delivered_mask = send_mask & (~outage_mask) & (~loss_mask)
        flat_delivered = delivered_mask.reshape(-1)
        immediate_deliveries = 0
        if bool(flat_delivered.any().item()):
            immediate_deliveries = self._queue_or_deliver(
                env_indices=env_grid.reshape(-1)[flat_delivered],
                ego_indices=ego_grid.reshape(-1)[flat_delivered],
                source_indices=source_indices.reshape(-1)[flat_delivered],
                payload=current_payload.detach()
                .reshape(-1, payload_dim)[flat_delivered],
                send_steps=step_grid.reshape(-1)[flat_delivered],
            )

        dropped_now = int((loss_mask | outage_mask).sum().item())
        self.total_dropped_messages += dropped_now
        self.total_outage_blocked_messages += int(outage_mask.sum().item())
        if self.delay_steps > 0:
            self.total_delayed_messages += int(delivered_mask.sum().item())

        # Read after immediate delivery; delayed packets intentionally remain
        # invisible until a future apply call reaches their arrival step.
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
            usable.unsqueeze(-1), cached_payload, fallback_payload
        )

        messages_sent = int(send_mask.sum().item())
        bytes_sent = messages_sent * payload_dim * element_size
        messages_received = arrivals + immediate_deliveries
        bytes_received = messages_received * payload_dim * element_size
        self._record_statistics(
            valid_mask=usable,
            age=cached_age,
            eligible_mask=eligible_mask,
            messages_sent=messages_sent,
            messages_received=messages_received,
            bytes_sent=bytes_sent,
            bytes_received=bytes_received,
        )

        return CommunicationResult(
            payload=output_payload,
            send_mask=send_mask,
            valid_mask=usable,
            age=cached_age,
            messages_sent=messages_sent,
            messages_received=messages_received,
            bytes_sent=bytes_sent,
            bytes_received=bytes_received,
            delivered_mask=delivered_mask,
            loss_mask=loss_mask,
            outage_mask=outage_mask,
        )
