from __future__ import annotations

from typing import Optional

import torch


class CommunicationCache:
    """Timestamped per-environment, per-agent communication cache.

    The cache is deliberately independent of SigmaRL's simulator-history
    buffers. It stores only payloads explicitly delivered through the
    communication interface.
    """

    def __init__(
        self,
        batch_size: int,
        n_agents: int,
        payload_dim: int,
        *,
        device: str | torch.device = "cpu",
        dtype: torch.dtype = torch.float32,
    ) -> None:
        if batch_size <= 0:
            raise ValueError("batch_size must be positive")
        if n_agents <= 0:
            raise ValueError("n_agents must be positive")
        if payload_dim <= 0:
            raise ValueError("payload_dim must be positive")

        self.batch_size = int(batch_size)
        self.n_agents = int(n_agents)
        self.payload_dim = int(payload_dim)
        self.device = torch.device(device)
        self.dtype = dtype

        self.messages = torch.zeros(
            (
                self.batch_size,
                self.n_agents,
                self.n_agents,
                self.payload_dim,
            ),
            device=self.device,
            dtype=self.dtype,
        )
        self.valid = torch.zeros(
            (
                self.batch_size,
                self.n_agents,
                self.n_agents,
            ),
            device=self.device,
            dtype=torch.bool,
        )
        self.last_update_step = torch.full(
            (
                self.batch_size,
                self.n_agents,
                self.n_agents,
            ),
            -1,
            device=self.device,
            dtype=torch.long,
        )

    def _indices(
        self,
        values,
        *,
        name: str,
        upper_bound: int,
    ) -> torch.Tensor:
        indices = torch.as_tensor(
            values,
            device=self.device,
            dtype=torch.long,
        ).reshape(-1)

        if indices.numel() == 0:
            raise ValueError(f"{name} cannot be empty")

        if bool(torch.any(indices < 0).item()) or bool(
            torch.any(indices >= upper_bound).item()
        ):
            raise IndexError(
                f"{name} contains an index outside [0, {upper_bound})"
            )

        return indices

    def reset(self, env_mask: Optional[torch.Tensor] = None) -> None:
        """Reset all environments or only the environments selected by a mask."""
        if env_mask is None:
            self.messages.zero_()
            self.valid.zero_()
            self.last_update_step.fill_(-1)
            return

        mask = torch.as_tensor(
            env_mask,
            device=self.device,
            dtype=torch.bool,
        ).reshape(-1)

        if mask.numel() != self.batch_size:
            raise ValueError(
                "env_mask must have one Boolean value per environment"
            )

        self.messages[mask] = 0
        self.valid[mask] = False
        self.last_update_step[mask] = -1

    def reset_agents(self, env_indices, agent_indices) -> None:
        """Invalidate incoming and outgoing messages for reset agents.

        Both directions are cleared:
        - ego agent receives from all sources;
        - all egos receive from the reset source agent.
        """
        env = self._indices(
            env_indices,
            name="env_indices",
            upper_bound=self.batch_size,
        )
        agent = self._indices(
            agent_indices,
            name="agent_indices",
            upper_bound=self.n_agents,
        )

        if env.numel() != agent.numel():
            raise ValueError(
                "env_indices and agent_indices must have the same number of entries"
            )

        for env_i, agent_i in zip(env.tolist(), agent.tolist()):
            self.messages[env_i, agent_i, :] = 0
            self.messages[env_i, :, agent_i] = 0

            self.valid[env_i, agent_i, :] = False
            self.valid[env_i, :, agent_i] = False

            self.last_update_step[env_i, agent_i, :] = -1
            self.last_update_step[env_i, :, agent_i] = -1

    def update(
        self,
        env_indices,
        ego_indices,
        source_indices,
        payload: torch.Tensor,
        step,
    ) -> None:
        """Store one or more explicitly received messages."""
        env = self._indices(
            env_indices,
            name="env_indices",
            upper_bound=self.batch_size,
        )
        ego = self._indices(
            ego_indices,
            name="ego_indices",
            upper_bound=self.n_agents,
        )
        source = self._indices(
            source_indices,
            name="source_indices",
            upper_bound=self.n_agents,
        )

        n_updates = env.numel()
        if ego.numel() != n_updates or source.numel() != n_updates:
            raise ValueError(
                "env_indices, ego_indices, and source_indices must have "
                "the same number of entries"
            )

        payload_tensor = torch.as_tensor(
            payload,
            device=self.device,
            dtype=self.dtype,
        )
        if payload_tensor.ndim == 1:
            if n_updates != 1 or payload_tensor.numel() != self.payload_dim:
                raise ValueError(
                    "A one-dimensional payload is allowed only for one update"
                )
            payload_tensor = payload_tensor.unsqueeze(0)

        expected_shape = (n_updates, self.payload_dim)
        if tuple(payload_tensor.shape) != expected_shape:
            raise ValueError(
                f"payload must have shape {expected_shape}, "
                f"received {tuple(payload_tensor.shape)}"
            )

        step_tensor = torch.as_tensor(
            step,
            device=self.device,
            dtype=torch.long,
        )
        if step_tensor.ndim == 0:
            step_tensor = step_tensor.expand(n_updates)
        else:
            step_tensor = step_tensor.reshape(-1)
            if step_tensor.numel() != n_updates:
                raise ValueError(
                    "step must be scalar or contain one value per update"
                )

        if bool(torch.any(step_tensor < 0).item()):
            raise ValueError("decision steps must be non-negative")

        self.messages[env, ego, source] = payload_tensor
        self.valid[env, ego, source] = True
        self.last_update_step[env, ego, source] = step_tensor

    def age(self, current_step) -> torch.Tensor:
        """Return age for every cache entry.

        ``current_step`` may be either:
        - a scalar, which is broadcast to every environment; or
        - one integer step value per environment.
        """
        current = torch.as_tensor(
            current_step,
            device=self.device,
            dtype=torch.long,
        ).reshape(-1)

        if current.numel() == 1:
            current = current.expand(self.batch_size)
        elif current.numel() == self.batch_size:
            current = current.reshape(self.batch_size)
        else:
            raise ValueError(
                "current_step must be scalar or contain one value per environment"
            )

        if bool(torch.any(current < 0).item()):
            raise ValueError("current_step values must be non-negative")

        ages = current[:, None, None] - self.last_update_step
        invalid_age = torch.full_like(ages, -1)
        return torch.where(self.valid, ages, invalid_age)

    def read(
        self,
        env_indices,
        ego_indices,
        source_indices,
        current_step: int,
    ):
        """Read payload, validity, and age for selected cache entries."""
        env = self._indices(
            env_indices,
            name="env_indices",
            upper_bound=self.batch_size,
        )
        ego = self._indices(
            ego_indices,
            name="ego_indices",
            upper_bound=self.n_agents,
        )
        source = self._indices(
            source_indices,
            name="source_indices",
            upper_bound=self.n_agents,
        )

        if not (env.numel() == ego.numel() == source.numel()):
            raise ValueError(
                "read indices must have the same number of entries"
            )

        ages = self.age(current_step)
        return (
            self.messages[env, ego, source].clone(),
            self.valid[env, ego, source].clone(),
            ages[env, ego, source].clone(),
        )

    def is_stale(self, current_step: int, max_age_steps: int) -> torch.Tensor:
        """Return True for invalid or older-than-allowed cache entries."""
        if max_age_steps < 0:
            raise ValueError("max_age_steps must be non-negative")
        return (~self.valid) | (self.age(current_step) > max_age_steps)

