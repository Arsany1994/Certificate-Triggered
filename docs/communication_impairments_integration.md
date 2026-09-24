# Communication-impairment implementation

The complete adapter replacement is in
`communication_adapter_impairments.py`. Copy it over the repository's
`utilities/communication_adapter.py` only after making a clean checkpoint.

The implementation is deliberately backward compatible when all three new
parameters have their defaults:

```json
{
  "communication_delay_steps": 0,
  "communication_loss_probability": 0.0,
  "communication_outage_windows": [],
  "communication_random_seed": 0
}
```

`perfect` remains the ideal oracle and rejects non-default impairment values.
The impairment experiments should use `certificate`, `age`, or `periodic`.

## 1. Add fields to `Parameters`

In `utilities/helper_training.py`, append these arguments to the
`Parameters.__init__` signature, next to the existing communication fields:

```python
communication_delay_steps: int = 0,
communication_loss_probability: float = 0.0,
communication_outage_windows=None,
communication_random_seed: int = None,
```

After the existing assignments for `communication_period_steps` and
`communication_max_age_steps`, add:

```python
self.communication_delay_steps = int(communication_delay_steps)
self.communication_loss_probability = float(
    communication_loss_probability
)
self.communication_outage_windows = (
    [] if communication_outage_windows is None
    else [list(window) for window in communication_outage_windows]
)
self.communication_random_seed = (
    self.seed
    if communication_random_seed is None
    else int(communication_random_seed)
)

if self.communication_delay_steps < 0:
    raise ValueError("communication_delay_steps must be non-negative")
if not 0.0 <= self.communication_loss_probability <= 1.0:
    raise ValueError(
        "communication_loss_probability must be in [0, 1]"
    )
for window in self.communication_outage_windows:
    if len(window) != 2:
        raise ValueError(
            "communication_outage_windows entries must be [start, end]"
        )
    start, end = int(window[0]), int(window[1])
    if start < 0 or end <= start:
        raise ValueError(
            "outage windows require 0 <= start < end"
        )
```

Because `Parameters.to_dict()` returns `self.__dict__`, the new values are
automatically included in provenance/config files.

## 2. Replace the adapter

Copy the complete file from this directory into the repository:

```bash
cd /home/arsany/paper2/development/sigmarl-proposed
cp /path/to/communication_adapter_impairments.py utilities/communication_adapter.py
```

The replacement keeps the old result fields and adds:

```text
dropped_messages
delayed_messages
outage_blocked_messages
delay_steps
loss_probability
outage_windows
random_seed
```

The meaning is explicit:

* `messages_sent`/`bytes_sent`: scheduled transmission attempts.
* `messages_received`/`bytes_received`: packets that arrive at this step.
* a delayed packet contributes to `messages_received` only at arrival;
  its information age still includes the delay.
* loss and outage packets remain sent attempts but are not received.

## 3. Pass the fields from `ScenarioRoadTraffic`

In `scenarios/road_traffic.py`, extend the existing adapter constructor in
`_get_communication_adapter`:

```python
self.communication_adapter = CommunicationAdapter(
    batch_size=self.world.batch_dim,
    n_agents=self.n_agents,
    mode=self.parameters.communication_mode,
    period_steps=self.parameters.communication_period_steps,
    max_age_steps=self.parameters.communication_max_age_steps,
    delay_steps=self.parameters.communication_delay_steps,
    loss_probability=self.parameters.communication_loss_probability,
    outage_windows=self.parameters.communication_outage_windows,
    random_seed=self.parameters.communication_random_seed,
)
```

The adapter owns pending packets. Therefore, when the existing scenario reset
logic resets communication state, call the adapter as well:

```python
adapter = getattr(self, "communication_adapter", None)
if adapter is not None:
    if agent_index is not None:
        adapter.reset_agents([env_index], [agent_index])
    elif env_index is None:
        adapter.reset()
    else:
        env_mask = torch.zeros(
            self.world.batch_dim,
            device=self.world.device,
            dtype=torch.bool,
        )
        env_mask[env_index] = True
        adapter.reset(env_mask)
```

Keep the existing cache reset too; it is harmless and preserves the current
scenario-side cache lifecycle.

## 4. Add unit tests

Create `tests/test_communication_impairments.py`:

```python
import torch

from utilities.communication_adapter import CommunicationAdapter


def inputs(value):
    payload = torch.tensor([[[[float(value)]]]])
    source = torch.tensor([[[1]]])
    fallback = torch.tensor([[[[-1.0]]]])
    ego = torch.tensor([0])
    return payload, source, fallback, ego


def apply(adapter, value, step):
    payload, source, fallback, ego = inputs(value)
    return adapter.apply(
        current_payload=payload,
        source_indices=source,
        current_step=step,
        fallback_payload=fallback,
        ego_indices=ego,
    )


def test_default_behavior_is_lossless_and_immediate():
    adapter = CommunicationAdapter(
        batch_size=1,
        n_agents=2,
        mode="periodic",
        period_steps=1,
    )
    result = apply(adapter, 7.0, 0)
    assert result.messages_sent == 1
    assert result.messages_received == 1
    assert result.payload.item() == 7.0
    assert adapter.statistics()["dropped_messages"] == 0


def test_delay_keeps_old_payload_until_arrival_and_preserves_age():
    adapter = CommunicationAdapter(
        batch_size=1,
        n_agents=2,
        mode="periodic",
        period_steps=1,
        delay_steps=2,
    )
    first = apply(adapter, 1.0, 0)
    assert first.messages_sent == 1
    assert first.messages_received == 0
    assert first.valid_mask.item() is False

    apply(adapter, 2.0, 1)
    arrived = apply(adapter, 3.0, 2)
    assert arrived.messages_received == 1
    assert arrived.payload.item() == 1.0
    assert arrived.age.item() == 2
    assert adapter.statistics()["delayed_messages"] >= 1


def test_random_loss_is_seeded_and_counts_attempts_separately():
    a = CommunicationAdapter(
        batch_size=1,
        n_agents=2,
        mode="periodic",
        period_steps=1,
        loss_probability=1.0,
        random_seed=123,
    )
    result = apply(a, 1.0, 0)
    stats = a.statistics()
    assert result.messages_sent == 1
    assert result.messages_received == 0
    assert stats["dropped_messages"] == 1
    assert stats["bytes_received"] == 0


def test_outage_blocks_delivery_only_inside_half_open_window():
    adapter = CommunicationAdapter(
        batch_size=1,
        n_agents=2,
        mode="periodic",
        period_steps=1,
        outage_windows=[(0, 2)],
    )
    blocked = apply(adapter, 1.0, 0)
    assert blocked.messages_sent == 1
    assert blocked.messages_received == 0
    assert blocked.outage_mask.item() is True

    restored = apply(adapter, 2.0, 2)
    assert restored.messages_received == 1
    assert restored.payload.item() == 2.0


def test_invalid_impairment_values_are_rejected():
    for kwargs in (
        {"delay_steps": -1},
        {"loss_probability": -0.1},
        {"loss_probability": 1.1},
        {"outage_windows": [(3, 3)]},
    ):
        try:
            CommunicationAdapter(
                batch_size=1,
                n_agents=2,
                mode="periodic",
                period_steps=1,
                **kwargs,
            )
        except ValueError:
            continue
        raise AssertionError(f"invalid values accepted: {kwargs}")
```

## 5. Validate before any long simulation

Run these commands in the real repository after copying the adapter and
making the three integration edits:

```bash
cd /home/arsany/paper2/development/sigmarl-proposed

python -m py_compile \
  utilities/communication_adapter.py \
  utilities/helper_training.py \
  scenarios/road_traffic.py

python tests/test_communication_impairments.py

python tests/test_communication_adapter.py
python tests/test_communication_cache.py
python tests/test_communication_accounting.py
python tests/test_safety_certificate.py

git diff --check
```

The expected outcome is that all tests pass and `git diff --check` prints no
output.  Do not start the 250-iteration training or 32-rollout evaluation at
this stage; these tests establish that defaults are unchanged first.

## 6. Example impairment configuration

After the tests pass, create a separate evaluation config (do not alter the
training configs):

```json
{
  "communication_mode": "certificate",
  "communication_delay_steps": 2,
  "communication_loss_probability": 0.10,
  "communication_outage_windows": [[300, 340], [800, 840]],
  "communication_random_seed": 0
}
```

Use the same frozen policy and held-out scenario for the first smoke run. A
publication-grade sweep can then vary one impairment at a time:

```text
delay: 0, 1, 2, 4 steps
loss: 0%, 5%, 10%, 20%
outage: none, one 40-step outage, two 40-step outages
```

Keep the ideal `perfect` run as the no-impairment reference. Record the new
communication counters and the existing collision/speed/deviation metrics in
separate robustness tables.
