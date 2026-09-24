"""Unit tests for the impairment-aware communication adapter.

Copy this file to the repository's ``tests/`` directory after copying the
adapter replacement.
"""

import torch

from utilities.communication_adapter import CommunicationAdapter


def _inputs(value):
    payload = torch.tensor([[[[float(value)]]]])
    source = torch.tensor([[[1]]])
    fallback = torch.tensor([[[[-1.0]]]])
    ego = torch.tensor([0])
    return payload, source, fallback, ego


def _apply(adapter, value, step):
    payload, source, fallback, ego = _inputs(value)
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
    result = _apply(adapter, 7.0, 0)
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
    first = _apply(adapter, 1.0, 0)
    assert first.messages_sent == 1
    assert first.messages_received == 0
    assert first.valid_mask.item() is False

    _apply(adapter, 2.0, 1)
    arrived = _apply(adapter, 3.0, 2)
    assert arrived.messages_received == 1
    assert arrived.payload.item() == 1.0
    assert arrived.age.item() == 2
    assert adapter.statistics()["delayed_messages"] >= 1


def test_random_loss_counts_attempts_separately():
    adapter = CommunicationAdapter(
        batch_size=1,
        n_agents=2,
        mode="periodic",
        period_steps=1,
        loss_probability=1.0,
        random_seed=123,
    )
    result = _apply(adapter, 1.0, 0)
    stats = adapter.statistics()
    assert result.messages_sent == 1
    assert result.messages_received == 0
    assert stats["dropped_messages"] == 1
    assert stats["bytes_received"] == 0


def test_outage_is_half_open_and_restores_after_end():
    adapter = CommunicationAdapter(
        batch_size=1,
        n_agents=2,
        mode="periodic",
        period_steps=1,
        outage_windows=[(0, 2)],
    )
    blocked = _apply(adapter, 1.0, 0)
    assert blocked.messages_sent == 1
    assert blocked.messages_received == 0
    assert blocked.outage_mask.item() is True

    restored = _apply(adapter, 2.0, 2)
    assert restored.messages_received == 1
    assert restored.payload.item() == 2.0


def test_invalid_impairment_values_are_rejected():
    invalid_values = (
        {"delay_steps": -1},
        {"loss_probability": -0.1},
        {"loss_probability": 1.1},
        {"outage_windows": [(3, 3)]},
    )
    for kwargs in invalid_values:
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


def test_reset_clears_pending_packets():
    adapter = CommunicationAdapter(
        batch_size=1,
        n_agents=2,
        mode="periodic",
        period_steps=1,
        delay_steps=2,
    )

    first = _apply(adapter, 1.0, 0)
    assert first.messages_sent == 1
    assert first.messages_received == 0

    adapter.reset()

    after_reset = _apply(adapter, 2.0, 2)
    assert after_reset.messages_received == 0
    assert after_reset.payload.item() == -1.0


def test_agent_reset_clears_agent_pending_packets():
    adapter = CommunicationAdapter(
        batch_size=1,
        n_agents=2,
        mode="periodic",
        period_steps=1,
        delay_steps=2,
    )

    first = _apply(adapter, 1.0, 0)
    assert first.messages_sent == 1
    assert first.messages_received == 0

    adapter.reset_agents(
        env_indices=[0],
        agent_indices=[0],
    )

    after_reset = _apply(adapter, 2.0, 2)
    assert after_reset.messages_received == 0
    assert after_reset.payload.item() == -1.0


if __name__ == "__main__":
    test_default_behavior_is_lossless_and_immediate()
    test_delay_keeps_old_payload_until_arrival_and_preserves_age()
    test_random_loss_counts_attempts_separately()
    test_outage_is_half_open_and_restores_after_end()
    test_invalid_impairment_values_are_rejected()
    test_reset_clears_pending_packets()
    test_agent_reset_clears_agent_pending_packets()
    print("PASS: communication impairment tests")
