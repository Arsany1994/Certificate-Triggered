import torch

from utilities.communication_adapter import CommunicationAdapter


def apply_once(adapter, step):
    current_payload = torch.tensor(
        [[[[float(step), 10.0, 20.0]]]],
        dtype=torch.float32,
    )
    fallback_payload = torch.tensor(
        [[[[-1.0, -1.0, -1.0]]]],
        dtype=torch.float32,
    )

    return adapter.apply(
        current_payload=current_payload,
        source_indices=torch.tensor([[[1]]], dtype=torch.long),
        current_step=step,
        fallback_payload=fallback_payload,
        eligible_mask=torch.ones(
            (1, 1, 1),
            dtype=torch.bool,
        ),
        ego_indices=torch.tensor([0], dtype=torch.long),
    )


def test_mode(mode, **kwargs):
    adapter = CommunicationAdapter(
        batch_size=1,
        n_agents=2,
        mode=mode,
        **kwargs,
    )

    outputs = []
    sends = []

    for step in range(4):
        result = apply_once(adapter, step)
        outputs.append(float(result.payload[0, 0, 0, 0].item()))
        sends.append(bool(result.send_mask.item()))

    stats = adapter.statistics()

    print(f"MODE: {mode}")
    print(f"OUTPUT PAYLOAD FIRST VALUE: {outputs}")
    print(f"SEND MASKS: {sends}")
    print(f"STATISTICS: {stats}")

    assert stats["payload_dim"] == 3
    assert stats["payload_element_size_bytes"] == 4
    assert stats["fallback_events"] == 0
    assert stats["max_valid_information_age"] >= 0

    return outputs, sends, stats


perfect_outputs, perfect_sends, perfect_stats = test_mode("perfect")
assert perfect_outputs == [0.0, 1.0, 2.0, 3.0]
assert perfect_sends == [True, True, True, True]
assert perfect_stats["messages_sent"] == 4
assert perfect_stats["bytes_sent"] == 4 * 3 * 4

periodic_outputs, periodic_sends, periodic_stats = test_mode(
    "periodic",
    period_steps=2,
)
assert periodic_outputs == [0.0, 0.0, 2.0, 2.0]
assert periodic_sends == [True, False, True, False]
assert periodic_stats["messages_sent"] == 2
assert periodic_stats["bytes_sent"] == 2 * 3 * 4

age_outputs, age_sends, age_stats = test_mode(
    "age",
    max_age_steps=2,
)
assert age_outputs == [0.0, 0.0, 2.0, 2.0]
assert age_sends == [True, False, True, False]
assert age_stats["messages_sent"] == 2
assert age_stats["bytes_sent"] == 2 * 3 * 4

print("PASS: communication accounting for perfect, periodic, and age modes")
