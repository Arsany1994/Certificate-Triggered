import torch

from utilities.communication_adapter import CommunicationAdapter


def main() -> None:
    current_0 = torch.tensor(
        [
            [
                [[1.0, 2.0]],
                [[3.0, 4.0]],
            ],
            [
                [[5.0, 6.0]],
                [[7.0, 8.0]],
            ],
        ]
    )

    current_1 = current_0 + 10.0

    source_indices = torch.tensor(
        [
            [[1], [0]],
            [[1], [0]],
        ],
        dtype=torch.long,
    )

    fallback = torch.full_like(current_0, -9.0)
    steps_0 = torch.tensor([0, 0])
    steps_1 = torch.tensor([1, 1])
    steps_2 = torch.tensor([2, 2])

    perfect = CommunicationAdapter(
        batch_size=2,
        n_agents=2,
        mode="perfect",
    )

    perfect_result = perfect.apply(
        current_payload=current_0,
        source_indices=source_indices,
        current_step=steps_0,
        fallback_payload=fallback,
    )

    assert torch.equal(perfect_result.payload, current_0)
    assert perfect_result.messages_sent == 4
    print("ADAPTER perfect: current payload preserved")

    periodic = CommunicationAdapter(
        batch_size=2,
        n_agents=2,
        mode="periodic",
        period_steps=2,
    )

    first = periodic.apply(
        current_payload=current_0,
        source_indices=source_indices,
        current_step=steps_0,
        fallback_payload=fallback,
    )

    assert torch.equal(first.payload, current_0)
    assert first.messages_sent == 4

    second = periodic.apply(
        current_payload=current_1,
        source_indices=source_indices,
        current_step=steps_1,
        fallback_payload=fallback,
    )

    assert torch.equal(second.payload, current_0)
    assert second.messages_sent == 0

    third = periodic.apply(
        current_payload=current_1,
        source_indices=source_indices,
        current_step=steps_2,
        fallback_payload=fallback,
    )

    assert torch.equal(third.payload, current_1)
    assert third.messages_sent == 4
    print("ADAPTER periodic: stale payload retained between periods")

    age = CommunicationAdapter(
        batch_size=2,
        n_agents=2,
        mode="age",
        max_age_steps=2,
    )

    age.apply(
        current_payload=current_0,
        source_indices=source_indices,
        current_step=steps_0,
        fallback_payload=fallback,
    )

    age_result = age.apply(
        current_payload=current_1,
        source_indices=source_indices,
        current_step=steps_1,
        fallback_payload=fallback,
    )

    assert torch.equal(age_result.payload, current_0)
    assert age_result.messages_sent == 0
    print("ADAPTER age: age threshold not reached")

    certificate = CommunicationAdapter(
        batch_size=2,
        n_agents=2,
        mode="certificate",
    )

    certificate.apply(
        current_payload=current_0,
        source_indices=source_indices,
        current_step=steps_0,
        fallback_payload=fallback,
        trigger_mask=torch.ones(2, 2, 1, dtype=torch.bool),
    )

    certificate_result = certificate.apply(
        current_payload=current_1,
        source_indices=source_indices,
        current_step=steps_1,
        fallback_payload=fallback,
        trigger_mask=torch.zeros(2, 2, 1, dtype=torch.bool),
    )

    assert torch.equal(certificate_result.payload, current_0)
    assert certificate_result.messages_sent == 0
    print("ADAPTER certificate interface: external trigger respected")

    single_ego = CommunicationAdapter(
        batch_size=1,
        n_agents=3,
        mode="periodic",
        period_steps=2,
    )

    single_result = single_ego.apply(
        current_payload=torch.tensor(
            [[[[9.0, 10.0]]]]
        ),
        source_indices=torch.tensor(
            [[[2]]],
            dtype=torch.long,
        ),
        ego_indices=torch.tensor([1]),
        current_step=torch.tensor([0]),
        fallback_payload=torch.full(
            (1, 1, 1, 2),
            -9.0,
        ),
    )

    assert torch.equal(
        single_result.payload,
        torch.tensor([[[[9.0, 10.0]]]]),
    )
    assert single_ego.cache.valid[0, 1, 2].item()
    print("ADAPTER single ego: global ego/source identity preserved")

    print("PASS: communication adapter modes and fallback behavior")


if __name__ == "__main__":
    main()
