import torch

from utilities.communication_cache import CommunicationCache


def main() -> None:
    cache = CommunicationCache(
        batch_size=1,
        n_agents=2,
        payload_dim=3,
        device="cpu",
    )

    message_1 = torch.tensor([[1.0, 2.0, 3.0]])
    cache.update(
        env_indices=[0],
        ego_indices=[0],
        source_indices=[1],
        payload=message_1,
        step=4,
    )

    received, valid, age = cache.read(
        env_indices=[0],
        ego_indices=[0],
        source_indices=[1],
        current_step=4,
    )
    assert bool(valid.item())
    assert age.item() == 0
    assert torch.equal(received, message_1)
    print(
        f"CACHE receive: valid={bool(valid.item())}, age={age.item()}, "
        f"payload={received.flatten().tolist()}"
    )

    _, valid, age = cache.read(
        env_indices=[0],
        ego_indices=[0],
        source_indices=[1],
        current_step=5,
    )
    assert bool(valid.item())
    assert age.item() == 1
    print(f"CACHE stale-age increase: valid={bool(valid.item())}, age={age.item()}")

    message_2 = torch.tensor([[4.0, 5.0, 6.0]])
    cache.update([0], [0], [1], message_2, step=5)
    received, valid, age = cache.read([0], [0], [1], current_step=5)
    assert bool(valid.item())
    assert age.item() == 0
    assert torch.equal(received, message_2)
    print(
        f"CACHE refresh: valid={bool(valid.item())}, age={age.item()}, "
        f"payload={received.flatten().tolist()}"
    )

    cache.reset()
    _, valid, age = cache.read([0], [0], [1], current_step=6)
    assert not bool(valid.item())
    assert age.item() == -1
    print(f"CACHE reset: valid={bool(valid.item())}, age={age.item()}")

    vector_cache = CommunicationCache(
        batch_size=2,
        n_agents=2,
        payload_dim=3,
        device="cpu",
    )

    vector_cache.update(
        env_indices=[0, 1],
        ego_indices=[0, 0],
        source_indices=[1, 1],
        payload=torch.tensor(
            [
                [10.0, 11.0, 12.0],
                [20.0, 21.0, 22.0],
            ]
        ),
        step=[4, 7],
    )

    vector_ages = vector_cache.age(torch.tensor([5, 9]))
    assert vector_ages[0, 0, 1].item() == 1
    assert vector_ages[1, 0, 1].item() == 2

    vector_messages, vector_valid, selected_ages = vector_cache.read(
        env_indices=[0, 1],
        ego_indices=[0, 0],
        source_indices=[1, 1],
        current_step=torch.tensor([5, 9]),
    )

    assert torch.equal(
        vector_messages,
        torch.tensor(
            [
                [10.0, 11.0, 12.0],
                [20.0, 21.0, 22.0],
            ]
        ),
    )
    assert vector_valid.tolist() == [True, True]
    assert selected_ages.tolist() == [1, 2]

    print(
        f"CACHE vectorized ages: valid={vector_valid.tolist()}, "
        f"ages={selected_ages.tolist()}"
    )

    print("PASS: communication cache receive, aging, refresh, and reset")


if __name__ == "__main__":
    main()
