import torch

from utilities.communication_adapter import CommunicationAdapter


def main() -> None:
    adapter = CommunicationAdapter(
        batch_size=1,
        n_agents=2,
        mode="certificate",
    )

    payload = torch.tensor(
        [[
            [
                [10.0, 11.0, 12.0],
                [20.0, 21.0, 22.0],
            ]
        ]]
    )

    source_indices = torch.tensor([[[0, 1]]])
    eligible_mask = torch.tensor([[[False, True]]])
    trigger_mask = torch.tensor([[[False, True]]])

    adapter.prepare_cache(
        payload_dim=3,
        device=payload.device,
        dtype=payload.dtype,
    )

    result = adapter.apply(
        current_payload=payload,
        source_indices=source_indices,
        current_step=0,
        fallback_payload=torch.zeros_like(payload),
        eligible_mask=eligible_mask,
        trigger_mask=trigger_mask,
        ego_indices=torch.tensor([0]),
    )

    assert result.send_mask.tolist() == [[[False, True]]]

    cached_payload, cached_valid, cached_age = adapter.cached_view(
        source_indices=source_indices,
        current_step=0,
        ego_indices=torch.tensor([0]),
    )

    assert bool(cached_valid[0, 0, 1].item())
    assert cached_age[0, 0, 1].item() == 0
    assert torch.equal(
        cached_payload[0, 0, 1],
        payload[0, 0, 1],
    )

    _, cached_valid_later, cached_age_later = adapter.cached_view(
        source_indices=source_indices,
        current_step=2,
        ego_indices=torch.tensor([0]),
    )

    assert bool(cached_valid_later[0, 0, 1].item())
    assert cached_age_later[0, 0, 1].item() == 2

    stats = adapter.statistics()

    assert stats["certificate_evaluations"] == 1
    assert stats["certificate_refresh_requests"] == 1

    print(
        "ADAPTER cached view: "
        f"valid={cached_valid.tolist()}, "
        f"age={cached_age.tolist()}"
    )

    print(
        "ADAPTER cached view after aging: "
        f"valid={cached_valid_later.tolist()}, "
        f"age={cached_age_later.tolist()}"
    )

    print(
        "ADAPTER certificate counters: "
        f"evaluations={stats['certificate_evaluations']}, "
        f"refresh_requests={stats['certificate_refresh_requests']}"
    )

    print("PASS: adapter cached-view and certificate accounting")


if __name__ == "__main__":
    main()
