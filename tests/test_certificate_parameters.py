from utilities.helper_training import Parameters


def main() -> None:
    parameters = Parameters(
        communication_mode="certificate",
    )

    assert parameters.certificate_relative_speed_bound_mps == 2.0
    assert parameters.certificate_base_position_error_m == 0.0
    assert parameters.certificate_required_clearance_m == 0.0
    assert parameters.certificate_trigger_threshold_m == 0.0

    saved = parameters.to_dict()

    for key in (
        "certificate_relative_speed_bound_mps",
        "certificate_base_position_error_m",
        "certificate_required_clearance_m",
        "certificate_trigger_threshold_m",
    ):
        assert key in saved

    invalid_values = (
        {"certificate_relative_speed_bound_mps": -1.0},
        {"certificate_base_position_error_m": -0.01},
        {"certificate_required_clearance_m": -0.01},
        {"certificate_trigger_threshold_m": -0.01},
    )

    for invalid in invalid_values:
        try:
            Parameters(**invalid)
        except ValueError:
            pass
        else:
            raise AssertionError(
                f"Expected ValueError for invalid parameters: {invalid}"
            )

    print("CERTIFICATE PARAMETERS: defaults and serialization passed")
    print("CERTIFICATE PARAMETERS: negative-value validation passed")
    print("PASS: certificate parameter configuration")


if __name__ == "__main__":
    main()
