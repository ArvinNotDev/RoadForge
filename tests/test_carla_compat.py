from datacollector.carla_compat import (
    sticky_control_error_message,
    version_mismatch_message,
)


def test_matching_carla_client_and_server_versions_are_accepted():
    assert version_mismatch_message("1.0.0", "1.0.0", "carla.pyd") is None


def test_client_server_version_mismatch_has_actionable_diagnostic():
    message = version_mismatch_message(
        "0.9.16-abc", "1.0.0", r"C:\CARLA_0.9.16\PythonAPI\carla.pyd"
    )

    assert "version mismatch" in message
    assert "0.9.16-abc" in message
    assert "1.0.0" in message
    assert r"C:\CARLA_0.9.16" in message
    assert "exact same CARLA server release" in message


def test_sticky_control_error_explains_api_mismatch_and_shows_versions():
    message = sticky_control_error_message("0.9.16", "1.0.0", "carla.pyd")

    assert "sticky_control" in message
    assert "does not match" in message
    assert "0.9.16" in message
    assert "1.0.0" in message
    assert "carla.pyd" in message
