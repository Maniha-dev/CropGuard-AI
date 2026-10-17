import pytest

from src.webrtc_config import STUN_SERVER, build_rtc_configuration


def test_rtc_configuration_uses_stun_without_turn_settings():
    assert build_rtc_configuration() == {"iceServers": [{"urls": [STUN_SERVER]}]}


def test_rtc_configuration_adds_authenticated_turn_servers():
    configuration = build_rtc_configuration(
        {
            "urls": ["turn:turn.example:3478?transport=udp", "turns:turn.example:443"],
            "username": "user",
            "credential": "secret",
        }
    )

    assert configuration["iceServers"][1] == {
        "urls": ["turn:turn.example:3478?transport=udp", "turns:turn.example:443"],
        "username": "user",
        "credential": "secret",
    }


def test_rtc_configuration_rejects_incomplete_turn_settings():
    with pytest.raises(ValueError, match="username and credential"):
        build_rtc_configuration({"urls": "turn:turn.example:3478"})