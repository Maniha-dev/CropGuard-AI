STUN_SERVER = "stun:stun.l.google.com:19302"


def build_rtc_configuration(turn_settings=None):
    ice_servers = [{"urls": [STUN_SERVER]}]
    if not turn_settings:
        return {"iceServers": ice_servers}

    urls = turn_settings.get("urls", [])
    if isinstance(urls, str):
        urls = [url.strip() for url in urls.split(",") if url.strip()]
    if not urls:
        return {"iceServers": ice_servers}
    if any(not url.lower().startswith(("turn:", "turns:")) for url in urls):
        raise ValueError("TURN URLs must start with 'turn:' or 'turns:'.")

    username = turn_settings.get("username")
    credential = turn_settings.get("credential")
    if not username or not credential:
        raise ValueError("TURN URLs require both a username and credential.")

    ice_servers.append(
        {"urls": urls, "username": username, "credential": credential}
    )
    return {"iceServers": ice_servers}