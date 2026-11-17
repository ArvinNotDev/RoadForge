"""CARLA client/server API compatibility diagnostics."""


def version_mismatch_message(client_version, server_version, module_path=None):
    """Return an actionable error when the connected client and server differ."""
    client_version = str(client_version or "").strip()
    server_version = str(server_version or "").strip()
    if not client_version or not server_version or client_version == server_version:
        return None

    message = (
        "CARLA Python API / Simulator version mismatch. "
        f"Client API: {client_version}; Simulator: {server_version}. "
        "Use the Python API shipped with the exact same CARLA server release, "
        "then restart RoadForge."
    )
    if module_path:
        message += f" Loaded Python module: {module_path}."
    return message


def sticky_control_error_message(client_version=None, server_version=None, module_path=None):
    """Explain the common Blueprint deserialization error caused by API mismatch."""
    versions = ""
    if client_version or server_version:
        versions = (
            f" Reported client API: {client_version or 'unknown'}; "
            f"simulator: {server_version or 'unknown'}."
        )
    module = f" Loaded Python module: {module_path}." if module_path else ""
    return (
        "CARLA could not deserialize the vehicle Blueprint metadata "
        "('sticky_control'). This usually means the imported CARLA Python API "
        "does not match the running Simulator build. Install/use the matching "
        "Python API from that CARLA installation and restart RoadForge."
        + versions
        + module
    )
