"""Helpers for validating map names returned by the live CARLA server."""


def normalize_map_name(map_reference):
    """Return the display name from a CARLA map name or full package reference."""
    reference = str(map_reference or "").strip().rstrip("/")
    return reference.rsplit("/", 1)[-1] if reference else ""


def build_map_reference_index(map_references):
    """Index exact server-provided map references by their display name."""
    indexed = {}
    for raw_reference in map_references:
        reference = str(raw_reference or "").strip().rstrip("/")
        name = normalize_map_name(reference)
        if not name:
            continue
        previous = indexed.get(name)
        if previous is not None and previous != reference:
            raise ValueError(
                f"CARLA reported multiple map references with the same name '{name}': "
                f"'{previous}' and '{reference}'."
            )
        indexed[name] = reference
    return dict(sorted(indexed.items()))


def validate_selected_map_names(selected_names, map_references):
    """Reject stale/unsupported selections before any dataset or world changes."""
    selected = list(dict.fromkeys(
        str(name).strip() for name in selected_names if str(name).strip()
    ))
    unavailable = [name for name in selected if name not in map_references]
    if unavailable:
        available = ", ".join(sorted(map_references)) or "(none)"
        raise ValueError(
            "Selected CARLA map(s) are not installed on the connected server: "
            + ", ".join(unavailable)
            + ". Maps reported by the server: "
            + available
            + ". Reconnect to refresh the map list."
        )
    return selected


def require_map_reference(map_name, map_references):
    """Resolve a selected display name without ever guessing a CARLA map path."""
    try:
        return map_references[map_name]
    except KeyError:
        raise ValueError(
            f"CARLA map '{map_name}' was not discovered on the connected server. "
            "Reconnect and select an installed map."
        ) from None
