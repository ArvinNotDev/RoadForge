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



def select_collection_towns(
    town_specs,
    map_references,
    *,
    use_current_world=False,
    current_map_reference=None,
):
    """Resolve the collection targets, optionally restricting capture to get_world()."""
    town_specs = town_specs or {}
    if use_current_world:
        current_town = normalize_map_name(current_map_reference)
        if not current_town:
            raise ValueError(
                "Could not identify the currently loaded CARLA map from client.get_world()."
            )
        if current_town not in map_references:
            raise ValueError(
                f"The currently loaded CARLA map '{current_town}' was not discovered by the server. "
                "Reconnect to refresh the installed map list."
            )
        spec = town_specs.get(current_town)
        if not isinstance(spec, dict) or not bool(spec.get("enabled", False)):
            raise ValueError(
                f"Enable the currently loaded map '{current_town}' in the map targets before collecting."
            )
        try:
            target = int(spec.get("target", 0))
        except (TypeError, ValueError):
            target = 0
        if target <= 0:
            raise ValueError(
                f"Set a positive image target for the currently loaded map '{current_town}'."
            )
        return [current_town]

    selected = [
        name for name, spec in town_specs.items()
        if spec.get("enabled") and int(spec.get("target", 0)) > 0
    ]
    if not selected:
        raise ValueError("Select at least one installed CARLA map with a positive target")
    return validate_selected_map_names(selected, map_references)
