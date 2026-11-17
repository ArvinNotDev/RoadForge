"""Reliable CARLA weather preset lookup, application, and verification."""

from math import isclose


# CARLA exposes preset objects on carla.WeatherParameters, not on carla itself.
# Keep the constructor field list compatible with CARLA 0.9.16 and 1.0.
WEATHER_FIELDS = (
    "cloudiness",
    "precipitation",
    "precipitation_deposits",
    "wind_intensity",
    "sun_azimuth_angle",
    "sun_altitude_angle",
    "fog_density",
    "fog_distance",
    "wetness",
    "fog_falloff",
    "scattering_intensity",
    "mie_scattering_scale",
    "rayleigh_scattering_scale",
)

VERIFY_FIELDS = (
    "cloudiness",
    "precipitation",
    "precipitation_deposits",
    "wetness",
    "sun_altitude_angle",
    "sun_azimuth_angle",
    "fog_density",
    "fog_distance",
)


def weather_parameter_values(weather):
    """Extract comparable numeric values from a CARLA WeatherParameters object."""
    values = {}
    for field in WEATHER_FIELDS:
        try:
            value = getattr(weather, field)
            values[field] = float(value)
        except (AttributeError, TypeError, ValueError):
            continue
    return values


def resolve_weather_preset(carla_module, preset_name):
    """Resolve and copy a named preset from carla.WeatherParameters."""
    name = str(preset_name or "").strip()
    if not name or name == "Default":
        return None

    weather_type = getattr(carla_module, "WeatherParameters", None)
    if weather_type is None:
        raise RuntimeError(
            "The imported CARLA Python API does not expose carla.WeatherParameters."
        )

    try:
        preset = getattr(weather_type, name)
    except AttributeError:
        available = sorted(
            candidate for candidate in dir(weather_type)
            if candidate and candidate[0].isupper() and not candidate.startswith("_")
        )
        available_text = ", ".join(available) if available else "(not exposed)"
        raise ValueError(
            f"Unknown CARLA weather preset '{name}'. "
            f"WeatherParameters exposes: {available_text}."
        ) from None

    values = weather_parameter_values(preset)
    if not values:
        raise TypeError(
            f"carla.WeatherParameters.{name} is not a readable weather preset."
        )

    # Presets are native static objects. Build a fresh instance so the collector
    # never passes a shared preset object that another caller could have mutated.
    try:
        return weather_type(**values)
    except Exception as exc:
        raise RuntimeError(
            f"Could not create an independent WeatherParameters instance for '{name}': {exc}"
        ) from exc


def weather_mismatches(actual, expected, tolerance=0.05):
    """Return fields that CARLA did not report at the requested preset values."""
    actual_values = weather_parameter_values(actual)
    expected_values = weather_parameter_values(expected)
    compared = [
        field for field in VERIFY_FIELDS
        if field in actual_values and field in expected_values
    ]
    if not compared:
        return {"verification": ("no comparable weather fields", "no comparable weather fields")}

    mismatches = {}
    for field in compared:
        observed = actual_values[field]
        requested = expected_values[field]
        if not isclose(observed, requested, rel_tol=0.0, abs_tol=float(tolerance)):
            mismatches[field] = (requested, observed)
    return mismatches


def format_weather_summary(weather):
    """Format the most useful visual weather fields for the collector log."""
    values = weather_parameter_values(weather)
    keys = (
        "cloudiness",
        "precipitation",
        "precipitation_deposits",
        "wetness",
        "sun_altitude_angle",
        "fog_density",
    )
    return " · ".join(
        f"{key}={values[key]:g}" for key in keys if key in values
    ) or str(weather)


def apply_weather_preset(world, carla_module, preset_name, advance=None):
    """Apply a named preset, advance one sync frame if requested, and verify it."""
    preset = resolve_weather_preset(carla_module, preset_name)
    if preset is None:
        return None

    world.set_weather(preset)
    if advance is not None:
        advance()

    try:
        actual = world.get_weather()
    except Exception as exc:
        raise RuntimeError(
            f"CARLA accepted set_weather(), but World.get_weather() could not verify it: {exc}"
        ) from exc

    mismatches = weather_mismatches(actual, preset)
    if mismatches:
        details = "; ".join(
            f"{field}: requested {requested:g}, simulator reports {observed:g}"
            if isinstance(requested, (int, float)) and isinstance(observed, (int, float))
            else f"{field}: {requested} / {observed}"
            for field, (requested, observed) in mismatches.items()
        )
        raise RuntimeError(
            "CARLA's reported active weather does not match the selected preset. " + details
        )
    return actual
