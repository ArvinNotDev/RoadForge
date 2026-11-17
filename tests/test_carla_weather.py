from datacollector.weather import (
    apply_weather_preset,
    format_weather_summary,
    resolve_weather_preset,
    weather_mismatches,
)


class FakeWeather:
    def __init__(self, **values):
        defaults = {
            "cloudiness": 0.0,
            "precipitation": 0.0,
            "precipitation_deposits": 0.0,
            "wind_intensity": 0.0,
            "sun_azimuth_angle": 0.0,
            "sun_altitude_angle": 90.0,
            "fog_density": 0.0,
            "fog_distance": 0.0,
            "wetness": 0.0,
            "fog_falloff": 0.0,
            "scattering_intensity": 0.0,
            "mie_scattering_scale": 0.0,
            "rayleigh_scattering_scale": 0.0331,
        }
        defaults.update(values)
        for key, value in defaults.items():
            setattr(self, key, float(value))


class FakeWeatherParameters:
    def __init__(self, **values):
        self.instance = FakeWeather(**values)
        for key, value in vars(self.instance).items():
            setattr(self, key, value)


# Static class-level values mirror CARLA's WeatherParameters preset API.
FakeWeatherParameters.ClearNoon = FakeWeather(
    cloudiness=5, precipitation=0, precipitation_deposits=0, wetness=0,
    sun_altitude_angle=45, fog_density=2, fog_distance=0.75
)
FakeWeatherParameters.WetNoon = FakeWeather(
    cloudiness=5, precipitation=0, precipitation_deposits=50, wetness=0,
    sun_altitude_angle=45, fog_density=3, fog_distance=0.75
)


class FakeCarla:
    WeatherParameters = FakeWeatherParameters


class FakeWorld:
    def __init__(self, weather=None):
        self.weather = weather or FakeWeather(cloudiness=75, precipitation=0)
        self.set_calls = 0
        self.tick_calls = 0

    def set_weather(self, weather):
        self.set_calls += 1
        self.weather = weather

    def get_weather(self):
        return self.weather

    def tick(self):
        self.tick_calls += 1


def test_preset_is_resolved_from_weather_parameters_and_copied():
    preset = resolve_weather_preset(FakeCarla, "WetNoon")

    assert preset is not FakeWeatherParameters.WetNoon
    assert preset.cloudiness == 5
    assert preset.precipitation_deposits == 50
    assert preset.sun_altitude_angle == 45


def test_default_keeps_the_current_map_weather_unchanged():
    assert resolve_weather_preset(FakeCarla, "Default") is None


def test_unknown_preset_reports_weather_parameters_choices():
    try:
        resolve_weather_preset(FakeCarla, "RainyEvening")
    except ValueError as exc:
        assert "WeatherParameters exposes" in str(exc)
        assert "ClearNoon" in str(exc)
    else:
        raise AssertionError("Unknown weather preset should be rejected")


def test_apply_weather_preset_sets_and_verifies_the_simulator_weather():
    world = FakeWorld()
    applied = apply_weather_preset(world, FakeCarla, "WetNoon", advance=world.tick)

    assert world.set_calls == 1
    assert world.tick_calls == 1
    assert applied is world.weather
    assert applied.precipitation_deposits == 50


def test_weather_verification_detects_when_simulator_keeps_old_weather():
    expected = resolve_weather_preset(FakeCarla, "WetNoon")
    actual = FakeWeather(cloudiness=70, precipitation=0, precipitation_deposits=0,
                         wetness=0, sun_altitude_angle=90, fog_density=0, fog_distance=0)

    mismatches = weather_mismatches(actual, expected)

    assert "cloudiness" in mismatches
    assert "sun_altitude_angle" in mismatches


def test_apply_weather_preset_fails_instead_of_silently_keeping_old_weather():
    class IgnoringWorld(FakeWorld):
        def set_weather(self, weather):
            self.set_calls += 1

    world = IgnoringWorld()
    try:
        apply_weather_preset(world, FakeCarla, "WetNoon")
    except RuntimeError as exc:
        assert "does not match the selected preset" in str(exc)
    else:
        raise AssertionError("A weather mismatch must not pass silently")


def test_weather_summary_contains_applied_visual_parameters():
    summary = format_weather_summary(resolve_weather_preset(FakeCarla, "WetNoon"))

    assert "precipitation_deposits=50" in summary
    assert "sun_altitude_angle=45" in summary
