"""Defaults copied from CARLA_Autonomous_driving/config_city.py's active RGB camera."""
from dataclasses import dataclass, asdict


# Fallback list for CARLA 1.0. The collector replaces this list with the exact
# maps reported by the connected server via Client.get_available_maps().
TOWNS = ("Town01_Opt", "Town02_Opt", "Town03_Opt", "Town04_Opt", "Town05_Opt", "Town10HD_Opt")
CAMERA_REFERENCE = {
    "width": 640, "height": 384, "fov": 120.0,
    "x": 1.4, "y": 0.0, "z": 2.2,
    "pitch": -25.0, "yaw": 0.0, "roll": 0.0,
    "sensor_tick": 0.03,
}


@dataclass
class CollectionConfig:
    host: str = "127.0.0.1"
    port: int = 2000
    timeout: float = 20.0
    towns: dict | None = None
    capture_interval: float = 1.0
    images_per_spawn: int = 30
    vehicle_blueprint: str = "vehicle.tesla.model3"
    tm_speed_difference: float = 10.0
    background_traffic: bool = False
    background_vehicle_count: int = 20
    pedestrians: bool = False
    weather: str = "Default"
    output_directory: str = "datasets"
    camera: dict | None = None

    def __post_init__(self):
        if self.towns is None:
            self.towns = {town: {"enabled": True, "target": 700} for town in TOWNS}
        if self.camera is None:
            self.camera = dict(CAMERA_REFERENCE)

    def to_dict(self):
        return asdict(self)
