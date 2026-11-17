"""CARLA work runs here, away from the Qt GUI thread."""
import json
import queue
import random
import threading
import time
import traceback

import numpy as np
from PySide6.QtCore import QThread, Signal

try:
    from .storage import DatasetStore
except ImportError:
    from storage import DatasetStore


class CarlaWorker(QThread):
    status = Signal(str)
    log = Signal(str)
    progress = Signal(dict)
    preview = Signal(object)
    finished_result = Signal(str)
    maps_available = Signal(list)

    def __init__(self, config, connect_only=False):
        super().__init__()
        self.config = config
        self.connect_only = connect_only
        self._pause = threading.Event()
        self._stop = threading.Event()
        self.client = self.world = self.vehicle = self.camera = None
        self.tm = None
        self._map_refs = {}
        self._images = queue.Queue(maxsize=8)
        self._owned_vehicles = []
        self._owned_walkers = []
        self._saved_world_settings = None
        self._old_world = None

    def pause(self): self._pause.set()
    def resume(self): self._pause.clear()
    def stop(self): self._stop.set(); self._pause.clear()

    def _say(self, text):
        self.status.emit(text)

    def _connect(self):
        import carla
        self._say(f"Connecting to CARLA at {self.config['host']}:{self.config['port']}…")
        self.client = carla.Client(self.config["host"], int(self.config["port"]))
        self.client.set_timeout(float(self.config["timeout"]))
        self.world = self.client.get_world()
        current_town = self.world.get_map().name.split("/")[-1]
        self._say(f"Connected · current town: {current_town}")
        get_maps = getattr(self.client, "get_available_maps", None)
        if callable(get_maps):
            try:
                available = list(get_maps())
                for map_reference in available:
                    name = str(map_reference).rstrip("/").split("/")[-1]
                    if name:
                        self._map_refs[name] = str(map_reference)
                if self._map_refs:
                    self.maps_available.emit(sorted(self._map_refs))
                    self.log.emit(
                        f"CARLA server reports {len(self._map_refs)} installed maps."
                    )
            except Exception as exc:
                self.log.emit(f"Could not enumerate installed CARLA maps: {exc}")
        if not self._map_refs:
            self._map_refs[current_town] = current_town
            self.maps_available.emit([current_town])

    def run(self):
        result = "Stopped"
        try:
            self._connect()
            if self.connect_only:
                result = "Connected"
            else:
                self._collect()
                result = "Stopped" if self._stop.is_set() else "Collection complete"
        except Exception as exc:
            self.log.emit(traceback.format_exc())
            result = f"Error: {exc}"
            self.status.emit(result)
        finally:
            self._cleanup()
            self.finished_result.emit(result)

    def _collect(self):
        from datetime import datetime
        selected = [name for name, spec in self.config["towns"].items() if spec["enabled"] and int(spec["target"]) > 0]
        if not selected:
            raise ValueError("Select at least one town with a positive target")
        store = DatasetStore(self.config["output_directory"], self.config)
        counts = {town: store.existing_count(town) for town in selected}
        targets = {town: int(self.config["towns"][town]["target"]) for town in selected}
        self._store = store
        self._targets = targets
        self._counts = counts
        self._started = time.monotonic()
        self._capture_count = 0
        self._failed = 0
        self._say(f"Dataset: {store.path}")
        if self.config.get("background_traffic"):
            self.log.emit("Background vehicle spawning is enabled; traffic actors will be owned and cleaned up by this collector.")
        for town in selected:
            if self._stop.is_set(): break
            if counts[town] >= targets[town]:
                continue
            self._load_town(town)
            self._configure_world()
            self._configure_weather()
            self._spawn_traffic()
            spawn_points = list(self.world.get_map().get_spawn_points())
            random.shuffle(spawn_points)
            if not spawn_points:
                raise RuntimeError(f"No vehicle spawn points in {town}")
            used = set()
            self._next_capture = None
            last_timestamp = None
            while counts[town] < targets[town] and not self._stop.is_set():
                index = self._spawn_ego(spawn_points, used)
                if index is None:
                    raise RuntimeError(f"Could not spawn Ego at any available point in {town}")
                images_here = 0
                pause_announced = False
                while images_here < int(self.config["images_per_spawn"]) and counts[town] < targets[town] and not self._stop.is_set():
                    if self._pause.is_set():
                        if not pause_announced:
                            self._say("Paused · simulation is held")
                            pause_announced = True
                        while self._pause.is_set() and not self._stop.wait(0.1): pass
                        if self._stop.is_set(): break
                        self._say("Resumed")
                        pause_announced = False
                    frame_id = self.world.tick()
                    snapshot = self.world.get_snapshot()
                    timestamp = snapshot.timestamp.elapsed_seconds
                    frame = self._wait_image(frame_id, 3.0)
                    if frame is None:
                        self._failed += 1
                        self.log.emit(f"No camera frame for CARLA frame {snapshot.frame}")
                        continue
                    if self._next_capture is None:
                        self._next_capture = timestamp
                    if timestamp + 1e-6 < self._next_capture:
                        continue
                    try:
                        rgb = frame["bgra"][:, :, :3][:, :, ::-1].copy()
                        vehicle_tf = self.vehicle.get_transform()
                        camera = self.config["camera"]
                        pose = vehicle_tf
                        metadata = {
                            "carla_frame": frame["frame"], "simulation_timestamp": timestamp,
                            "simulation_delta_since_capture": None if last_timestamp is None else timestamp-last_timestamp,
                            "ego_transform": {"location": {"x":pose.location.x,"y":pose.location.y,"z":pose.location.z}, "rotation": {"pitch":pose.rotation.pitch,"yaw":pose.rotation.yaw,"roll":pose.rotation.roll}},
                            "spawn_point_id": index, "camera_resolution": [camera["width"],camera["height"]],
                            "camera_fov": camera["fov"], "camera_transform": {k:camera[k] for k in ("x","y","z","pitch","yaw","roll")},
                            "weather": self.config["weather"], "vehicle_blueprint": self.config["vehicle_blueprint"],
                            "carla_map_reference": self._map_refs.get(town, town),
                            "traffic_manager": {"port": self.tm.get_port(), "speed_difference_percent": self.config["tm_speed_difference"]},
                            "town_target": targets[town], "town_count": counts[town]+1,
                        }
                        saved_path = store.save(town, rgb, metadata)
                        counts[town] += 1; images_here += 1; self._capture_count += 1
                        last_timestamp = timestamp
                        self.preview.emit(rgb)
                        self._emit_progress(town, index, timestamp)
                        if counts[town] % 25 == 0 or counts[town] == targets[town]:
                            self.log.emit(f"{town}: {counts[town]}/{targets[town]} saved ({saved_path})")
                    except Exception as exc:
                        self._failed += 1
                        self.log.emit(f"Image save failed: {exc}")
                    self._next_capture += float(self.config["capture_interval"])
                self._destroy_ego()
            self._destroy_traffic()
        self._say("All selected town targets reached" if not self._stop.is_set() else "Collection stopped")

    def _load_town(self, town):
        self._say(f"Loading {town} · CARLA will replace its current world")
        self._destroy_ego(); self._destroy_traffic(); self._restore_world()
        map_reference = self._map_refs.get(town, town)
        self.log.emit(f"CARLA map reference: {map_reference}")
        self.world = self.client.load_world(map_reference)
        self._old_world = self.world

    def _configure_world(self):
        settings = self.world.get_settings()
        # CARLA's boost::python WorldSettings wrapper cannot be copied or
        # pickled. Snapshot its scalar properties and restore them later onto
        # a fresh settings wrapper.
        self._saved_world_settings = {}
        for name in dir(settings):
            if name.startswith("_"):
                continue
            try:
                value = getattr(settings, name)
            except Exception:
                continue
            if isinstance(value, (bool, int, float, str, type(None))):
                self._saved_world_settings[name] = value
        settings.synchronous_mode = True
        settings.fixed_delta_seconds = 0.05
        settings.no_rendering_mode = False
        self.world.apply_settings(settings)
        self.tm = self.client.get_trafficmanager()
        self.tm.set_synchronous_mode(True)
        self.tm.set_random_device_seed(random.randrange(1, 2**30))

    def _configure_weather(self):
        preset = self.config.get("weather", "Default")
        if preset == "Default": return
        if not hasattr(__import__("carla"), preset):
            self.log.emit(f"Unknown weather preset {preset}; retaining map default")
            return
        weather = getattr(__import__("carla"), preset)
        self.world.set_weather(weather)

    def _spawn_traffic(self):
        if not self.config.get("background_traffic"): return
        import carla
        library = self.world.get_blueprint_library().filter("vehicle.*")
        points = list(self.world.get_map().get_spawn_points()); random.shuffle(points)
        for point in points[:int(self.config.get("background_vehicle_count",20))]:
            bp = random.choice(library)
            if bp.has_attribute("role_name"): bp.set_attribute("role_name", "autopilot")
            actor = self.world.try_spawn_actor(bp, point)
            if actor:
                actor.set_autopilot(True, self.tm.get_port()); self._owned_vehicles.append(actor)

    def _spawn_ego(self, points, used):
        import carla
        self._say("Spawning Ego vehicle…")
        blueprint = self.world.get_blueprint_library().find(self.config["vehicle_blueprint"])
        if blueprint.has_attribute("role_name"): blueprint.set_attribute("role_name", "hero")
        order = [i for i in range(len(points)) if i not in used] + [i for i in range(len(points)) if i in used]
        for i in order:
            actor = self.world.try_spawn_actor(blueprint, points[i])
            if actor is None: continue
            self.vehicle = actor; used.add(i)
            actor.set_autopilot(True, self.tm.get_port())
            self.tm.vehicle_percentage_speed_difference(actor, float(self.config["tm_speed_difference"]))
            camera_cfg = self.config["camera"]
            bp = self.world.get_blueprint_library().find("sensor.camera.rgb")
            for key, attr in (("width","image_size_x"),("height","image_size_y"),("fov","fov"),("sensor_tick","sensor_tick")):
                bp.set_attribute(attr, str(camera_cfg[key]))
            tf = carla.Transform(carla.Location(x=float(camera_cfg["x"]),y=float(camera_cfg["y"]),z=float(camera_cfg["z"])),carla.Rotation(pitch=float(camera_cfg["pitch"]),yaw=float(camera_cfg["yaw"]),roll=float(camera_cfg["roll"])))
            self.camera = self.world.spawn_actor(bp, tf, attach_to=actor)
            self.camera.listen(self._camera_callback)
            self._say(f"Ego autopilot active · spawn point {i}")
            return i
        return None

    def _camera_callback(self, image):
        try:
            arr = np.frombuffer(image.raw_data, dtype=np.uint8).reshape((image.height,image.width,4)).copy()
            packet = {"frame": int(image.frame), "timestamp": float(image.timestamp), "bgra": arr}
            try: self._images.put_nowait(packet)
            except queue.Full:
                try: self._images.get_nowait()
                except queue.Empty: pass
                self._images.put_nowait(packet)
        except Exception as exc:
            self.log.emit(f"Camera callback error: {exc}")

    def _wait_image(self, frame, timeout):
        deadline=time.monotonic()+timeout
        while not self._stop.is_set() and time.monotonic()<deadline:
            try:
                packet=self._images.get(timeout=.05)
                if packet["frame"] == int(frame): return packet
            except queue.Empty: pass
        return None

    def _emit_progress(self, town, spawn, sim_time):
        self.progress.emit({"town":town,"counts":dict(self._counts),"targets":dict(self._targets),"spawn":spawn,"sim_time":sim_time,"captures":self._capture_count,"failed":self._failed,"elapsed":time.monotonic()-self._started,"rate":self._capture_count/max(.001,time.monotonic()-self._started)})

    def _destroy_ego(self):
        if self.vehicle:
            try:
                port = self.tm.get_port() if self.tm else None
                if port is None:
                    self.vehicle.set_autopilot(False)
                else:
                    self.vehicle.set_autopilot(False, port)
            except Exception as exc:
                self.log.emit(f"Could not detach Ego from Traffic Manager: {exc}")
        for actor in (self.camera,self.vehicle):
            if actor:
                try:
                    if actor is self.camera: actor.stop()
                    actor.destroy()
                except Exception as exc:
                    self.log.emit(f"Could not destroy Ego {'camera' if actor is self.camera else 'vehicle'}: {exc}")
        self.camera=self.vehicle=None
        while True:
            try: self._images.get_nowait()
            except queue.Empty: break

    def _destroy_traffic(self):
        actors = self._owned_vehicles + self._owned_walkers
        if not actors:
            return

        # Detach autopilot actors from Traffic Manager before destroying them.
        # Keeping TM active while a whole town is unloaded can leave it
        # processing actors from the old world during the next load_world().
        tm_port = None
        if self.tm:
            try:
                tm_port = self.tm.get_port()
                self.tm.set_synchronous_mode(False)
            except Exception as exc:
                self.log.emit(f"Traffic Manager shutdown warning: {exc}")

        for actor in self._owned_vehicles:
            try:
                actor.set_autopilot(False, tm_port) if tm_port is not None else actor.set_autopilot(False)
            except Exception as exc:
                self.log.emit(f"Could not detach traffic actor {getattr(actor, 'id', '?')}: {exc}")

        # Batch destruction is much faster and more reliable than issuing a
        # separate RPC for every vehicle, especially on large towns.
        pending = []
        try:
            import carla
            identified_actors = []
            for actor in actors:
                try:
                    identified_actors.append((actor, int(actor.id)))
                except Exception as exc:
                    self.log.emit(f"Could not read traffic actor id: {exc}")
            if identified_actors and self.client is not None:
                responses = self.client.apply_batch_sync(
                    [carla.command.DestroyActor(actor_id) for _, actor_id in identified_actors], False
                )
                pending = []
                for (actor, _), response in zip(identified_actors, responses):
                    error = getattr(response, "error", "")
                    if error:
                        self.log.emit(f"Batch cleanup failed for traffic actor {getattr(actor, 'id', '?')}: {error}")
                        pending.append(actor)
                pending.extend(actor for actor, _ in identified_actors[len(responses):])
                known = {id(actor) for actor, _ in identified_actors}
                pending.extend(actor for actor in actors if id(actor) not in known)
            else:
                pending = actors[:]
        except Exception as exc:
            self.log.emit(f"Batch traffic cleanup failed; trying individual cleanup: {exc}")
            pending = actors[:]

        # Retain an individual fallback for actors rejected by the batch (or
        # CARLA clients that do not support batch cleanup).
        for actor in pending:
            try:
                actor.destroy()
            except Exception as exc:
                self.log.emit(f"Could not destroy traffic actor {getattr(actor, 'id', '?')}: {exc}")

        self._owned_vehicles.clear()
        self._owned_walkers.clear()

    def _restore_world(self):
        if self.tm:
            try: self.tm.set_synchronous_mode(False)
            except Exception: pass
        if self.world and self._saved_world_settings is not None:
            try:
                settings = self.world.get_settings()
                for name, value in self._saved_world_settings.items():
                    try: setattr(settings, name, value)
                    except (AttributeError, RuntimeError, TypeError): pass
                self.world.apply_settings(settings)
            except Exception: pass
        self._saved_world_settings=None; self.tm=None

    def _cleanup(self):
        self._destroy_ego(); self._destroy_traffic(); self._restore_world()
