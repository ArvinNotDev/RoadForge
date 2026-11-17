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
    from .carla_compat import sticky_control_error_message, version_mismatch_message
    from .map_catalog import (
        build_map_reference_index, normalize_map_name, require_map_reference,
        select_collection_towns,
    )
    from .storage import DatasetStore
except ImportError:
    from carla_compat import sticky_control_error_message, version_mismatch_message
    from map_catalog import (
        build_map_reference_index, normalize_map_name, require_map_reference,
        select_collection_towns,
    )
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
        self._carla_client_version = None
        self._carla_server_version = None
        self._carla_module_path = None
        self._images = queue.Queue(maxsize=64)
        try:
            from .camera_sync import CameraFrameSynchronizer
        except ImportError:
            from camera_sync import CameraFrameSynchronizer
        self._camera_frames = CameraFrameSynchronizer()
        self._camera_no_data_since = None
        self._last_camera_wait_log = 0.0
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
        self._carla_module_path = getattr(carla, "__file__", None)
        try:
            self._carla_client_version = str(self.client.get_client_version())
            self._carla_server_version = str(self.client.get_server_version())
            self.log.emit(f"CARLA Python API version: {self._carla_client_version}")
            self.log.emit(f"CARLA Simulator version: {self._carla_server_version}")
            mismatch = version_mismatch_message(
                self._carla_client_version,
                self._carla_server_version,
                self._carla_module_path,
            )
            if mismatch:
                raise RuntimeError(mismatch)
        except AttributeError:
            self.log.emit(
                "This CARLA API does not expose version-check methods; "
                "compatibility will be checked when CARLA objects are accessed."
            )
        self.world = self.client.get_world()
        current_map_reference = str(self.world.get_map().name).strip()
        current_town = normalize_map_name(current_map_reference)
        self._say(f"Connected · current town: {current_town}")
        get_maps = getattr(self.client, "get_available_maps", None)
        if callable(get_maps):
            try:
                available = list(get_maps())
                self._map_refs = build_map_reference_index(available)
            except Exception as exc:
                self.maps_available.emit([])
                raise RuntimeError(
                    f"Could not enumerate installed CARLA maps: {exc}"
                ) from exc
            if not self._map_refs:
                self.maps_available.emit([])
                raise RuntimeError(
                    "CARLA returned no installed maps. Check the server's map packages and CARLA Python API."
                )
        else:
            # Older CARLA Python APIs may not expose get_available_maps().
            # In that case, the already loaded world is the only verified map.
            self._map_refs = build_map_reference_index([current_map_reference])
            if not self._map_refs:
                raise RuntimeError("CARLA could not provide a valid reference for its current map.")
            self.log.emit(
                "This CARLA API does not expose get_available_maps(); only the currently loaded map is available."
            )
        # The active world is authoritative even if a custom/package map was omitted
        # from get_available_maps() by the server build.
        if current_town and current_map_reference:
            self._map_refs.setdefault(current_town, current_map_reference)
        self.maps_available.emit(sorted(self._map_refs))
        self.log.emit(
            f"CARLA server reports {len(self._map_refs)} available map(s): "
            + ", ".join(sorted(self._map_refs))
        )

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
        use_current_world = bool(self.config.get("use_current_world", False))
        current_map_reference = (
            str(self.world.get_map().name).strip() if use_current_world else None
        )
        selected = select_collection_towns(
            self.config.get("towns", {}),
            self._map_refs,
            use_current_world=use_current_world,
            current_map_reference=current_map_reference,
        )
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
            if use_current_world:
                live_reference = str(self.world.get_map().name).strip()
                live_town = normalize_map_name(live_reference)
                if live_town != town:
                    raise RuntimeError(
                        f"CARLA's current world changed from '{town}' to '{live_town}' before collection started."
                    )
                self.log.emit(
                    f"Reusing current world from client.get_world(): {live_reference} · load_world() skipped"
                )
                self._say(f"Using current world · {town} · load_world() skipped")
            else:
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
                    snapshot = self._tick_world()
                    ready_frames = self._drain_ready_images()
                    if not ready_frames:
                        self._check_camera_delivery(snapshot.frame)
                        continue
                    self._camera_no_data_since = None
                    self._last_camera_wait_log = 0.0
                    for frame, frame_snapshot in ready_frames:
                        if (
                            images_here >= int(self.config["images_per_spawn"])
                            or counts[town] >= targets[town]
                            or self._stop.is_set()
                        ):
                            break
                        timestamp = float(frame["timestamp"])
                        if self._next_capture is None:
                            self._next_capture = timestamp
                        if timestamp + 1e-6 < self._next_capture:
                            continue
                        actor_snapshot = frame_snapshot.find(int(self.vehicle.id))
                        if actor_snapshot is None:
                            self.log.emit(
                                f"Camera frame {frame['frame']} has no matching Ego snapshot; skipping that frame."
                            )
                            continue
                        try:
                            rgb = frame["bgra"][:, :, :3][:, :, ::-1].copy()
                            pose = actor_snapshot.get_transform()
                            camera = self.config["camera"]
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
        map_reference = require_map_reference(town, self._map_refs)
        self._say(f"Loading {town} · CARLA will replace its current world")
        self._destroy_ego(); self._destroy_traffic(); self._restore_world()
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
        self.log.emit(
            "Synchronous capture enabled · RoadForge owns world.tick(). Close other clients/scripts "
            "that also tick this CARLA server while collection is running."
        )
        self.log.emit(f"Traffic Manager synchronous mode enabled on port {self.tm.get_port()}.")

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
        try:
            blueprint_library = self.world.get_blueprint_library()
            blueprint = blueprint_library.find(self.config["vehicle_blueprint"])
        except ValueError as exc:
            if "sticky_control" in str(exc):
                raise RuntimeError(sticky_control_error_message(
                    self._carla_client_version,
                    self._carla_server_version,
                    self._carla_module_path,
                )) from exc
            raise
        if blueprint.has_attribute("role_name"): blueprint.set_attribute("role_name", "hero")
        order = [i for i in range(len(points)) if i not in used] + [i for i in range(len(points)) if i in used]
        for i in order:
            actor = self.world.try_spawn_actor(blueprint, points[i])
            if actor is None: continue
            self.vehicle = actor; used.add(i)
            # CARLA 1.0 actor creation is reflected in snapshots on a subsequent
            # simulation frame. Do not attach controls/sensors before that frame.
            self._wait_for_actor_snapshot(actor, "Ego vehicle")
            actor.set_autopilot(True, self.tm.get_port())
            self.tm.vehicle_percentage_speed_difference(actor, float(self.config["tm_speed_difference"]))
            camera_cfg = self.config["camera"]
            bp = self.world.get_blueprint_library().find("sensor.camera.rgb")
            for key, attr in (("width","image_size_x"),("height","image_size_y"),("fov","fov"),("sensor_tick","sensor_tick")):
                bp.set_attribute(attr, str(camera_cfg[key]))
            tf = carla.Transform(carla.Location(x=float(camera_cfg["x"]),y=float(camera_cfg["y"]),z=float(camera_cfg["z"])),carla.Rotation(pitch=float(camera_cfg["pitch"]),yaw=float(camera_cfg["yaw"]),roll=float(camera_cfg["roll"])))
            self.camera = self.world.spawn_actor(bp, tf, attach_to=actor)
            self._wait_for_actor_snapshot(self.camera, "RGB camera")
            self._camera_frames.reset()
            self._camera_no_data_since = time.monotonic()
            self._last_camera_wait_log = 0.0
            self.camera.listen(self._camera_callback)
            self.log.emit(
                "RGB camera listener attached. Camera frames are buffered by CARLA frame ID; "
                "GPU rendering latency will not block the next world tick."
            )
            for warmup_index in range(3):
                warmup_snapshot = self._tick_world()
                self.log.emit(
                    f"RGB camera warm-up {warmup_index + 1}/3 · CARLA frame {warmup_snapshot.frame}"
                )
            self._say(f"Ego autopilot active · spawn point {i} · RGB camera ready")
            return i
        return None

    def _tick_world(self):
        """Advance one synchronous frame with a bounded wait and cache its snapshot."""
        if self.world is None:
            raise RuntimeError("Cannot tick CARLA because no world is connected.")
        tick_timeout = max(1.0, min(float(self.config.get("timeout", 20.0)), 10.0))
        try:
            self.world.tick(tick_timeout)
            snapshot = self.world.get_snapshot()
        except Exception as exc:
            raise RuntimeError(
                f"CARLA did not complete a synchronous world.tick() within {tick_timeout:.1f}s. "
                "RoadForge must be the only client advancing this server during collection. "
                "Close other Python clients, ScenarioRunner instances, or tools that call world.tick(), "
                "and make sure a second Traffic Manager is not running in synchronous mode. "
                f"Original CARLA error: {exc}"
            ) from exc
        if snapshot is None:
            raise RuntimeError("CARLA returned no WorldSnapshot after a successful world.tick().")
        frame_number = int(snapshot.frame)
        self._camera_frames.add_snapshot(snapshot)
        return snapshot

    def _wait_for_actor_snapshot(self, actor, label, max_ticks=5):
        """Wait until CARLA 1.0 publishes a newly spawned actor in a world snapshot."""
        actor_id = int(actor.id)
        for _ in range(max_ticks):
            snapshot = self._tick_world()
            if snapshot.find(actor_id) is not None:
                return snapshot
        raise RuntimeError(
            f"{label} (actor {actor_id}) was spawned, but did not appear in a CARLA WorldSnapshot "
            f"after {max_ticks} synchronous ticks."
        )

    def _drain_ready_images(self):
        """Return camera packets only when their exact frame snapshot is cached."""
        while True:
            try:
                packet = self._images.get_nowait()
            except queue.Empty:
                break
            self._camera_frames.add_packet(packet)

        ready = self._camera_frames.drain_ready()
        if ready:
            return ready
        # Report packets discarded because their matching snapshot aged out.
        return []

    def _check_camera_delivery(self, world_frame):
        """Keep ticking through normal GPU latency, but fail clearly if no image arrives."""
        now = time.monotonic()
        if self._camera_no_data_since is None:
            self._camera_no_data_since = now
        elapsed = now - self._camera_no_data_since
        if elapsed >= 30.0:
            raise RuntimeError(
                "The RGB camera has not delivered any frames for 30 seconds, even though CARLA world ticks "
                f"are advancing (latest world frame: {world_frame}). Check that the CARLA server is rendering "
                "camera sensors, that sensor.camera.rgb is available, and that the server has not run out of GPU memory."
            )
        if elapsed >= 3.0 and now - self._last_camera_wait_log >= 3.0:
            self.log.emit(
                f"Waiting for RGB camera output · {elapsed:.1f}s without a frame · world frame {world_frame}. "
                "The collector continues ticking so CARLA's delayed GPU frames can arrive."
            )
            self._last_camera_wait_log = now

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
        self._camera_frames.reset()
        self._camera_no_data_since = None
        self._last_camera_wait_log = 0.0

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
