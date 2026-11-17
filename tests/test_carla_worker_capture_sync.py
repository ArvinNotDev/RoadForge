from collections import OrderedDict
import queue

from datacollector.worker import CarlaWorker


class FakeSnapshot:
    def __init__(self, frame):
        self.frame = frame


def make_worker():
    worker = CarlaWorker({})
    worker._images = queue.Queue(maxsize=64)
    worker._snapshot_history = OrderedDict()
    worker._pending_camera_packets = {}
    worker._last_processed_camera_frame = -1
    return worker


def packet(frame):
    return {"frame": frame, "timestamp": frame * 0.05, "bgra": None}


def test_delayed_camera_packet_is_buffered_until_its_snapshot_arrives():
    worker = make_worker()
    worker._images.put(packet(101))

    assert worker._drain_ready_images() == []
    assert 101 in worker._pending_camera_packets

    snapshot = FakeSnapshot(101)
    worker._snapshot_history[101] = snapshot
    ready = worker._drain_ready_images()

    assert len(ready) == 1
    assert ready[0][0]["frame"] == 101
    assert ready[0][1] is snapshot
    assert worker._drain_ready_images() == []


def test_camera_packets_are_returned_in_frame_order_and_only_once():
    worker = make_worker()
    first = FakeSnapshot(100)
    second = FakeSnapshot(101)
    worker._snapshot_history[100] = first
    worker._snapshot_history[101] = second
    worker._images.put(packet(101))
    worker._images.put(packet(100))

    ready = worker._drain_ready_images()

    assert [item[0]["frame"] for item in ready] == [100, 101]
    assert [item[1] for item in ready] == [first, second]
    assert worker._drain_ready_images() == []
