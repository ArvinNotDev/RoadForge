from datacollector.camera_sync import CameraFrameSynchronizer


class FakeSnapshot:
    def __init__(self, frame):
        self.frame = frame


def packet(frame):
    return {"frame": frame, "timestamp": frame * 0.05, "bgra": None}


def test_delayed_camera_packet_is_buffered_until_its_snapshot_arrives():
    sync = CameraFrameSynchronizer()
    sync.add_packet(packet(101))

    assert sync.drain_ready() == []
    assert 101 in sync.pending_packets

    snapshot = FakeSnapshot(101)
    sync.add_snapshot(snapshot)
    ready = sync.drain_ready()

    assert len(ready) == 1
    assert ready[0][0]["frame"] == 101
    assert ready[0][1] is snapshot
    assert sync.drain_ready() == []


def test_camera_packets_are_returned_in_frame_order_and_only_once():
    sync = CameraFrameSynchronizer()
    first = FakeSnapshot(100)
    second = FakeSnapshot(101)
    sync.add_snapshot(first)
    sync.add_snapshot(second)
    sync.add_packet(packet(101))
    sync.add_packet(packet(100))

    ready = sync.drain_ready()

    assert [item[0]["frame"] for item in ready] == [100, 101]
    assert [item[1] for item in ready] == [first, second]
    assert sync.drain_ready() == []


def test_snapshot_history_is_bounded():
    sync = CameraFrameSynchronizer(snapshot_limit=2)
    for frame in (10, 11, 12):
        sync.add_snapshot(FakeSnapshot(frame))

    assert list(sync.snapshots) == [11, 12]


def test_camera_packets_older_than_snapshot_history_are_discarded():
    sync = CameraFrameSynchronizer(snapshot_limit=2)
    for frame in (10, 11, 12):
        sync.add_snapshot(FakeSnapshot(frame))
    sync.add_packet(packet(9))

    assert sync.drain_ready() == []
    assert sync.pending_packets == {}
