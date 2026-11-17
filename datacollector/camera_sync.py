"""Frame-ID synchronization for asynchronous CARLA camera sensors."""
from collections import OrderedDict


class CameraFrameSynchronizer:
    """Pair delayed sensor packets with the exact CARLA world snapshots."""

    def __init__(self, snapshot_limit=128, packet_limit=64):
        self.snapshot_limit = int(snapshot_limit)
        self.packet_limit = int(packet_limit)
        self.snapshots = OrderedDict()
        self.pending_packets = {}
        self.last_processed_frame = -1

    def add_snapshot(self, snapshot):
        frame_number = int(snapshot.frame)
        self.snapshots[frame_number] = snapshot
        self.snapshots.move_to_end(frame_number)
        while len(self.snapshots) > self.snapshot_limit:
            self.snapshots.popitem(last=False)

    def add_packet(self, packet):
        frame_number = int(packet["frame"])
        if frame_number > self.last_processed_frame:
            self.pending_packets[frame_number] = packet

    def drain_ready(self):
        ready = []
        oldest_snapshot = next(iter(self.snapshots), None)
        for frame_number in sorted(tuple(self.pending_packets)):
            packet = self.pending_packets[frame_number]
            snapshot = self.snapshots.get(frame_number)
            if snapshot is not None:
                ready.append((packet, snapshot))
                self.pending_packets.pop(frame_number, None)
                self.last_processed_frame = max(self.last_processed_frame, frame_number)
            elif oldest_snapshot is not None and frame_number < oldest_snapshot:
                self.pending_packets.pop(frame_number, None)
        while len(self.pending_packets) > self.packet_limit:
            oldest_frame = min(self.pending_packets)
            self.pending_packets.pop(oldest_frame, None)
        return ready

    def reset(self):
        self.snapshots.clear()
        self.pending_packets.clear()
        self.last_processed_frame = -1
