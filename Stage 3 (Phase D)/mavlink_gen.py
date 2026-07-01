import time
import numpy as np
try:
    from pymavlink.dialects.v20 import ardupilotmega as mavlink2
except ImportError:
    from pymavlink.dialects.v10 import ardupilotmega as mavlink2

class MAVLinkGenerator:
    """
    Stateless MAVLink v2 packet factory.
    No socket, no SITL, no ArduPilot required —
    pack() returns raw bytes directly.
    """

    def __init__(self, sys_id: int = 1, comp_id: int = 1):
        self._mav = mavlink2.MAVLink(None)
        self._mav.srcSystem = sys_id
        self._mav.srcComponent = comp_id

    def heartbeat(self) -> bytes:
        return self._mav.heartbeat_encode(
            type=mavlink2.MAV_TYPE_QUADROTOR,
            autopilot=mavlink2.MAV_AUTOPILOT_ARDUPILOTMEGA,
            base_mode=209,
            custom_mode=0,
            system_status=mavlink2.MAV_STATE_ACTIVE,
            mavlink_version=3,
        ).pack(self._mav)

    def attitude(self, roll: float, pitch: float, yaw: float) -> bytes:
        return self._mav.attitude_encode(
            time_boot_ms=int(time.monotonic() * 1000) & 0xFFFFFFFF,
            roll=roll, pitch=pitch, yaw=yaw,
            rollspeed=0.0, pitchspeed=0.0, yawspeed=0.0,
        ).pack(self._mav)

    def gps_raw(self, lat_deg: float, lon_deg: float, alt_m: float) -> bytes:
        return self._mav.gps_raw_int_encode(
            time_usec=int(time.time() * 1e6) & 0xFFFFFFFFFFFFFFFF,
            fix_type=3,
            lat=int(lat_deg * 1e7),
            lon=int(lon_deg * 1e7),
            alt=int(alt_m * 1e3),
            eph=120, epv=200, vel=500, cog=0,
            satellites_visible=8,
        ).pack(self._mav)

    def burst(self, n: int, rng: np.random.Generator) -> list[bytes]:
        """Mixed telemetry stream mimicking a real UAV downlink."""
        generators = [
            self.heartbeat,
            lambda: self.attitude(*rng.normal(0, 0.1, 3)),
            lambda: self.gps_raw(28.6139, 77.2090, 120.0 + rng.normal(0, 0.5)),
        ]
        return [generators[i % len(generators)]() for i in range(n)]
