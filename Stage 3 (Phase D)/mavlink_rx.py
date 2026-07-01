from dataclasses import dataclass
from typing import Optional
try:
    from pymavlink.dialects.v20 import ardupilotmega as mavlink2
except ImportError:
    from pymavlink.dialects.v10 import ardupilotmega as mavlink2

@dataclass
class RxStats:
    sent: int = 0
    hard_dropped: int = 0
    crc_failed: int = 0
    decoded: int = 0

    @property
    def application_per(self) -> float:
        return 0.0 if self.sent == 0 else (self.sent - self.decoded) / self.sent

    @property
    def goodput_fraction(self) -> float:
        return 1.0 - self.application_per


class MAVLinkReceiver:
    """
    Stateful MAVLink v2 byte-stream parser.
    robust_parsing=True: CRC failures return None, not exceptions.
    """

    def __init__(self):
        self._parser = mavlink2.MAVLink(None)
        self._parser.robust_parsing = True

    def receive(self, raw: Optional[bytes], stats: RxStats) -> list:
        """
        raw=None  → hard drop (Stage 1 outage)
        raw=bytes → feed byte-by-byte into MAVLink parser
        Returns list of successfully decoded MAVLink message objects.
        """
        stats.sent += 1

        if raw is None:
            stats.hard_dropped += 1
            return []

        decoded = []
        try:
            msgs = self._parser.parse_buffer(raw)
            if msgs:
                for msg in msgs:
                    if msg is not None and msg.get_type() != 'BAD_DATA':
                        decoded.append(msg)
        except Exception:
            pass

        if not decoded:
            stats.crc_failed += 1
        else:
            stats.decoded += len(decoded)

        return decoded
