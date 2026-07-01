import numpy as np
from scipy.special import erfc
from dataclasses import dataclass
from typing import Optional

@dataclass(frozen=True)
class ChannelSnapshot:
    """One time-step slice of your existing simulator_core.py output."""
    sinr_post_db: float   # Phase B LCMV output
    p_out: float          # Phase A Marcum-Q output
    sigma_theta: float    # Phase C UKF angular uncertainty


class EWChannelBridge:
    """
    Converts physics outputs into byte-level packet corruption.

    Stage 1 (p_out):       Bernoulli hard drop — models deep Rician fade / outage
    Stage 2 (sinr_post):   Bit-level BPSK BER corruption — models residual jamming
    """

    _SINR_FLOOR_DB = -20.0
    _SINR_CEIL_DB  =  30.0

    def __init__(self, seed: int = 42):
        self.rng = np.random.default_rng(seed)

    @staticmethod
    def sinr_to_ber(sinr_db: float) -> float:
        sinr_db = np.clip(sinr_db, EWChannelBridge._SINR_FLOOR_DB,
                                   EWChannelBridge._SINR_CEIL_DB)
        return float(0.5 * erfc(np.sqrt(10 ** (sinr_db / 10))))

    def transmit(self, pkt: bytes, snap: ChannelSnapshot) -> Optional[bytes]:
        """
        Returns corrupted bytes, or None on hard outage drop.
        Caller feeds the return value directly to MAVLinkReceiver.receive().
        """
        # Stage 1 — hard outage
        if self.rng.random() < snap.p_out:
            return None

        # Stage 2 — bit-level corruption
        ber = self.sinr_to_ber(snap.sinr_post_db)
        if ber < 1e-12:
            return pkt   # numerically clean channel

        arr = np.frombuffer(pkt, dtype=np.uint8).copy()
        bit_errors = self.rng.random(len(arr) * 8) < ber
        # Reshape to (n_bytes, 8) and pack each row into an error-mask byte
        masks = np.packbits(bit_errors.reshape(-1, 8), axis=1).flatten()
        return bytes(np.bitwise_xor(arr, masks))

    def expected_per(self, snap: ChannelSnapshot, pkt_bytes: int = 28) -> dict:
        """
        Analytical PER prediction for a given snapshot.
        Useful for validating Monte Carlo results against closed form.
        """
        ber = self.sinr_to_ber(snap.sinr_post_db)
        per_ber_only = 1.0 - (1.0 - ber) ** (pkt_bytes * 8)
        per_combined = snap.p_out + (1.0 - snap.p_out) * per_ber_only
        return {
            'ber': ber,
            'per_ber_only': per_ber_only,
            'per_combined': per_combined,
        }
