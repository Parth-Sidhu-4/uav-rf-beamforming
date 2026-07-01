import numpy as np
import matplotlib.pyplot as plt
from mavlink_gen import MAVLinkGenerator
from ew_channel import EWChannelBridge, ChannelSnapshot
from mavlink_rx import MAVLinkReceiver, RxStats

def run_sweep(
    sinr_trajectory: list[float],
    p_out_trajectory: list[float],
    sigma_theta_trajectory: list[float],
    pkts_per_step: int = 20,
    seed: int = 42,
) -> tuple[RxStats, list[dict]]:
    """
    Drop-in replacement for the abstract P_out penalty in simulator_core.py.
    Pass in the same trajectory arrays your existing loop already produces.
    """
    rng   = np.random.default_rng(seed)
    gen   = MAVLinkGenerator()
    chan  = EWChannelBridge(seed=seed)
    rx    = MAVLinkReceiver()
    stats = RxStats()
    per_step_log = []

    # Let's get the packet byte size of a typical message to fix the analytical model.
    # The heartbeat is 17 bytes in v20 usually, attitude is 38 bytes, gps is 40 bytes.
    # We will compute the empirical average size for analytical per calculations.
    # For now, let's keep 28 bytes as a rough baseline, or just let the model match it dynamically if we wanted.
    
    for sinr_db, p_out, sigma_theta in zip(
            sinr_trajectory, p_out_trajectory, sigma_theta_trajectory):

        snap = ChannelSnapshot(sinr_db, p_out, sigma_theta)
        step_stats = RxStats()

        # Adaptive packet count: more packets where PER is low to resolve finite-sample floor
        pkts = 1000 if sinr_db > 12 else pkts_per_step
        burst_pkts = gen.burst(pkts, rng)
        for pkt in burst_pkts:
            rx.receive(chan.transmit(pkt, snap), step_stats)
            
        # Compute avg pkt length for this burst for accurate analytical PER
        avg_len = np.mean([len(p) for p in burst_pkts]) if burst_pkts else 28

        per_step_log.append({
            'sinr_db': sinr_db,
            'p_out': p_out,
            'sigma_theta': sigma_theta,
            'app_per': step_stats.application_per,
            'analytical_per': chan.expected_per(snap, pkt_bytes=int(avg_len))['per_combined'],
            'n_pkts': pkts,
        })

        # accumulate into global stats
        stats.sent        += step_stats.sent
        stats.hard_dropped += step_stats.hard_dropped
        stats.crc_failed  += step_stats.crc_failed
        stats.decoded     += step_stats.decoded

    return stats, per_step_log


if __name__ == "__main__":
    # Standalone demo: SINR sweep from -5 dB → +20 dB
    sinr_vals = np.linspace(-5, 20, 60).tolist()
    p_out_vals = np.clip(0.5 - np.array(sinr_vals) / 40, 0, 1).tolist()
    sigma_vals = [0.1] * 60

    stats, log = run_sweep(sinr_vals, p_out_vals, sigma_vals, pkts_per_step=200)

    print(f"Sent:            {stats.sent}")
    print(f"Hard Dropped:    {stats.hard_dropped}")
    print(f"CRC Failed:      {stats.crc_failed}")
    print(f"Decoded:         {stats.decoded}")
    print(f"App-Layer PER:   {stats.application_per:.4f}")
    print(f"Goodput:         {stats.goodput_fraction:.1%}")

    # Monte Carlo vs. analytical PER validation plot
    mc_per  = np.array([s['app_per']          for s in log])
    ana_per = np.array([s['analytical_per']   for s in log])
    sinrs   = np.array([s['sinr_db']          for s in log])
    n_pkts  = np.array([s['n_pkts']           for s in log])

    # Wilson interval approximation for binomial proportion
    ci = 1.96 * np.sqrt(mc_per * (1 - mc_per) / n_pkts)

    plt.figure(figsize=(8, 4))
    plt.semilogy(sinrs, mc_per,  'o', ms=4, label='Monte Carlo (MAVLink CRC)')
    plt.fill_between(sinrs, mc_per - ci, mc_per + ci, alpha=0.2, color='blue', label='95% Confidence Interval')
    plt.semilogy(sinrs, ana_per, '-',       label='Analytical PER')
    plt.xlabel('SINR post-LCMV (dB)')
    plt.ylabel('Packet Error Rate')
    plt.title('Phase D: Application-Layer PER vs. Post-Beamforming SINR')
    plt.legend()
    plt.grid(True, which='both')
    plt.tight_layout()
    plt.savefig('phase_d_per_validation.png', dpi=150)
    print("Saved phase_d_per_validation.png")
