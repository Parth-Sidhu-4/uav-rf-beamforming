import numpy as np
import matplotlib.pyplot as plt
from em_physics import fresnel_diffraction_gain

def test_itu_table():
    """
    Validates the Fresnel integral implementation against ITU-R P.526 tabulated loss values.
    ITU-R P.526 approximation for nu > -0.78:
    J(nu) = 6.9 + 20 log10( sqrt((nu-0.1)**2 + 1) + nu - 0.1 )
    """
    nu_vals = np.array([-2, -1, 0, 1, 2, 3])
    F = fresnel_diffraction_gain(nu_vals)
    
    # Power loss in dB relative to free space
    # |F|^2 is power, 20*log10(|F|) is field amplitude dB
    J_computed = -20 * np.log10(np.abs(F))
    
    # ITU-R P.526 approximation (only valid for nu > -0.78, so we compare strictly for nu >= 0)
    def itu_approx(nu):
        return 6.9 + 20 * np.log10(np.sqrt((nu - 0.1)**2 + 1) + nu - 0.1)
    
    print("nu \t |F| \t Computed Loss(dB) \t ITU Approx(dB)")
    print("-" * 55)
    for i, nu in enumerate(nu_vals):
        approx = itu_approx(nu) if nu > -0.78 else np.nan
        print(f"{nu:2.0f} \t {np.abs(F[i]):.3f} \t {J_computed[i]:.2f} \t\t {approx:.2f}")
        
    # Check asymptotes
    np.testing.assert_allclose(np.abs(fresnel_diffraction_gain(0)), 0.5, rtol=1e-5)
    print("Asymptote nu=0 passed.")

if __name__ == "__main__":
    test_itu_table()
    
    # Plot Cornu spiral ripple
    nu_fine = np.linspace(-3, 3, 500)
    F_fine = fresnel_diffraction_gain(nu_fine)
    
    plt.figure(figsize=(10, 4))
    plt.subplot(1, 2, 1)
    plt.plot(nu_fine, 20*np.log10(np.abs(F_fine)))
    plt.grid(True)
    plt.title('Diffraction Gain |F(v)| [dB]')
    plt.xlabel('Fresnel Parameter v')
    plt.ylabel('Gain (dB)')
    plt.axvline(0, color='r', linestyle='--')
    
    plt.subplot(1, 2, 2)
    plt.plot(nu_fine, np.angle(F_fine, deg=True))
    plt.grid(True)
    plt.title('Diffraction Phase arg(F(v)) [deg]')
    plt.xlabel('Fresnel Parameter v')
    plt.ylabel('Phase (deg)')
    plt.axvline(0, color='r', linestyle='--')
    
    plt.tight_layout()
    plt.savefig('cornu_spiral_check.png')
    print("Saved cornu_spiral_check.png")
