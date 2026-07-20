"""
materials.py -- Si, 4H-SiC and GaN parameter sets for the reverse-recovery model.

Extension of the Benda & Spenke (silicon) reverse-recovery reproduction to
wide-bandgap power semiconductors.  Each material supplies the four quantities
the ambipolar/space-charge model needs, plus the critical (avalanche) field
used only to judge whether the reverse-recovery voltage would trigger dynamic
avalanche.  Values are room-temperature, lightly-doped literature figures --
cited below in the module docstring, in the same defensible style as the SiC/GaN
entries in the sibling Merchant reproduction.

  mu_n, mu_p : electron/hole drift mobility          (cm^2/V.s)
  eps_r      : relative permittivity
  Ec         : critical field for avalanche          (V/cm)
  tau        : a representative high-injection lifetime (s) -- NB device- and
               quality-dependent; used only for the ABSOLUTE-time/voltage plots.
               The normalised results (t/tau, n/nbar) do NOT depend on tau.

Sources (representative, room temperature):
  Si     mu_n=1350, mu_p=450    Sze & Ng, "Physics of Semiconductor Devices" 3e.
         eps_r=11.9,  Ec~3.0e5 V/cm.  tau ~ 10 us (float-zone power diode).
  4H-SiC mu_n=950,  mu_p=120    Roschke & Schwierz, IEEE TED 48, 1442 (2001);
         Ayalew, PhD TU Wien (2004).  eps_r=9.7,  Ec~2.5e6 V/cm.  tau ~ 1 us
         (best-in-class 4H-SiC PiN, Kimoto & Cooper).
  GaN    mu_n=1000, mu_p=30     electron: Farahmand et al., IEEE TED 48, 535
         (2001); hole mobility is notoriously low/uncertain (bulk ~20-40, up to
         ~200 in the cleanest material) -- 30 taken as a representative p-GaN
         value; see note.  eps_r=9.0,  Ec~3.3e6 V/cm.  tau ~ 1 ns (GaN minority
         lifetime is short; vertical GaN PiN ~ns-tens of ns).

GaN caveat: its hole mobility (and hence its mobility asymmetry B) is the most
uncertain number here.  With mu_p in [30, 200], B ranges from ~0.94 down to
~0.67.  The comparison therefore also prints B's sensitivity to mu_p for GaN.
"""
from __future__ import annotations
from dataclasses import dataclass
from benda_spenke_model import Device


@dataclass(frozen=True)
class Material:
    name: str
    mu_n: float      # cm^2/V.s
    mu_p: float
    eps_r: float
    Ec: float        # V/cm  (critical/avalanche field)
    tau: float       # s     (representative high-injection lifetime)
    color: str

    @property
    def B(self):
        return (self.mu_n - self.mu_p) / (self.mu_n + self.mu_p)

    def device(self, d_over_L=1.0, iF=100.0):
        """Build a Benda-Spenke Device for this material."""
        return Device(d_over_L=d_over_L, mu_n=self.mu_n, mu_p=self.mu_p,
                      tau=self.tau, iF=iF, eps_r=self.eps_r)


SI = Material("Si", 1350.0, 450.0, 11.9, 3.0e5, 1.0e-5, "tab:blue")
SIC = Material("4H-SiC", 950.0, 120.0, 9.7, 2.5e6, 1.0e-6, "tab:green")
GAN = Material("GaN", 1000.0, 30.0, 9.0, 3.3e6, 1.0e-9, "tab:red")

MATERIALS = [SI, SIC, GAN]


if __name__ == "__main__":
    print(f"{'material':<8}{'mu_n':>7}{'mu_p':>7}{'b=mu_n/mu_p':>13}"
          f"{'B':>7}{'eps_r':>7}{'Ec[MV/cm]':>11}{'tau':>10}")
    for m in MATERIALS:
        print(f"{m.name:<8}{m.mu_n:>7.0f}{m.mu_p:>7.0f}{m.mu_n/m.mu_p:>13.1f}"
              f"{m.B:>7.3f}{m.eps_r:>7.1f}{m.Ec/1e6:>11.2f}{m.tau:>10.1e}")
    print("\nGaN B sensitivity to hole mobility:")
    for mup in (30, 60, 120, 200):
        B = (1000 - mup) / (1000 + mup)
        print(f"  mu_p={mup:4d} -> b={1000/mup:5.1f}, B={B:.3f}")
