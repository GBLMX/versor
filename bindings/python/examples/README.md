# Black-hole Blender example

`blackhole_3d.py` builds a 3D accretion disk in Blender, integrates particle
orbits, and bakes the result into shape keys.  The camera and disk are oriented
close to the reference PNG (edge-on disk, photon ring, star field).

```bash
blender --background --python blackhole_3d.py -- \
    --out /tmp/black_hole_3d.blend \
    --render /tmp/black_hole_3d.png \
    --frames 120 --samples 32 --width 800
```

## Units and exact Schwarzschild scales

The script uses geometrized units with `r_s = 1`, hence `M = r_s/2 = 0.5`.
For a non-rotating black hole:

```text
event horizon                 r_s = 2GM/c^2 = 1
photon sphere                 r_ph = 3GM/c^2 = 1.5
ISCO                          r_ISCO = 6GM/c^2 = 3
critical impact parameter     b_c = 3*sqrt(3) GM/c^2 = 2.598
```

The photon ring in the scene is drawn at `b_c`; the disk inner edge is at
`r_ISCO`; `blackhole_3d.py` exposes `PHOTON_SPHERE`, `ISCO`, and
`CRITICAL_IMPACT_PARAMETER` for these values.

## Accretion-disk dynamics

The particle integrator uses the Paczyński–Wiita pseudo-Newtonian potential

```text
Phi(r) = -GM / (r - r_G),          r_G = 2GM/c^2 = r_s
a(r)   = -GM / (r - r_G)^2 * r_hat
```

which reproduces the Schwarzschild ISCO and marginally bound orbit exactly.
The circular speed used for the initial conditions follows from the same
potential:

```text
v_c(r) = sqrt(GM r) / (r - r_G)
Omega_PW(r) = sqrt(GM/r^3) * r/(r - r_G)
```

The Schwarzschild coordinate angular velocity is
`Omega_SCH = sqrt(GM/r^3)`; the two differ at small radii.  The code uses the
PW form because it is cheap, stable with a velocity-Verlet step, and is the
standard pseudo-Newtonian model for this kind of visualization.

The disk colour follows the Shakura–Sunyaev zero-torque thin-disk temperature
proxy

```text
T(r) = T0 * x^(-3/4) * (1 - sqrt(1/x))^(1/4),   x = r/r_in
F(r) ∝ x^(-3) * (1 - sqrt(1/x))
```

with `r_in = r_ISCO`.  The strict zero-torque profile vanishes at the inner
edge; the renderer adds a small brightness floor so the particle ring remains
visible.

## Photon geodesics / lensing reference

For photons in Schwarzschild spacetime, the exact null Binet equation is

```text
d^2u/dphi^2 + u = (3GM/c^2) u^2,        u = 1/r
```

with critical impact parameter `b_c = 3*sqrt(3) GM/c^2`.  The photon sphere is
the unstable circular orbit at `r = 3GM/c^2`.  The implementation in this
example does **not** ray-trace the background: the photon ring is an emissive
torus at `b_c`, and the upper/lower arcs are geometric stand-ins for the
secondary lensed disk images.  For a true lensed image, integrate the Binet
equation per camera ray and sample the sky map.

## Geometric algebra

The initial tangential directions are generated through the compiled binding:

```text
R = exp(B/2),  B a spacelike bivector
v' = R v R~
```

`vsr_runtime.e3_rotate_many(radial, e3_rotor(0, 1, 0, pi/2))` rotates the
radial vectors `(x, 0, z)` into `(-z, 0, x)` for the prograde disk.  The
NumPy fallback performs the same cross-product operation when the binding is
not built.

The underlying geometric-algebra formulation of general relativity uses the
spacetime algebra with `gamma_0^2 = 1`, `gamma_i^2 = -1`, the rotor
`R = exp(B/2)`, and the geodesic equation

```text
t · ∇ t = 0
```

where `∇` is the covariant derivative with connection bivector `Omega(a)`.
The Schwarzschild line element in the GA frame formulation is

```text
dτ^2 = (1 - 2M/r) dt^2
     + 2(f1 g2 - f2 g1) dt dr
     - (f1^2 - f2^2) dr^2
     - r^2 dΩ^2
```

with the gauge `f1 g1 - f2 g2 = 1`; choosing
`f1 g2 - f2 g1 = 0` gives the ordinary Schwarzschild metric.

## References

- M. A. Abramowicz, *The Paczyński–Wiita potential: A step-by-step
  “derivation”*, arXiv:0904.0913, A&A 500 (2009).
- M. A. Abramowicz & P. C. Fragile, *Foundations of Black Hole Accretion Disk
  Theory*, Living Reviews in Relativity 16 (2013), arXiv:1104.5499.
- N. I. Shakura & R. A. Sunyaev, *Black holes in binary systems. Observational
  appearance*, A&A 24 (1973) 337.
- I. D. Novikov & K. S. Thorne, *Astrophysics of Black Holes*, 1973.
- D. Tong, *General Relativity*, Cambridge lecture notes,
  <https://www.damtp.cam.ac.uk/user/tong/gr/> (Schwarzschild geodesics,
  photon sphere, null effective potential).
- M. R. Francis & A. Kosowsky, *Geometric Algebra Techniques for General
  Relativity*, arXiv:gr-qc/0311007.
- A. N. Lasenby, *Geometric Algebra, Gravity and Gravitational Waves*,
  arXiv:1912.05960.
- S. Setiawan, *Applications of geometric algebra to black holes and Hawking
  radiation*, arXiv:physics/0412070.
