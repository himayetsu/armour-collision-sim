# ArmorSim 3D

3D soft-body armor penetration simulation: spring–mass particles, material-based bond breaking, and collision response. Modular Qt application.

**Run:** `pip install -r requirements.txt` then `python armorsim_qt.py`

**Windows exe:** `build_release.bat` (needs Python 3.11+). Output is `dist\ArmorSim.exe`. Upload that file to a GitHub Release.

**Geometry:** Dimensions are entered in millimeters and stored in meters. Cylinder
rings reach the requested radius, including diameters smaller than the lattice
spacing. Flat plate lattices span their exact width, height, and thickness.
Select a plate in the Scene to set yaw, pitch, and roll in its **Orientation**
controls. In **Curvature**, choose whether to bend across the width or height
and set the total bend angle. The selected dimension is the arc length along the
middle of the plate. Use a bend angle of zero for a flat plate.

Use **3D Style** to switch between Solid exterior faces and Wireframe particles
and bonds. The three 2D panels in Quad View display particles for editing.
**Preview Burst** plays a visual effect at the selected body or camera target.
Dimensions are measured at particle centers.

Use **Detail** in the Simulation panel to change node density for inert bodies.
Medium uses the original sampling, Low is coarser, and High is halfway from
Medium to Ultra. Ultra has a nominal 10 mm spacing and retains finer sampling
where Medium already uses it. The nodes fit the shape's exact dimensions, so
actual gaps can vary. Changing detail pauses and resets the simulation. Finer
meshes can run more slowly and affect results by changing particle counts and
bonds. Exported layouts store the setting, and older layouts open at Medium.

**Geometry regression checks:** `python -m unittest discover -s tests -v`

---

## Technologies

- **Python 3.8+** – Core logic and numerics
- **NumPy** – Vectorized particle/spring arrays and linear algebra
- **PyQt5** – Main window, docks, controls, 2D ortho canvases (QPainter)
- **PyQtGraph (OpenGL)** – 3D viewport (`GLViewWidget`), scatter plots, line plots, custom ortho projection
- **OpenGL** – Via PyQtGraph / PyOpenGL for 3D rendering

---

## Physics Model

### Soft bodies

Bodies (penetrator, armor) are particle meshes. Each particle has position **x**, velocity **v**, mass **m** (from density × spacing³). Springs connect nearby particles (rest length **L₀**, stiffness **k**).

**Spring force (per spring):**

- **ε** = strain = (L − L₀) / L₀  
- **F** = k · ε · **ê**  (ê = unit vector along the spring)

**Stiffness** (from Young’s modulus *E* and spacing):

- k = E · (spacing² / spacing) · 1e−7  (penetrator)  
- Armor: same form with factor 0.6

Particles within `spacing × 1.8` are connected. Penetrator uses all pairs within that distance; armor uses a 3D grid so only 27-neighbor cells are checked (O(N) in particles).

### Bond breaking

- If |ε| > **breaking_strain** (material property), that spring is removed from the body (no force, not drawn).
- No explicit “fragment” objects: disconnected clusters simply stop sharing springs and are no longer damped if they were armor (see below).

### Compression limit (repulsion)

- **r** = L / L₀  
- If **r < compression_limit** (material property), an extra repulsive term is added so particles do not pass through each other:  
  **F** += (compression_limit − r) · k · 3 · **ê**

### Time integration

- Fixed base step with **4 substeps** per frame.
- Per substep: collision handling → spring forces (and bond removal) → **v** += **F**/m · dt → **x** += **v** · dt.
- Static (boundary) particles have **v** = 0 and are not updated.

### Damping

- Applied only to **armor** particles that still belong to the **connected component of the fixed boundary** (flood-fill from static particles along springs).
- Those particles: **v** *= 0.98 each substep.
- Detached fragments and the penetrator are not damped.

### Collisions

**Broad phase:** spatial hash with cell size ≈ 1.5 × collision distance; only particles in the same or neighboring cells are tested.

**Pair test:** two particles collide if (i) distance < collision distance, or (ii) relative velocity is large and they would reach that distance within the substep (predicted distance).

**Resolution:**

- Overlap **δ** = collision_distance − distance; **n** = unit vector between particles.
- Overlap is corrected by moving each particle along **n**: split by strength ratio  
  **ratio₁** = s₂/(s₁+s₂), **ratio₂** = s₁/(s₁+s₂), scaled by a contact stiffness factor from (s₁+s₂) and a correction factor (0.8 penetrator–armor, 0.5 same-body).

**Impulse (approaching pair, v_rel · n > 0):**

- Reduced mass μ = m₁m₂/(m₁+m₂).
- Restitution **e** from material **e₁,e₂**: base **e** = √(e₁e₂), reduced at high impact speed; strength ratio **s_ratio** = min(s₁,s₂)/max(s₁,s₂) adds loss term (1 − s_ratio)·0.4 (capped).
- Impulse magnitude: **J** = (1 + e)·μ·(v_rel·n), clamped; **Δv₁** = −**J n**/m₁, **Δv₂** = **J n**/m₂.

**Separating or slow contact:** repulsive force from overlap and stiffness (same-body uses average Young’s modulus and contact area; cross-body uses overlap-based force), converted to impulse and capped.

---

## Materials

`materials.py` contains nominal bulk properties and legacy solver settings.
The [material-property audit](docs/material_properties.md) records the sources,
units, grade/condition limitations, and values still requiring verification.

| Property | Role |
|----------|------|
| **density** | Nominal kg/m³; node mass = density × spacing³ |
| **youngs_modulus** | Nominal Pa; used with numerical spring/contact scaling |
| **strength** | Legacy contact coefficient in Pa; definitions differ between entries |
| **restitution**, **friction_coeff** | Legacy contact settings; actual values depend on the contacting pair and conditions |
| **hardness** | Unused legacy ranking; no consistent hardness scale |
| **poisson_ratio** | Dimensionless metadata; currently unused by the solver |
| **breaking_strain**, **yield_strain**, **compression_breaking_strain** | Legacy spring deformation/failure thresholds |
| **compression_limit**, **spring_min_ratio**, **stiffness_scale**, **shear_boost** | Numerical-model settings, not measured bulk constants |

Representative stored bulk values: **Tungsten** – ρ=19,300 kg/m³, E=411 GPa;
**Steel** – ρ=7,850 kg/m³, E=207 GPa; **Alumina** – ρ=3,960 kg/m³, E=370 GPa.
Agreement with a bulk-property reference does not validate the simulation's
contact or fracture response. Runtime values were preserved during this audit.

---

## Numerical details

- **Substeps:** 4 per frame; time scale slider multiplies effective dt.
- **Collision distance:** fixed (e.g. 0.012 m); particle radius used for overlap and response.
- **Activation:** penetrator always “active”; armor particles become active when velocity² > threshold or via propagation along springs from collisions.
- **Angular velocity:** computed from spring torques about center of mass (for visualization / possible future use); primary motion is still linear **v** from spring forces.

---

## Project layout

```
├── armorsim_qt.py   # Entry point
├── mainwindow.py    # Main window, docks, scene tree, properties, simulation controls
├── views.py         # 3D view (SimViewWidget), 2D ortho canvases (OrthoCanvas)
├── gizmo.py         # Transform gizmo for dragging bodies
├── physics.py       # Physics engine (springs, collisions, bond breaking)
├── bodies.py        # Soft bodies (Penetrator, ArmorPlate) and geometry
├── materials.py     # Material properties (tungsten, steel, ceramic)
├── requirements.txt
└── README.md
```
