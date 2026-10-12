# Thermal-Aware Power Budgeting on a Silicon Die

### A PDE-Constrained Linear Programming Approach

**Team Odd Group** · Department of Electronics (S3 EC), Saintgits College of Engineering
Capstone for *PDEs, Transforms and Optimization Techniques* · AY 2025–2026 · Problem A

| Member | Roll No. |
| :--- | :---: |
| Chriss Mathew Rajan | 23 |
| Devika Liju | 25 |
| Renjitha Babu | 55 |
| Sethuparvathy K J | 58 |
| Sharon Thomas | 61 |

---

## Table of Contents

1. [Overview](#overview)
2. [Repository Contents](#repository-contents)
3. [Method in Brief](#method-in-brief)
4. [Key Results](#key-results)
5. [Getting Started](#getting-started)
6. [Notebook Walkthrough](#notebook-walkthrough)
7. [Modelling Notes and Caveats](#modelling-notes-and-caveats)
8. [Limitations](#limitations)
9. [Future Work](#future-work)
10. [References](#references)
11. [Acknowledgements](#acknowledgements)

---

<a id="overview"></a>

## Overview

Uneven heat generation on a processor die creates hotspots, and sustained high temperature accelerates electromigration and device ageing. This project asks:

> How should a fixed **5 W** budget be divided among **four** heat-dissipating blocks on a **10 mm × 10 mm** silicon die so that the **highest temperature anywhere on the die is as low as possible**?

The approach combines four areas of applied mathematics into one pipeline:

```
Fourier's law  →  Poisson PDE  →  Finite-difference matrix  →  Minimax linear program
(Vector Calculus)    (PDEs)         (Linear Algebra, Laplace)      (Optimization, HiGHS)
```

---

<a id="repository-contents"></a>

## Repository Contents

| File | Description |
| :--- | :--- |
| [`report.pdf`](report.pdf) | Full written report (6 pages): derivation, numerical verification, scenarios, sensitivity and mesh studies, limitations. |
| [`Thermal_Aware_Power_Budgeting_final.pdf`](Thermal_Aware_Power_Budgeting_final.pdf) | Review presentation slides (8 slides). |
| [`thermal_power_budgeting_expanded_final_.ipynb`](thermal_power_budgeting_expanded_final_.ipynb) | Jupyter notebook with the full numerical pipeline and figures. |
| [`README.md`](README.md) | This file. |

---

<a id="method-in-brief"></a>

## Method in Brief

1. **Physics.** Fourier's law, **q″ = −k∇T**, with silicon conductivity *k* = 150 W m⁻¹ K⁻¹.
2. **PDE.** Steady-state energy balance gives the 2-D Poisson equation **∇²T = −Q/k** with isothermal edges, *T* = 300 K (an ideal heat sink).
3. **Laplace transform.** Taking *s* → 0 in the transient equation shows the steady-state thermal impedance is the zero-frequency limit, so temperature is linear in power.
4. **Finite differences.** A five-point stencil on a 40 × 40 grid gives a sparse system **Kθ = Bq**. Inverting it yields the thermal impedance matrix **Z = K⁻¹B**, with θ = Zq.
5. **Minimax LP.** The peak rise max(Zq) is convex and piecewise linear, so an epigraph variable *t* turns it into a linear program:

   ```
   minimize   t
   subject to Zq ≤ t·1,   q₁ + q₂ + q₃ + q₄ = P_max,   q ≥ 0
   ```

   It is solved with [HiGHS](https://highs.dev/) through `scipy.optimize.linprog`. A full 1605-variable form (temperatures included) and a reduced 5-variable form are both solved as a consistency check.

---

<a id="key-results"></a>

## Key Results

> Results below follow the **report** (`report.pdf`), which uses an assumed die thickness of 0.5 mm. See [Modelling Notes and Caveats](#modelling-notes-and-caveats) for how this differs from the notebook and slides.

**Numerical soundness**

- Relative residual of the heat-equation solve: ~10⁻¹⁴ to 10⁻¹⁵.
- Full and reduced LPs agree to within 10⁻⁸ K; HiGHS reports status 0 (optimal).
- Condition number of the interior block of K is about 616, so the solves are well conditioned.

**Scenario A: four identical point-like blocks, symmetric layout**

- The equal split of 1.25 W per block is already optimal (peak 312.55 K).
- A mirror-symmetry argument plus convexity explains this. The optimiser can only help when the blocks differ.

**Scenario B: blocks of 0.5, 1.0, 1.5 and 2.0 mm on a side**

| Block | Footprint | Equal split (W) | LP-optimal (W) | Block peak, LP (K) |
| :---: | :---: | :---: | :---: | :---: |
| B1 | 0.5 × 0.5 mm | 1.25 | 0.90 | 307.04 |
| B2 | 1.0 × 1.0 mm | 1.25 | 1.14 | 307.04 |
| B3 | 1.5 × 1.5 mm | 1.25 | 1.39 | 307.04 |
| B4 | 2.0 × 2.0 mm | 1.25 | 1.57 | 307.04 |

- Die peak falls from **309.15 K to 307.04 K**, a **23.1%** cut in the rise above ambient.
- All four blocks reach the same peak temperature, which is the signature of a minimax optimum.
- The benefit scales linearly with the budget: 0.42 K at 1 W up to 4.22 K at 10 W.

**Modelling pitfall:** if the budget is written as Σqⱼ ≤ P_max, zero power is trivially optimal. The budget must be imposed as an **equality**.

---

<a id="getting-started"></a>

## Getting Started

### Requirements

- Python 3.9 or newer
- Packages: `numpy`, `scipy`, `sympy`, `matplotlib`, `jupyter`

The report was produced with NumPy 2.4.4, SciPy 1.17.1, SymPy 1.14.0 and Matplotlib 3.10.8, but any reasonably recent versions should work.

### Install

```bash
# optional: create a virtual environment
python -m venv .venv
source .venv/bin/activate        # Windows: .venv\Scripts\activate

pip install numpy scipy sympy matplotlib jupyter
```

### Run

```bash
jupyter notebook thermal_power_budgeting_expanded_final_.ipynb
```

Then choose **Kernel → Restart & Run All**. The model is small and sparse, so the whole notebook runs on a single CPU core in seconds. The LP itself solves in well under a second.

---

<a id="notebook-walkthrough"></a>

## Notebook Walkthrough

| Stage | Content | Tools |
| :--- | :--- | :--- |
| Setup | Domain constants, mesh, plotting style | NumPy, Matplotlib |
| 1 | Fourier's law, divergence theorem, Poisson form (symbolic check) | SymPy |
| 2 | Laplace transform, final-value theorem, thermal impedance | SymPy |
| 3 | Five-point stencil, sparse matrix assembly, impedance matrix | SciPy sparse, `spsolve` |
| 4 | Epigraph minimax LP | `scipy.optimize.linprog` (HiGHS) |
| 5 | Heat maps, centreline slices, allocation bars, budget sweep | Matplotlib |
| 6 | Conclusions and expected outcomes | n/a |

---

<a id="modelling-notes-and-caveats"></a>

## Modelling Notes and Caveats

The three deliverables were produced at different stages, so their numbers are **not interchangeable**. Please read this before comparing them.

| Aspect | `report.pdf` | Notebook and slides |
| :--- | :--- | :--- |
| Grid spacing Δx | 0.25 mm | 0.2439 mm |
| Die thickness | 0.5 mm (assumed) | not used (point sources, 2-D only) |
| Temperature rises | several kelvin | millikelvin (peak ≈ 300.0065 K) |
| Layout | Scenario A (point blocks) and Scenario B (finite footprints) | point-like cores only |
| LP result | equal split optimal in A; 23.1% reduction in B | near-equal split (1.286 / 1.251 / 1.251 / 1.213 W) |
| Headline comparison | equal vs. LP-optimal | LP-optimal vs. an unbalanced 3.5 / 0.5 / 0.5 / 0.5 W case (about 60% lower rise) |

Two points worth keeping in mind:

- The slides' "about 60% cooler" figure compares against a deliberately bad unbalanced split, not against the equal split. Against the equal split the notebook gain is only about 0.17 mK.
- In the report, the absolute kelvin values depend on the assumed die thickness. The relative improvement (23.1%) does not.

---

<a id="limitations"></a>

## Limitations

- Two-dimensional thin-plate model, with all heat leaving through four edges held at exactly 300 K. Real chips lose most heat vertically through the substrate, spreader and heat sink.
- Constant silicon conductivity and a steady-state analysis only, so transient hotspots are not captured.
- Uniform power density within each block, and fixed block locations (power is optimised, not floorplan).
- Scenario A temperatures depend on the mesh, since point sources in 2-D diverge logarithmically with refinement.
- **No validation** against measurements or an established simulator. The checks show internal consistency, not physical accuracy.

---

<a id="future-work"></a>

## Future Work

- Add a vertical heat path and finite heat-sink resistance (3-D or compact-network model).
- Use temperature-dependent conductivity and leakage, solved by iteration.
- Add per-block power limits and a performance-weighted objective (speed-versus-temperature trade-off).
- Use the transient impedance Zth(s) for time-varying workloads.
- Optimise block placement jointly with power.
- Validate against [HotSpot](http://lava.cs.virginia.edu/HotSpot/) or a finite-element package.

---

<a id="references"></a>

## References

1. F. P. Incropera, D. P. DeWitt, T. L. Bergman and A. S. Lavine, *Fundamentals of Heat and Mass Transfer*, 7th ed., Wiley, 2011.
2. R. J. LeVeque, [*Finite Difference Methods for Ordinary and Partial Differential Equations*](https://epubs.siam.org/doi/book/10.1137/1.9780898717839), SIAM, 2007.
3. E. Kreyszig, *Advanced Engineering Mathematics*, 10th ed., Wiley, 2011.
4. K. Skadron et al., "Temperature-aware microarchitecture," *Proc. ISCA*, 2003. [DOI: 10.1145/859618.859620](https://doi.org/10.1145/859618.859620)
5. W. Huang et al., "HotSpot: A compact thermal modeling methodology for early-stage VLSI design," *IEEE Trans. VLSI Syst.*, 14(5), 2006. [DOI: 10.1109/TVLSI.2006.876103](https://doi.org/10.1109/TVLSI.2006.876103) · [HotSpot homepage](http://lava.cs.virginia.edu/HotSpot/)
6. S. Boyd and L. Vandenberghe, [*Convex Optimization*](https://web.stanford.edu/~boyd/cvxbook/), Cambridge University Press, 2004.
7. Q. Huangfu and J. A. J. Hall, "Parallelizing the dual revised simplex method," *Math. Program. Comput.*, 10(1), 2018. [DOI: 10.1007/s12532-017-0130-5](https://doi.org/10.1007/s12532-017-0130-5) · [HiGHS](https://highs.dev/)
8. P. Virtanen et al., "SciPy 1.0: fundamental algorithms for scientific computing in Python," *Nature Methods*, 17, 2020. [DOI: 10.1038/s41592-019-0686-2](https://doi.org/10.1038/s41592-019-0686-2) · [`scipy.optimize.linprog`](https://docs.scipy.org/doc/scipy/reference/generated/scipy.optimize.linprog.html)
9. J. R. Black, "Electromigration: a brief survey and some recent results," *IEEE Trans. Electron Devices*, ED-16(4), 1969. [DOI: 10.1109/T-ED.1969.16754](https://doi.org/10.1109/T-ED.1969.16754)
10. D. Brooks and M. Martonosi, "Dynamic thermal management for high-performance microprocessors," *Proc. HPCA*, 2001. [DOI: 10.1109/HPCA.2001.903261](https://doi.org/10.1109/HPCA.2001.903261) *(cited in the slides)*

---

<a id="acknowledgements"></a>

## Acknowledgements

Prepared as part of the Applied Mathematics capstone at the Department of Electronics, Saintgits College of Engineering. Thanks to the developers of NumPy, SciPy, SymPy, Matplotlib and HiGHS.
