# Zeroth Review Presentation 
## Problem A: Thermal-Aware Power Budgeting on a Chip
**Team:** Odd Group &nbsp;|&nbsp; **Team members:** Chriss Mathew Rajan (Roll No. 23); Devika S Liju (Roll No. 25); Renjitha Babu (Roll No. 55); Sethuparvathy K J (Roll No. 58); Sharon Thomas (Roll No. 61)


---

## 1. Problem Statement

> **Allocate a maximum total thermal power budget of P_max = 5 W across N = 4 active heat-dissipating functional blocks on a silicon processor die to minimize the peak operating temperature T_max and eliminate localized thermal hotspots.**

### Why this matters
- Concentrated power dissipation → localized hotspots → accelerated electromigration and device degradation.
- Heat conduction on a chip is governed by a 2D elliptic Poisson PDE under steady-state conditions.
- The optimal power allocation is a convex minimax linear programming problem.

### Constraints
| Constraint | Mathematical Form |
|:-----------|:-----------------|
| Total power budget | q1 + q2 + q3 + q4 ≤ P_max = 5 W |
| Non-negativity | qj ≥ 0  for all j in {1, 2, 3, 4} |
| Ambient boundary | T(x,y) at boundary = T_amb = 300 K |

---

## 2. Physical Domain Parameters

| Parameter | Symbol | Value |
|:----------|:------:|:------|
| Die geometry | Lx × Ly | 10 mm × 10 mm (0.01 m × 0.01 m) |
| Silicon thermal conductivity | k | 150 W/(m·K) |
| Spatial grid resolution | Nx × Ny | 40 × 40 = 1600 nodes |
| Grid node spacing | Δx = Δy | 0.25 mm |
| Ambient / boundary temperature | T_amb | 300 K |
| Maximum power budget | P_max | 5 W |
| Number of heat-generating blocks | N | 4 cores |

### Core Block Coordinates (on 40×40 grid)
| Block | Grid Coordinate |
|:------|:---------------|
| Block 1 | (10, 10) |
| Block 2 | (10, 29) |
| Block 3 | (29, 10) |
| Block 4 | (29, 29) |

---

## 3. Mathematical Areas Referred

| Area | Role in Project |
|:-----|:---------------|
| **Vector Calculus** | Heat flux field q(r), Fourier's Law, Gauss divergence theorem for energy balance |
| **Partial Differential Equations** | 2D steady-state Elliptic Poisson heat equation with Dirichlet boundary conditions |
| **Laplace Transforms** | Transient-to-steady-state transformation; thermal impedance matrix derivation |
| **Linear Algebra** | FDM sparse matrix assembly (A in R^(1600x1600)), sparse direct linear system solving |
| **Linear Programming** | Minimax peak temperature LP via epigraph reformulation, HiGHS Simplex solver |

---

## 4. Important Governing Equations

### Stage 1 — Fourier's Law & Energy Balance (Vector Calculus)

```
q(r) = -k ∇T(r) = -k ( (∂T/∂x) î + (∂T/∂y) ĵ )

∇ · q(r) = Q(x,y)  ⟹  ∇ · (-k ∇T) = Q(x,y)
```

> **Meaning:** Heat flux vector from temperature gradient (Fourier's Law) combined with conservation of thermal energy (divergence theorem) yields the governing PDE.

---

### Stage 2 — 2D Steady-State Poisson PDE

```
∇²T(x,y) = ∂²T/∂x² + ∂²T/∂y² = -Q(x,y)/k

T(x,y)|boundary = T_amb = 300 K   (Isothermal Dirichlet BCs)
```

> **Meaning:** Elliptic 2D PDE governing steady-state heat distribution across the die. Dirichlet BCs model the heat sink at all 4 outer edges.

---

### Stage 3 — Laplace Transform & Thermal Impedance Matrix

```
k ∇² T̄(r, s) - ρ cp s T̄(r, s) = -Q̄(r, s)

Z_th(s) = lim(s→0) ΔT̄(s)/P̄(s) = A⁻¹
```

> **Meaning:** Laplace-domain heat equation. The steady-state limit (s → 0) gives the thermal impedance matrix, which equals the inverse of the FDM coefficient matrix A.

---

### Stage 4 — 5-Point Finite Difference Discretization Stencil

```
(T_(i+1,j) + T_(i-1,j) + T_(i,j+1) + T_(i,j-1) - 4 T_(i,j)) / Δx²  =  -Q_(i,j) / k

⟹  A T = B q
```

> **Meaning:** The continuous Laplacian is discretized at each of the 1600 grid nodes. Produces a sparse 1600×1600 block-tridiagonal linear system. A is the discrete Laplacian, T is the temperature vector, B maps block power injections to source terms.

---

### Stage 5 — Linear Programming Epigraph Minimax Formulation

```
minimize t
  over (q1, q2, q3, q4, T, t)

subject to:
  A T - B q = 0             (heat equation)
  T_i - T_amb ≤ t           (minimax bound, for all i = 1..1600)
  q1 + q2 + q3 + q4 ≤ 5 W   (power budget)
  qj ≥ 0                    (non-negativity)
```

> **Meaning:** The minimax temperature objective is converted to a convex LP via epigraph form. Variable t represents the peak temperature above ambient. Solved by `scipy.optimize.linprog` using the HiGHS Simplex or Interior-Point algorithm.

---

## 5. Python Libraries & Functional Purpose

| Library | Module | Specific Functional Purpose |
|:--------|:-------|:---------------------------|
| **SymPy** | `sympy` | Symbolic derivation of heat PDE; Laplace transform computation; closed-form thermal impedance verification; cross-check FDM stencil coefficients |
| **NumPy** | `numpy` | High-performance N-dimensional arrays; 40×40 spatial meshgrid generation; boundary condition coordinate masking; matrix-vector operations |
| **SciPy** | `scipy.sparse` | Memory-efficient sparse matrix creation (`lil_matrix`, `csr_matrix`); assembly of 1600×1600 block-tridiagonal Laplacian A without RAM overflow |
| **SciPy** | `scipy.sparse.linalg` | Fast sparse direct solver (`spsolve`); condition number diagnostics; matrix stability audit |
| **SciPy** | `scipy.optimize.linprog` | LP solver (HiGHS Simplex / Interior-Point); minimax epigraph LP execution; optimal q* and T* computation |
| **Matplotlib** | `matplotlib.pyplot` | 2D thermal distribution heatmaps (`pcolormesh`, `coolwarm` colormap); isothermal contour lines |
| **Matplotlib** | `matplotlib.pyplot` | Power allocation bar charts (baseline vs LP-optimal); P_max sweep: T_peak vs total power budget |

### Tool Infrastructure
```
Git Repository
├── main                    ← stable releases
├── feature/pde             ← Student A continuous derivations
├── feature/fdm             ← Student C sparse matrix assembly
└── feature/lp-solver       ← Student D linprog optimization

requirements.txt            ← version-pinned: numpy, scipy, matplotlib, sympy
LaTeX template              ← 6-page departmental report (amsmath, pgfplots, booktabs)
```

---

## 6. Module Division & Student Assignments

### Odd Group — 5-Member Responsibility Matrix

---

### Student A (Chriss Mathew Rajan) — Field Model & PDE Analyst (Stages 1 & 2)

**Primary Scope:** Vector Calculus → PDE Formulation

**What to Cover:**
- Derive Fourier's Law: q = -k∇T from thermodynamic first principles.
- Apply Gauss divergence theorem: ∇·q = Q(x,y) (energy conservation).
- Combine to derive the governing 2D Poisson PDE: ∇²T = -Q/k.
- Classify PDE type (Elliptic — guarantees unique stable solution).
- Model Dirichlet boundary conditions: T = T_amb = 300 K on all 4 outer edges.
- Physical justification for symmetric block placement at (10,10), (10,29), (29,10), (29,29).
- Write Sections 1 & 2 of the LaTeX report.

---

### Student B (Sharon Thomas) — Transforms Specialist (Stage 3)

**Primary Scope:** Laplace Transforms → Thermal Impedance Matrix

**What to Cover:**
- Apply Laplace transform to the transient heat equation.
- Derive Laplace-domain governing equation: k∇²T̄ - ρ cp s T̄ = -Q̄.
- Evaluate the steady-state limit (s → 0) to extract thermal impedance Z_th(s).
- Prove: Z_th(s)|_(s=0) = ΔT/P = A⁻¹ (relating to the FDM matrix).
- Verify all symbolic derivations using SymPy `laplace_transform()` and `limit()`.
- Write Section 3 of the LaTeX report.

---

### Student C (Devika Liju) — Linear Algebra & Discretization Lead (Stage 4)

**Primary Scope:** FDM Sparse Matrix Assembly → Linear System

**What to Cover:**
- Implement 5-point finite difference Laplacian stencil on 40×40 grid.
- Map 2D grid indices (i, j) to 1D flat node index n = i·40 + j (1600 equations total).
- Assemble sparse 1600×1600 block-tridiagonal matrix A using `scipy.sparse.lil_matrix`.
- Embed Dirichlet BCs directly on diagonal entries (set row to identity for boundary nodes).
- Audit diagonal dominance: verify each diagonal entry |A_nn| exceeds sum of off-diagonal magnitudes.
- Check condition number using `scipy.sparse.linalg.norm` to confirm numerical stability.
- Write Section 4 of the LaTeX report.

---

### Student D (Renjitha Babu) — LP Optimization Engineer (Stage 5)

**Primary Scope:** LP Formulation → Optimal Power Allocation

**What to Cover:**
- Perform epigraph transformation of the minimax peak temperature objective.
- Define decision variable vector: x = [q1, q2, q3, q4, T1, ..., T1600, t]ᵀ.
- Build LP constraint matrices for `scipy.optimize.linprog` standard form.
- Run solver with `method='highs'`; verify status code = 0 (optimal).
- Validate: qj ≥ 0, Σqj ≤ 5 W, extract T_max_LP from optimal solution.
- Write Section 5 of the LaTeX report.

---

### Student E (Sethuparvathy K J) — Pipeline Architect & Integration Lead (Stage 6)

**Primary Scope:** Python Integration → Visualization → Repo Management

**What to Cover:**
- Build master execution script `run.py` that imports all 5 student modules in order.
- Generate Matplotlib figures:
  - **Heatmap 1:** Baseline uniform power distribution temperature field.
  - **Heatmap 2:** LP-optimal power allocation temperature field (side-by-side).
  - **Bar Chart:** Baseline vs optimal power per block (q1 through q4).
  - **Sweep Plot:** T_peak vs P_max curve (validate: LP always beats uniform).
- Initialize and maintain Git repository; write `README.md` with setup instructions.
- Finalize and compile the 6-page LaTeX project report.
- Coordinate pre-submission pre-flight checklist verification.

---

## 7. 6-Day Execution Roadmap

| Day | Task | Responsible |
|:---:|:-----|:-----------|
| **Day 1** | Vector calculus derivation & PDE formulation. Git repo + LaTeX template setup. | Student A + Student E |
| **Day 2** | Laplace transform derivation & thermal impedance limit. Spatial mesh setup. | Student B + Student C |
| **Day 3** | 5-point FDM stencil implementation. Sparse matrix A assembly & stability audit. | Student C |
| **Day 4** | Epigraph LP formulation. `scipy.optimize.linprog` implementation & feasibility check. | Student D |
| **Day 5** | Full pipeline integration run. Heatmap generation & P_max sweep validation. | Student E |
| **Day 6** | LaTeX report final compile. Pre-flight checklist. 10-min live demo rehearsal. | All |

---

## 8. Risk Mitigation Register

| Risk | Trigger | Mitigation Strategy |
|:-----|:--------|:-------------------|
| **Matrix Singularity** | Boundary rows left as zero → singular A | Embed Dirichlet BCs directly on matrix diagonal; guarantees strictly diagonally dominant, negative-definite A |
| **LP Infeasibility** | No feasible (q, T, t) found | Strict qj ≥ 0 bounds + t ≥ 0 + Σqj ≤ 5 W ensures non-empty feasible region |
| **LP Non-Convergence** | Simplex cycling or numerical instability | Switch from 'simplex' to 'highs' interior-point method; assert result.status == 0 |
| **Grid Memory Overflow** | Dense 1600² matrix in RAM | Use scipy.sparse.csr_matrix; only O(N) non-zeros stored instead of O(N²) |

---

## 9. Target Quantitative Outcome

```
T_peak (Uniform allocation) ≈ 343 K
         ↓  LP Optimization
T_peak (LP Optimal)         ≈ 327 K

Minimum guaranteed reduction: ≥ 5 K (verified by LP optimality conditions)
```
---

## 10. References

### A. Heat Transfer and Vector Calculus (Stages 1 & 2)
1. F. P. Incropera, D. P. DeWitt, T. L. Bergman, A. S. Lavine, *Fundamentals of Heat and Mass Transfer*, Wiley. (Fourier's law, heat conduction equation, thermal conductivity of silicon.)
2. Y. A. Çengel, A. J. Ghajar, *Heat and Mass Transfer: Fundamentals and Applications*, McGraw-Hill.
3. J. Stewart, *Calculus: Early Transcendentals*, Cengage. (Gradient, divergence, divergence theorem.)

### B. Partial Differential Equations and Laplace Transforms (Stages 2 & 3)
4. E. Kreyszig, *Advanced Engineering Mathematics*, Wiley. (Poisson/Laplace equations, Laplace transforms, PDE classification.)
5. L. C. Evans, *Partial Differential Equations*, American Mathematical Society. (Elliptic PDEs, existence and uniqueness of solutions.)
6. H. S. Carslaw, J. C. Jaeger, *Conduction of Heat in Solids*, Oxford University Press. (Classical reference for transient and steady-state conduction.)

### C. Finite Difference Method and Numerical Linear Algebra (Stage 4)
7. R. J. LeVeque, *Finite Difference Methods for Ordinary and Partial Differential Equations*, SIAM, 2007. (Five-point Laplacian stencil, Dirichlet boundary conditions.)
8. G. Strang, *Computational Science and Engineering*, Wellesley-Cambridge Press, 2007. (Discrete Laplacian, sparse block-tridiagonal systems.)
9. Y. Saad, *Iterative Methods for Sparse Linear Systems*, SIAM, 2003.

### D. Linear and Convex Optimization (Stage 5)
10. S. Boyd, L. Vandenberghe, *Convex Optimization*, Cambridge University Press, 2004. (Epigraph form, minimax problems, LP.)
11. D. Bertsimas, J. N. Tsitsiklis, *Introduction to Linear Optimization*, Athena Scientific, 1997.
12. Q. Huangfu, J. A. J. Hall, "Parallelizing the dual revised simplex method," *Mathematical Programming Computation*, vol. 10, pp. 119–142, 2018. (The algorithm behind the HiGHS dual simplex solver.)
13. HiGHS: high-performance open-source LP/MIP solver. https://highs.dev

### E. Thermal Management of Chips (Motivation)
14. K. Skadron et al., "Temperature-aware microarchitecture," *Proc. 30th Int. Symposium on Computer Architecture (ISCA)*, 2003.
15. W. Huang et al., "HotSpot: A compact thermal modeling methodology for early-stage VLSI design," *IEEE Trans. on Very Large Scale Integration (VLSI) Systems*, vol. 14, no. 5, pp. 501–513, 2006.
16. J. R. Black, "Electromigration: A brief survey and some recent results," *IEEE Trans. on Electron Devices*, vol. 16, no. 4, pp. 338–347, 1969.

### F. Software and Libraries
17. P. Virtanen et al., "SciPy 1.0: fundamental algorithms for scientific computing in Python," *Nature Methods*, vol. 17, pp. 261–272, 2020.
18. C. R. Harris et al., "Array programming with NumPy," *Nature*, vol. 585, pp. 357–362, 2020.
19. A. Meurer et al., "SymPy: symbolic computing in Python," *PeerJ Computer Science*, 3:e103, 2017.
20. J. D. Hunter, "Matplotlib: A 2D graphics environment," *Computing in Science & Engineering*, vol. 9, no. 3, pp. 90–95, 2007.
21. SciPy documentation: `scipy.optimize.linprog`. https://docs.scipy.org/doc/scipy/reference/generated/scipy.optimize.linprog.html
22. SciPy documentation: `scipy.sparse` and `scipy.sparse.linalg`. https://docs.scipy.org/doc/scipy/reference/sparse.html
---
