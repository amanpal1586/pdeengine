# PDE Engine

A unified computational framework for solving Partial Differential Equations (PDEs) using multiple numerical methods across Python, MATLAB, and/or C++ — built to compare accuracy, stability, and computational performance, not just produce solutions.

## Overview

PDE Engine implements and benchmarks classical numerical schemes for standard PDE problems (heat, wave, diffusion equations), including:

- Finite Difference Methods (FDM)
- Crank–Nicolson (CN)
- Explicit / Implicit schemes
- IMEX (Implicit-Explicit) methods
- DuFort–Frankel
- Other relevant PDE-solving schemes

The goal isn't just to solve PDEs — it's to systematically understand *how well* and *how efficiently* each method solves them.

## Core Objectives

The engine evaluates each numerical method across:

- **Accuracy** of the numerical solution
- **Stability** under varying parameters and time-step sizes
- **Computational time and memory usage**
- **Convergence rate and error analysis**
- **Effect of spatial and temporal discretization**
- **Computational efficiency and optimization**

## Optimization Focus

Beyond correctness, the project explores performance improvements through:

- Vectorization
- Sparse matrix operations
- Efficient linear-system solvers
- Parallel computation
- Algorithmic optimization

## Statistical & Future ML Integration

- Statistical measures and error analysis are used to compare methods across experiments (Statistics/Probability lens).
- **Future scope:** Machine Learning models to recommend the most suitable numerical scheme or computational parameters based on PDE characteristics and desired accuracy.

## Intended Users

Designed to be **student-friendly**, primarily for undergraduate students studying Computational PDEs and Numerical Methods — enabling easy implementation, visualization, and experimental comparison of methods (e.g., when Crank–Nicolson outperforms IMEX for a given problem).

## Workflow

```
Define PDE
   → Select Numerical Method
      → Solve
         → Measure Error
            → Benchmark Computation
               → Visualize Solution
                  → Compare Methods
                     → Identify Most Efficient Method
```

## Project Structure

```
pde-engine/
├── python/            # Python implementations of PDE solvers
├── matlab/             # MATLAB implementations
├── cpp/                # C++ implementations (performance-critical solvers)
├── benchmarks/          # Timing, memory, and convergence benchmarking scripts
├── experiments/         # Comparative studies across methods/parameters
├── notebooks/           # Visualization and analysis notebooks
├── docs/                 # Theory notes, method derivations, references
└── README.md
```

*(Structure is illustrative and will evolve as implementation progresses.)*

## Status

🚧 **Work in progress.** This repository is in early development. Method implementations, benchmarking tools, and documentation are being built incrementally.

## Roadmap

- [ ] Implement core Finite Difference solvers (explicit/implicit)
- [ ] Implement Crank–Nicolson scheme
- [ ] Implement IMEX and DuFort–Frankel schemes
- [ ] Build error/convergence analysis module
- [ ] Build computational benchmarking module (time, memory)
- [ ] Add visualization tools for solutions and comparisons
- [ ] Cross-language performance comparison (Python vs MATLAB vs C++)
- [ ] Explore ML-assisted numerical method recommendation

## Vision

PDE Engine aims to bridge the gap between the **mathematical theory** of numerical PDEs and **practical computational performance**, laying groundwork for future extensions into high-performance computing and ML-assisted numerical method selection.

