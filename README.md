# GPU-Accelerated Database Query Engine
<p align="center">
  <img src="https://github.com/user-attachments/assets/74c0faa0-0400-41fc-89a7-c255a0ad587d" alt="Design Visualization" width="500">
</p>

## Table of Contents
- [Overview](#overview)
- [Pipeline](#pipeline)
- [Features](#features)
- [Performance](#performance)
- [Installation](#installation)
- [License](#license)

## Overview
A CUDA-based database query engine for efficient processing of large datasets using GPU parallelism. Supports filtering, aggregation, projection, sorting, and joins.

## Pipeline
Large datasets are split into batches, processed in parallel on the GPU, and merged on the CPU.
<img width="912" height="452" alt="image" src="https://github.com/user-attachments/assets/a54a5d62-1dac-4090-8b47-5817a103a69f" />

## Features
- **Filter Kernel**: Stack-based condition evaluation, NULL handling.
- **Aggregate Kernel**: Shared memory reductions, thread synchronization.
- **Project Kernel**: Coalesced memory access, precomputed offsets.
- **Order By Kernel**: Iterative merge sort with co-ranking, double buffering.
- **Join Kernel**: Nested loop joins with stack-based condition evaluation, shared memory tiling.


## Performance
| Query Type | GPU Time (s) | CPU Time (s) | Table Sizes |
|------------|--------------|--------------|-------------|
| Get All Rows | 4.9 | 1.62 | 264,821 Rows |
| Projection + Filter | 5.70 | 1.79 | 264,821 Rows |
| Max Aggregate | 7.65 | 1.89 | 330,609 Rows |
| Sorting (Numeric) | 17.14 | 2.02 | 264,821 Rows |
| Complex Join | 45.3 | 2.13 | 264,821 & 330,609 Rows |

**Observations**: CPU outperforms for simple/small queries; GPU excels for complex/large datasets.

## Installation
1. Clone: `https://github.com/Sara-Gamal12/GPU-Accelerated-Database-Query-Engine`
2. Install CUDA toolkit and C++ compiler.
3. Build: `make`
4. Run: `./main`


## License
MIT License. See [LICENSE](LICENSE).
