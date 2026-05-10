# Winograd Convolution Benchmark on Huawei Kunpeng

This repository was inspired by my Computer Organization and Architecture lab work. While conducting experiments on the Huawei Kunpeng processor, I became interested in how low-level optimization techniques such as loop unrolling, SIMD, and task-level parallelism can accelerate real computational workloads. Since convolution is a fundamental operation in image processing and deep learning, I decided to further explore convolution acceleration on the Kunpeng AArch64 platform. This repository was therefore created as a small benchmark project comparing a naive `3×3` convolution implementation with a Winograd `F(2×2, 3×3)` convolution implementation.

## Motivation

In the lab, I implemented several optimization experiments, including:

- C and AArch64 assembly mixed programming
- Loop unrolling optimization
- ARM NEON SIMD vector operations
- Multi-threaded task-level parallelism

Among these experiments, SIMD-based matrix multiplication made me realize that many image-processing and deep-learning operators are essentially composed of repeated multiply-add operations. Convolution is one of the most typical examples. This motivated me to test whether algorithm-level optimization, such as Winograd convolution, could bring measurable performance improvements on the Huawei Kunpeng processor.

## What This Project Does

This project compares two implementations of `3×3` convolution:

1. **Naive 3×3 convolution**

   The naive method directly computes each output element using 9 multiplications and several additions.

2. **Winograd F(2×2, 3×3) convolution**

   The Winograd method computes a `2×2` output block from a `4×4` input tile. It reduces the number of main multiplications from 36 to 16 for each `2×2` output block by introducing input transformation, kernel transformation, element-wise multiplication, and output transformation.

The benchmark uses:

```text
Input size:       514 × 514
Output size:      512 × 512
Kernel size:      3 × 3
Output channels:  128
Repeat:           3
