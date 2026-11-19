# Experiment 5 – Performance Results

## Test environment
- GPU: NVIDIA RTX 4500 Ada Generation
- CUDA Toolkit: 13.4
- Compilation target: `sm_89`
- Network dimensions: batch 256, input 128, hidden 64, output 10
- Each program execution averages timing over 20 repetitions.

## Recorded measurements

| Run | CPU time (ms) | GPU kernel-only (ms) | GPU end-to-end (ms) | Kernel-only speedup | End-to-end speedup |
|---:|---:|---:|---:|---:|---:|
| 1 | 0.717745 | 0.021043 | 0.091440 | 34.108168× | 7.849355× |
| 2 | 0.728520 | 0.031587 | 0.077320 | 23.063771× | 9.422142× |
| 3 | 0.720060 | 0.021248 | 0.084235 | 33.888366× | 8.548228× |
| 4 | 0.719140 | 0.043459 | 0.180730 | 16.547475× | 3.979085× |

## Arithmetic averages across the four runs

| Measurement | Average |
|---|---:|
| CPU execution time | 0.721366 ms |
| GPU kernel-only time | 0.029334 ms |
| GPU end-to-end time | 0.108431 ms |
| Kernel-only speedup (mean of run speedups) | 26.901945× |
| End-to-end speedup (mean of run speedups) | 7.449702× |

## Correctness results

All four recorded runs reported:
- Matrix multiplication + bias: PASSED
- ReLU: PASSED
- Sigmoid: PASSED
- Maximum absolute output difference: 0.000000
- CPU and GPU outputs: MATCH (within tolerance)

## Interpretation

GPU kernel-only time measures GPU computation after input data has already been copied to device memory. GPU end-to-end time includes host-to-device transfers, GPU execution, synchronization, and copying the output back. Therefore, end-to-end speedup is lower than kernel-only speedup for these runs. Small timing variations between runs are expected.

## Conclusion

The CUDA neural-network implementation passed all kernel tests and produced CPU/GPU outputs that matched within the program's tolerance. The measured runs show faster GPU execution for this workload, while also demonstrating the impact of data-transfer overhead on end-to-end performance.
