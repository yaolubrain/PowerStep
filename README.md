# PowerStep: Memory-Efficient Adaptive Optimization via $\ell_p$-Norm Steepest Descent

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)

Official implementation of **PowerStep**, a memory-efficient optimizer that achieves coordinate-wise adaptivity without storing second-moment statistics. PowerStep matches AdamW's convergence while halving optimizer memory, and enables stable training under aggressive `int8` quantization for ~8× memory reduction.

📄 **Paper:** [PowerStep: Memory-Efficient Adaptive Optimization via $\ell_p$-Norm Steepest Descent](https://arxiv.org/abs/2605.10335)

---

## 📦 Overview

Adam and AdamW maintain two optimizer states per parameter (first and second momentum), doubling the memory footprint compared to SGD. PowerStep eliminates the second-moment buffer entirely by applying a **signed power transform** directly to the momentum:

![PowerStep Algorithm Overview](algo.png)


This simple modification provides coordinate-wise adaptivity with **half the memory**, and the single-buffer design naturally supports aggressive `int8` quantization.

![Experiments](exp.png)

### Memory Use

![Memory](memory.png)

