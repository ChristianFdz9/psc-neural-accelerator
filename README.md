# Neural Accelerator for Predictive-Switching Control of Stochastic Gene Regulatory Networks

This repository contains the official code and dataset for the neural accelerator implementation presented in the paper:

> **"Predictive-Switching Control of Stochastic Gene Regulatory Networks: A Contractive PIDE Framework"** > *Christian Fernández, Manuel Pájaro, Gábor Szederkényi, and Irene Otero-Muras.*

## Abstract

This software implements a **Neural Accelerator** designed to enhance the application of optimal control policies in stochastic Gene Regulatory Networks (GRNs). The approach is built upon a Partial Integro-Differential Equation (PIDE) framework, providing a reliable and tractable model for biological systems.

While Predictive-Switching Control (PSC) offers a robust strategy for managing these networks, its application in higher-dimensional systems benefits significantly from computational acceleration. To this end, a feedforward neural network is trained to map system states directly to optimal control configurations. This hybrid framework ensures high-speed execution and scalability while maintaining the rigorous stability guarantees.

## Repository Content

* `train_psc_accelerator.m`: Main MATLAB script for training, validating, and testing the neural network architecture.
* `PSC_Master_Dataset.mat`: Master dataset containing 5,000 samples generated via exhaustive PSC simulations, including state feature vectors and their corresponding labeled optimal controls.
* `LICENSE`: Software license.

## System Requirements

To run the training code, the following are required:
* **MATLAB** (R2021a or later recommended).
* **Deep Learning Toolbox**.
* **Statistics and Machine Learning Toolbox** (for Z-score data preprocessing).

## Instructions for Use

1. **Clone the repository:**
    ```bash
    git clone [https://github.com/ChristianFdz9/psc-neural-accelerator.git](https://github.com/your-username/psc-neural-accelerator.git)
    cd psc-neural-accelerator
    ```
2. **Execute the training:**
    Open MATLAB and run the script:
    ```matlab
    run('train_psc_accelerator.m')
    ```
    The script automatically performs the following phases:
    * Loading the `PSC_Master_Dataset.mat` dataset.
    * Data normalization and partitioning (85% Train/Val, 15% Test).
    * Training via the Levenberg-Marquardt algorithm.
    * Evaluation of precision metrics (Exact Match and Bit Accuracy).

## Technical Details of the Neural Network

The implemented architecture follows the specifications detailed in **Appendix A** of the manuscript:
* **Topology:** Feedforward network with two hidden layers of [20, 10] neurons.
* **Activation Functions:**
    * Hidden layers: Hyperbolic tangent sigmoid (`tansig`).
    * Output layer: Symmetric saturating linear (`satlins`).
* **Input:** Feature vector $z(t_m)$ encoding the distribution state and control history.

## Reproduction of Results

The training is configured to replicate the following performance benchmarks:
* **Exact Match (EM):** ~55% (accuracy of the full control vector prediction).
* **Bit Accuracy (BA):** ~81.5% (accuracy for each individual inductor).

## Technical Environment

To ensure reproducibility, all numerical experiments and training sessions were performed under the following specifications:

* **Processor:** Intel Core i9-14900K CPU (24 cores, 3.2 GHz).
* **Memory:** 128 GB RAM.
* **GPU:** NVIDIA GeForce RTX 4060 (8 GB VRAM).
* **Operating System:** Microsoft Windows 11 (64-bit).
* **Software:** MATLAB R2024a.

The PIDE solvers and PSC algorithms were specifically optimized for this environment to achieve the reported computational speedups.

## License

See LICENSE file.

## Contact

Christian Fernández — GitHub: ChristianFdz9


