# Domain Generalization for Audio and Face Deception Detection

## Overview
This repository implements **domain generalization (DG) methods** for deception detection using audio and face modalities.  
The goal is to **train models that generalize across domains** (e.g., ethnic groups, languages) rather than merely performing well on in-domain data.


- **LODO (Leave-One-Domain-Out) Principle**:
- All methods follow the **LODO protocol**:
1. The dataset is divided into distinct domains, e.g., ethnic or language groups such as:
   - CHINESE
   - MALAY
   - HINDI
2. In each experiment, **one domain is held out as the test set**, while the remaining domains are used for training.
3. The model is evaluated on the held-out domain to measure its **generalization capability**.

- ### Example LODO Protocols

| Train Domains         | Test Domain |
|-----------------------|------------|
| ["CHINESE", "MALAY"]  | ["HINDI"]  |
| ["CHINESE", "HINDI"]  | ["MALAY"]  |
| ["MALAY", "HINDI"]    | ["CHINESE"]|

These protocols simulate real-world scenarios where the model encounters **new, unseen populations**.


## DG Methods Implemented
1. **Baseline** – standard supervised training with modality-specific backbones (ResNet50 for audio, SlowR50 for face).  
2. **Gradient Reversal (GRL)** – introduces a domain-adversarial branch to encourage **domain-invariant features**.  
3. **Contrastive Learning (CL)** – uses a contrastive loss to align features across domains.  
4. **Combined / Multi-Loss** – integrates baseline, GRL, and CL for **robust cross-domain generalization**.

## Dataset Notice
- The **ROSE V2 dataset** used in this project is **not publicly available**.
