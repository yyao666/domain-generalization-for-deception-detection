# Domain Generalization for Audio and Face Deception Detection

## Overview
This repository implements **domain generalization (DG) methods** for deception detection using audio and face modalities. 


This is a part of my **Master’s thesis** research at Nanyang Technological University **(NTU)**, Singapore.
It explores Domain Generalization (DG) techniques to improve the robustness of multimodal deception detection models under cross-domain settings.


🎯 Research Objective
The goal of this work is to evaluate how models trained on multiple domains perform on unseen domains, following the **Leave-One-Domain-Out (LODO)** evaluation protocol.

- **LODO (Leave-One-Domain-Out) Principle**:
- All methods follow the **LODO protocol**:
1. The dataset is divided into distinct domains, e.g., ethnic or language groups
   - CHINESE
   - MALAY
   - HINDI
2. In each experiment, **one domain is held out as the test set**, while the remaining domains are used for training.
3. The model is evaluated on the held-out domain to measure its **generalization capability**.

- ### Example LODO Protocols

- Train = {CHINESE, MALAY} → Test = {HINDI}  
- Train = {CHINESE, HINDI}  → Test = {MALAY}  
- Train = {MALAY, HINDI}    → Test = {CHINESE}

Performance reported on the held-out domain reflects the model’s ability to generalize across domain shifts (ethnicities/languages).


These protocols simulate real-world scenarios where the model encounters **new, unseen populations**.


## DG Methods Implemented
1. **Baseline** – standard supervised training with modality-specific backbones (ResNet50 for audio, SlowR50 for face).  
2. **Gradient Reversal (GRL)** – introduces a domain-adversarial branch to encourage **domain-invariant features**.  
3. **Contrastive Learning (CL)** – uses a contrastive loss to align features across domains.  
4. **Combined / Multi-Loss** – integrates baseline, GRL, and CL for **robust cross-domain generalization**.

## Dataset Notice
- The **ROSE V2 dataset** used in this project is **not publicly available**.
