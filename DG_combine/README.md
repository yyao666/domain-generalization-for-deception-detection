# Multi-Loss Approach for Domain Generalization - Loss function optimization


## Overview
Combines baseline, GRL, and CL:
- Standard classification loss for deception.
- Domain-adversarial loss for domain invariance.
- Contrastive loss for feature alignment.

## Advantages
- Encourages **accurate predictions**, **domain invariance**, and **robust embeddings** simultaneously.
- Often achieves **best performance under LODO evaluation**.
- **Introducing focal loss to solve the problem of category imbalance**

## Domain Generalization
- LODO protocol applied for fair comparison with other methods.
- Demonstrates **robust cross-domain generalization**.

## Usage
- Enable all loss terms in training.
- Specify source and held-out domains.

