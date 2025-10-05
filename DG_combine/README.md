# Multi-Loss Approach for Domain Generalization - Loss function optimization

## Overview
This module combines:
- Standard classification loss (baseline)
- Domain-adversarial loss (GRL)
- Contrastive loss (CL)

## Advantages
- Jointly encourages:
  - **Accurate deception prediction**
  - **Domain-invariant feature learning**
  - **Robust embedding alignment across domains**
- Typically achieves the **strongest cross-domain generalization**.

## Usage
- Enable all loss terms in training.
- Evaluate under the **same LODO protocols** as other methods.
