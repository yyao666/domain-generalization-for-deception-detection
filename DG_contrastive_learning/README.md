# Contrastive Learning for Domain Generalization

## Overview
This method leverages **contrastive learning** to enhance feature representations:
- Encourages samples of the same class to be closer in embedding space, even across domains.
- Often combined with standard cross-entropy loss.

## Key Difference from Baseline
- Introduces a **contrastive loss term** in addition to classification loss.
- Helps the model **generalize to unseen domains** by aligning features across domain gaps.

## Usage
- Construct positive/negative pairs from source domains.
- Evaluate on held-out LODO domain to assess generalization.
