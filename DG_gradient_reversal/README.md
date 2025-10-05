# Gradient Reversal Layer (GRL) for Domain Generalization

## Overview
This method introduces a **domain-adversarial branch** using a **gradient reversal layer (GRL)**:
- The GRL encourages the backbone to learn **domain-invariant features**.
- Compatible with audio or face modalities.

## Key Difference from Baseline
- Adds a **domain classifier** alongside the deception classifier.
- During backprop, GRL **reverses gradients** to reduce domain-specific cues.
- Improves cross-domain generalization compared to the baseline.

## Usage
- Specify source and target domains via LODO splits.
- Train with the GRL branch enabled to encourage domain-invariance.
