# Baseline Deception Detection

## Overview
Implements the standard supervised baseline for audio or face-based deception detection.

### Key Points
- Audio backbone: **ResNet50**
- Face backbone: **SlowR50**
- Linear classifier on top of the backbone.

## Domain Generalization
- Uses **LODO**: train on all domains except one, test on the held-out domain.
- Serves as a reference to evaluate improvements from other DG methods.

## Usage
- Configure dataset paths.
- Specify training/testing domains.
- Run the training script for baseline performance.
