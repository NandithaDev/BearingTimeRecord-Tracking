
# Passive Sonar Target Detection and Caption Generation

## Overview

This project generates a synthetic multimodal passive sonar dataset consisting of:

- Bearing-Time Record (BTR) images
- Ground truth masks
- Scene metadata (JSON)
- Natural language captions

The simulated sonar system uses a **32-hydrophone Uniform Linear Array (ULA)**. Captions are generated locally using **Google Gemma 3 (4B)** through **Ollama**.

---

## Workflow

```text
                 Random Target Generation
        (DOA, Velocity, SNR, No. of Targets)
                        │
                        ▼
            Passive Sonar Signal Simulation
             (32-Hydrophone ULA + Noise)
                        │
                        ▼
             Beamforming → BTR Generation
                        │
        ┌───────────────┼───────────────┐
        ▼               ▼               ▼
   BTR Images     Ground Truth      Scene JSON
                                         │
                                         ▼
                         Gemma 3 (via Ollama)
                                         │
                                         ▼
                          Natural Language Caption
```

---

## Dataset Contents

Each simulation generates:

- BTR image (`.jpg`)
- BTR matrix (`.npy`)
- Ground truth image (`.jpg`)
- Ground truth matrix (`.npy`)
- Scene metadata (`.json`)
- Caption (`.txt`)

---

