# ear-EEG Mental Stress Dataset — Directory & Format Guide

This document specifies the **directory layout** and **file format** expected by `train_v2.py` when training models (e.g., `Deep4AttNet`, `Deep4SelfAttNet`, `Deep4LightTransNet`) from `Deep4Net.py`.

---

## 1) Directory Layout

Place preprocessed files under a common root (default: `Preprocessed/`) with **two class folders**:

```
Preprocessed/
├── Relax/
│   ├── S01.npy
│   ├── S02.npy
│   ├── ...
│   └── S32.npy
└── Stress/
    ├── S01.npy
    ├── S02.npy
    ├── ...
    └── S32.npy
```

**Requirements**
- Each subject must appear **once** in `Relax/` and **once** in `Stress/` with **matching filenames** (e.g., `S07.npy` in both).
- Default setup assumes **32 subjects** (i.e., 32 files per class folder). If you change the number of subjects, adjust the script accordingly.

---

## 2) File Naming

- Use a consistent scheme such as `SXX.npy` (e.g., `S01.npy`, `S02.npy`, …).
- Filenames must match **1:1** between `Relax/` and `Stress/` so the script can pair them by subject ID.

---

## 3) File Format

Each `.npy` file should be a NumPy array with **channels-first** layout and **60 s windows at 125 Hz**.

- **Shape**: either
  - **(C, T)** for a single segment, or
  - **(S, C, T)** for multiple segments in the same file.
- **Our dataset uses**: **C × T = 8 × 7500** (8 channels, 7,500 samples).
- **Sampling rate**: 125 Hz → 60 s windows.
- **Data type**: `float32` is recommended (others will work if compatible).

> If your arrays are saved as (T, C), please **transpose to (C, T)** before saving to avoid shape errors.

---

## 4) Channel Map

The training script selects **ear-EEG channels only**:
- **Assumed indices (0-based)**: `channels = [6, 7]` → **left/right preauricular ear-EEG**.
- If your dataset stores ear-EEG at different indices (or 1-based labels), **update the channel list** in the script accordingly.

Example channel order (illustrative; update to match your data):
```
0: Af7, 1: Fpz, 2: AF8, 3: C3, 4: Cz, 5: C4, 6: Ear-L, 7: Ear-R
```

---

## 5) Labels

- Class labels are inferred from the **folder name**: `Relax/` → class 0, `Stress/` → class 1 (or as defined in the script).
- No per-file label arrays are required if you follow the folder structure above.

---

## 6) Subject Pairing & Cross-Validation

- Files are paired by subject ID across `Relax/` and `Stress/`.
- The default experiments use **8-fold subject-level cross-validation**. Ensure **no subject leakage** between folds if you modify the split logic.

---

## 7) Quick Shape Check (Python)

```python
import numpy as np, os

root = "Preprocessed"
for cls in ["Relax", "Stress"]:
    p = os.path.join(root, cls, "S01.npy")  # pick any file
    x = np.load(p)
    print(cls, p, "shape:", x.shape)

    # Expect (C, T) == (8, 7500) or (S, C, T) where C==8 and T==7500
    if x.ndim == 2:
        C, T = x.shape
        assert C == 8 and T == 7500, "Expected (8, 7500)."
    elif x.ndim == 3:
        S, C, T = x.shape
        assert C == 8 and T == 7500, "Expected (S, 8, 7500)."
    else:
        raise ValueError("Array must be 2D or 3D.")
```

---

## 8) Common Pitfalls

- **Mismatched filenames** between `Relax/` and `Stress/` (breaks subject pairing).
- **Wrong shape** (e.g., (T, C) instead of (C, T)).
- **Ear-EEG not located at indices [6, 7]** → update the channel list in the script.
- **Different sampling rate / window length** → ensure **T == 7500** for 60 s @ 125 Hz.

---

## 9) Reproducibility Notes

- Save arrays with a fixed dtype (e.g., `float32`) and consistent preprocessing steps across all files.
- Keep a separate log of preprocessing (filters, artifact removal, re-referencing) to ensure runs are comparable.

---s