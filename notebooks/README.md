# Notebooks

This directory is reserved for exploratory work by Team 10:

- Data exploration and dataset statistics
- OCR experiments on sample review screenshots
- Attention-weight visualisation prototypes
- Training curves and error analysis

## Conventions

1. Name notebooks descriptively, e.g. `01_data_exploration.ipynb`,
   `02_ocr_experiments.ipynb`, `03_attention_prototype.ipynb`.
2. Keep notebooks runnable **from the repository root** so imports work:

   ```python
   import sys
   from pathlib import Path
   sys.path.insert(0, str(Path.cwd()))
   from src.preprocessing import preprocess_review
   ```

3. Put reusable logic in `src/` modules; notebooks should only orchestrate
   and visualise, never hold the only copy of important code.
4. Do **not** commit large data files or trained weights from notebooks -
   `.gitignore` already excludes `data/`, `models/`, and `.ipynb_checkpoints/`.
5. Strip outputs before committing very large notebooks, or keep them small.
