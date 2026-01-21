# Ka-OCR
This is a monorepo of OCR project that is able to detect Georgian (Ka)
texts on images/PDFs.

The project consists of few parts or subprojects:
* Dataset generation - "dataset_gen"
* Source code for ML model fine-tuning - "ml_training"
* FastAPI Backend - "api"

Due to different Python version requirements, dataset_gen is better
to be opened and configured as separate project if working on it.
The other Two are share same environment and therefore can be opened
in IDE from project root (Ka-OCR).

# Setup
Using uv is recommended for quick and easy setup.
from project root
```bash
uv venv
```

Then depending on which subproject you work on:
### For "ml_training" or "api"
go to the subproject you want to work on and sync dependencies.
for example:
```bash
cd api
```
and
```bash
uv sync
```

### For "dataset_gen"
Open dataset_gen folder as independent project in your IDE, then:
```bash
uv sync
```