## How to run
Needs Python 3.12 or newer.

1. Create a Python virtual environment with `python -m venv .venv`
2. Activate it with `source .venv/bin/activate` (on Windows, `.venv\Scripts\activate`)
3. `pip install -r requirements.txt`
4. Put the test images in a folder named `images` in this folder,
   named `img_001`, `img_002`, and so on (any image extension works).
5. `python run_zeroshot.py`

The first run downloads the model, so it takes a while.

## Output
Saved in a folder named `results` in this folder.

- `predictions_<model_name>_<threshold>.csv`, which has one row per detected object.
- `img_XXX_<model_name>_t<threshold>.jpg`. This is each image with its boxes drawn on it.
