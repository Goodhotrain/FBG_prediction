# NACFN

NACFN predicts fasting blood glucose (FBG) from static clinical features, nutrition records, and historical FBG. The model pools the static feature tokens with sparsemax. For each day, it applies self-attention across nutrient tokens and averages them into a daily representation. The static representation then attends to the valid daily representations, and a learned gate incorporates historical FBG. Training uses a counterfactual consistency loss alongside the regression loss.

## Code overview

| Path | Purpose |
| --- | --- |
| [`main.py`](main.py) | Trains NACFN and saves metrics and checkpoints. |
| [`valid.py`](valid.py) | Evaluates a checkpoint and exports predictions to CSV. |
| [`opt.py`](opt.py) | Defines the shared command-line options. |
| [`models/model.py`](models/model.py) | Implements `NACFN`, the model used by both entry points. Other modules in `models/` contain earlier models and experimental components. |
| [`data/dataset.py`](data/dataset.py) | Reads the CSV inputs, preprocesses features, and builds masked nutrition history windows. |
| [`common/`](common/) | Contains the training loop, regression metrics, optimizer and scheduler setup, and checkpoint utilities. |
| [`tests/`](tests/) | Checks the data pipeline and model with synthetic inputs. |

## Training and evaluation

```bash
pip install -r requirements.txt
python main.py --run-name example \
  --label-path data/labels.csv --static-path data/static.csv --nutrition-path data/nutrition.csv
python valid.py --checkpoint results/example/checkpoints/best.pt \
  --label-path data/labels.csv --static-path data/static.csv --nutrition-path data/nutrition.csv
```

Supply your own CSVs. The label file starts with the numeric FBG target and the split (`train` or `test` by default). The static file follows the same row order as the labels. Both the static and nutrition files require `UserID` and `Day` columns. The last nutrition feature is historical FBG. Run `python main.py --help` for options, including feature dimensions. Use the same model settings for training and evaluation.

Checkpoints from the earlier cross-day encoder are incompatible with the intra-day encoder. Train a new checkpoint before evaluation.
