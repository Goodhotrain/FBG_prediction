from argparse import Namespace
import pandas as pd
from data.dataset import FBGDataset

def test_dataset_builds_left_padded_history(tmp_path):
    labels = pd.DataFrame({"FBG": [5.1, 5.4, 5.8], "split": ["train", "train", "test"]})
    static = pd.DataFrame({"UserID": [1, 1, 2], "Day": [1, 2, 1], "age": [30, 30, 50], "bmi": [20, 20, 25]})
    nutrition = pd.DataFrame({"UserID": [1, 1, 2], "Day": [1, 2, 1], "carb": [100, 120, 80], "fat": [40, 45, 60], "history_fbg": [5.0, 5.1, 5.7]})
    for name, frame in (("labels.csv", labels), ("static.csv", static), ("nutrition.csv", nutrition)):
        frame.to_csv(tmp_path / name, index=False)
    args = Namespace(label_path=tmp_path / "labels.csv", static_path=tmp_path / "static.csv", nutrition_path=tmp_path / "nutrition.csv", sequence_length=4)
    train = FBGDataset(args, "train")
    test = FBGDataset(args, "test", train.preprocessor)
    _, static_x, sequence, mask, target = test[0]
    assert static_x.shape == (2,)
    assert sequence.shape == (4, 3)
    assert mask.tolist() == [False, False, False, True]
    assert target.item() > 0
