from utils.make_dataset import load_split_dataset
from features.features_info import ID_COL, TARGET_COL
from models.model_selection.select import validate
from features.transformers import LogTransform

if __name__ == '__main__':
    x, y, _ = load_split_dataset('raw', 'train.csv', TARGET_COL, ID_COL)
    validate(x, LogTransform().transform(y))
