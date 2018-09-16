from data_io.make_dataset import Data
from features.features_info import ID_COL, TARGET_COL
from models.validation_utils.model_selection import validate

if __name__ == '__main__':
    data = Data('raw', 'train.csv', TARGET_COL, ID_COL)
    validate(data)
