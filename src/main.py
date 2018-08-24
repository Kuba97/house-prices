from data_io.make_dataset import Data
from models.validation import run_validation
from features_info import ID_COL, TARGET_COL

RAW_DATA = 'raw'
TRAIN_DATA = 'train.csv'
TEST_DATA = 'test.csv'


if __name__ == '__main__':
    train = Data(RAW_DATA, TRAIN_DATA, ID_COL, TARGET_COL)
    run_validation(train)
