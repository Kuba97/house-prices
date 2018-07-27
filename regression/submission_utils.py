import os
from datetime import datetime

PROJECT_FOLDER = os.path.join(os.getcwd(), '..')
SUBMISSION_FOLDER = 'submissions'
SUBMISSION_FOLDER_DIR = os.path.join(PROJECT_FOLDER, SUBMISSION_FOLDER)
SUBMISSION_NAME_SEP = ' '
SUBMISSION_EXTENSION = '.csv'


# todo: refine formatting file names (not necessary)
def save_submission(predictions, model_name, val_rme_error):
    filename = model_name + SUBMISSION_NAME_SEP + str(datetime.now().date()) + SUBMISSION_NAME_SEP + "{:.4f}".format(
        val_rme_error) + SUBMISSION_EXTENSION
    predictions.to_csv(os.path.join(SUBMISSION_FOLDER_DIR, filename))
