import os
import pickle as pkl

PATH_UP = '..'


def pickle_file(obj, filename, path):
    path = os.path.join(path, filename)
    with open(path, 'wb') as file:
        pkl.dump(obj, file, protocol=pkl.HIGHEST_PROTOCOL)


def unpickle_file(path):
    with open(path, 'rb') as file:
        loaded = pkl.load(file)
    return loaded
