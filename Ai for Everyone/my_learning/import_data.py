import h5py
import numpy as np

def load_data():
    train_data = h5py.File("./data/train_catvnoncat.h5", "r")
    test_data = h5py.File("./data/test_catvnoncat.h5", "r")

    m_train = train_data["train_set_y"].shape[0]
    m_test = test_data["test_set_y"].shape[0]

    train_set_x_orig = np.array(train_data["train_set_x"])
    train_set_y_orig = np.array(train_data["train_set_y"]).reshape((1, m_train))
    classes = np.array(train_data["list_classes"])

    test_set_x_orig = np.array(test_data["test_set_x"])
    test_set_y_orig = np.array(test_data["test_set_y"]).reshape((1, m_test))

    return(train_set_x_orig, train_set_y_orig, test_set_x_orig, test_set_y_orig, classes)