import functools
import h5py
import os
import shutil
import pickle

import matplotlib.pyplot as plt
import numpy as np
from PIL import Image


def load_image(file_name: str):
    img = Image.open(file_name)
    img.load()
    data = np.asarray(img, dtype="uint8")

    return data


def load_images(file_names: list):
    return [load_image(file_name) for file_name in file_names]


def save_image(np_data, out_file_name):
    img = Image.fromarray(np.asarray(np.clip(np_data, 0, 255), dtype="uint8"), "L")
    img.save(out_file_name)


def img_show(img, normalize=False):
    if normalize:
        img_max, img_min = np.max(img), np.min(img)
        img = 255.0 * (img - img_min) / (img_max - img_min)
    plt.imshow(img.astype('uint8'))
    plt.gca().axis('off')


def save_to_hdf5(images, labels, out_path, out_name):
    """ Stores an array of images to HDF5.
        Parameters:
        ---------------
        images       images array, (N, 32, 32, 3) to be stored
        labels       labels array, (N, 1) to be stored
    """

    if labels is None:
        x_set_name = "images_unlabeled"
    else:
        x_set_name = "images_labeled"

    with h5py.File(os.path.join(out_path, out_name + '.h5'), "w") as file:

        x_set = file.create_dataset(
            x_set_name, np.shape(images), h5py.h5t.STD_U8BE, data=images
        )

        if labels is not None:
            y_set = file.create_dataset(
                "labels", np.shape(labels), h5py.h5t.STD_U8BE, data=labels
            )


def unpickle(file):
    with open(file, "rb") as fo:
        dict_ = pickle.load(fo, encoding="bytes")
    return dict_


def full_indices(indices, full_length=6):
    res = []

    for index in indices:
        index_str = str(index)

        res.append('0' * (full_length - len(index_str)) + index_str + ".jpg")

    return res


def move_images_decorator(func):
    @functools.wraps(func)
    def wrapper(*args, **kwargs):
        paths = func(*args, **kwargs)

        dir_name = os.path.join(os.getcwd(), "labeled_images_names")

        if os.path.exists(dir_name) is False:
            os.mkdir(dir_name)

        with open(os.path.join(dir_name, "names.txt"), "w") as f:
            for path in paths:
                f.write(path + "\n")

    return wrapper


@move_images_decorator
def move_n_images(dir_path, dest_dir_path, n_all=4000, n_move=500, shuffle=True):
    """
        Moves n_move images from source directory to dest directory
        Example of usage: move_500_random("/path/to/source/dir/", "/path/to/dest/dir")

        return: list of indices of moved images
    """

    indices = np.arange(n_all)

    if shuffle:
        np.random.shuffle(indices)

    indices = indices[:n_move]
    indices = np.sort(indices)

    f_indices = full_indices(indices)

    for index in f_indices:
        full_path_before = os.path.join(dir_path, index)
        full_path_after = os.path.join(dest_dir_path, index)

        shutil.move(full_path_before, full_path_after)

    return f_indices

