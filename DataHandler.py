import os
import tensorflow as tf
from PIL import Image

class DataHandler:
    def __init__(self, dataset_path, dataset_name, dims, batch_size, class_names, val_split=0.2, seed=1337):
        self.dataset_path = dataset_path
        self.dataset_name = dataset_name
        self.dims = dims
        self.batch_size = batch_size
        self.class_names = class_names
        self.val_split = val_split
        self.seed = seed
        self.train_ds = None
        self.val_ds = None
        # init
        self.init_dataset()

    def filter_in(self, types, fpath):
        for ext in types:
            try:
                img = Image.open(fpath)
                exif_data = img._getexif()
                img.verify()
            except:
                os.remove(fpath)
                return 1
            return 0

    def remove_invalid(self):
        num_skipped = 0
        for folder_name in self.class_names:
            folder_path = os.path.join(self.dataset_path + self.dataset_name, folder_name)
            print(f"folder_path:\t{folder_path}")
            for fname in os.listdir(folder_path):
                if fname.endswith('.jpg'):
                    fpath = os.path.join(folder_path, fname)
                    num_skipped += self.filter_in(["jpg"], fpath)
        print("Deleted %d images" % num_skipped)

    def init_dataset(self):
        self.remove_invalid()

        self.train_ds = tf.keras.preprocessing.image_dataset_from_directory(
            os.path.join(self.dataset_path, self.dataset_name),
            validation_split=self.val_split,
            subset="training",
            seed=self.seed,
            image_size=self.dims,
            batch_size=self.batch_size,
        )
        self.val_ds = tf.keras.preprocessing.image_dataset_from_directory(
            os.path.join(self.dataset_path, self.dataset_name),
            validation_split=self.val_split,
            subset="validation",
            seed=self.seed,
            image_size=self.dims,
            batch_size=self.batch_size,
        )