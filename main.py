'''
classification on cats vs dogs data
'''
import datetime
import os

import numpy as np
import tensorflow as tf
from tensorflow import keras

from DataHandler import DataHandler
#from Visualizer import Visualizer
from ResNet import ResNet
from Mlp import Mlp
import matplotlib.pyplot as plt
#from ModelUmap import ModelUmap
import statistics as stats

import scikitplot as skplt

if __name__ == '__main__':
    dims = (180, 180)
    batch_size = 32
    epochs = 10
    path = os.getenv("HOME") + "/Desktop/datasets/cats_dogs/"
    classes = ['Cat', 'Dog']

    handler = DataHandler(dataset_path=path, dataset_name='PetImages', dims=dims, batch_size=batch_size, class_names=classes, val_split=0.2, seed=1337)
    #image_viewer = Visualizer()
    #image_viewer.show_n_images(data=handler.train_ds)      ### NOTHING POPPED UP ON DISPLAY


    handler.train_ds = handler.train_ds.prefetch(buffer_size=32)
    handler.val_ds = handler.val_ds.prefetch(buffer_size=32)

    my_model = Mlp()#ResNet()#
    my_model.model_create(dims=dims + (3,), num_classes=2)
    print(f"input shape:\t{dims+(3,)}")
    #keras.utils.plot_model(my_model.model, show_shapes=True)

    callbacks = [
        keras.callbacks.TensorBoard(
            log_dir=(f"{path}logs/scalars/" + datetime.datetime.now().strftime("%Y%m%d-%H%M%S")),
            write_graph=True,
            write_images=False,
            write_grads=1,
            write_steps_per_second=False,
            update_freq="epoch",
            profile_batch=0,
            embeddings_freq=0,
            embeddings_metadata=None,
        ),
        keras.callbacks.ModelCheckpoint("save_at_{epoch}.h5"),
    ]

    my_model.model.compile(
        optimizer=keras.optimizers.Adam(1e-3),
        loss="binary_crossentropy",
        metrics=["accuracy"],
    )
    my_model.model.fit(
        handler.train_ds, epochs=epochs, callbacks=callbacks, validation_data=handler.val_ds,
    )

    img = keras.preprocessing.image.load_img(
        f"{path}PetImages/Cat/6779.jpg", target_size=dims
    )
    img_array = keras.preprocessing.image.img_to_array(img)
    img_array = tf.expand_dims(img_array, 0)  # Create batch axis
    predictions = my_model.model.predict(img_array)
    score = predictions[0]
    print(
        "This image is %.2f percent cat and %.2f percent dog."
        % (100 * (1 - score), 100 * score)
    )

    # get confusion matrix for test data
    images = list(handler.val_ds.map(lambda x, y: x))
    images = tf.concat([mini_batch for mini_batch in images], 0)

    labels = list(handler.val_ds.map(lambda x, y: y))
    labels = tf.concat([mini_batch for mini_batch in labels], 0)

    predictions = my_model.model.predict_on_batch(images)
    predictions = [round(pred[0]) for pred in predictions]

    skplt.metrics.plot_confusion_matrix(labels, predictions, normalize=True) # cat:0, dog:1
    plt.savefig(f"{path}confusion_matrix.png")

    score = [1 if real == pred else 0 for pred, real in zip(predictions, labels)]
    print(f"score:\t{score}")
    average_acc = stats.mean(score)
    print(f"final accuracy:\t{average_acc*100} %")
    
    # create umap
    #umap_test = handler.val_ds.unbatch()
    #images = list(handler.val_ds.map(lambda x, y: x))
    #labels = list(handler.val_ds.map(lambda x, y: y))

    #umap_visual = ModelUmap(model=my_model.model, dims=dims, path=path)
    #umap_visual.make_umap(data=(images, labels))
    
