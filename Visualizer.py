import matplotlib.pyplot as plt


class Visualizer:
    def show_n_images(self, data, n_imgs=9, dims=(10, 10)):
        plt.figure(figsize=dims)
        for images, labels in data.take(1):
            for i in range(n_imgs):
                ax = plt.subplot(int(n_imgs / 3), 3, i + 1)
                plt.imshow(images[i].numpy().astype("uint8"))
                plt.title(int(labels[i]))
                plt.axis("off")
