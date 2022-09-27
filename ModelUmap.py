

import numpy as np
#import matplotlib.pyplot as plt
from keras import models
#from tensorflow.keras import activations, datasets, layers, losses, metrics, models, optimizers, regularizers
#import seaborn as sns
import pandas as pd
#import umap

from io import BytesIO
import base64
from PIL import Image

from bokeh import plotting, palettes
from bokeh.models import HoverTool, ColumnDataSource, CategoricalColorMapper


import umap
#import matplotlib.pyplot as plt
import warnings



warnings.filterwarnings('ignore')
#plotting.output_notebook()

class ModelUmap:
    def __int__(self, model: object, dims: object, path: object) -> object:
        self.dims = dims
        self.model = model
        self.path = path
        self.embedder = None
        self.embedding = None

    def predictUntilLayer(self, layerIndex, data):
        """ Execute prediction on a portion of the model """
        intermediateModel = models.Model(inputs=self.model.input,
                                         outputs=self.model.layers[layerIndex].output)
        return intermediateModel.predict(data)

    def embeddableImage(self, data):
        img_data = (255 * (1 - data)).astype(np.uint8)
        image = Image.fromarray(img_data, mode='L')
        buffer = BytesIO()
        image.save(buffer, format='png')
        return 'data:image/png;base64,' + base64.b64encode(buffer.getvalue()).decode()

    def umapPlot(self, embedding, x, y, yTrue=None, title=''):
        """ Plot the embedding of X and y with popovers using Bokeh """

        df = pd.DataFrame(embedding, columns=('x', 'y'))
        df['image'] = list(map(self.embeddableImage, x))
        df['digit'] = [str(d) for d in y]
        if yTrue is not None:
            df['trueDigit'] = [str(d) for d in yTrue]

        datasource = ColumnDataSource(df)

        colorMapping = CategoricalColorMapper(factors=np.arange(10).astype(np.str), palette=palettes.Spectral10)

        plotFigure = plotting.figure(
            title=title,
            plot_width=600,
            plot_height=600,
            tools=('pan, wheel_zoom, reset')
        )

        if yTrue is None:
            tooltip = """
                <div>
                    <div>
                        <img src='@image' style='float: left; margin: 5px 5px 5px 5px'/>
                    </div>
                    <div>
                        <span style='font-size: 16px; color: #224499'>Digit:</span>
                        <span style='font-size: 18px'>@digit</span>
                    </div>
                </div>
                """
        else:
            tooltip = """
                <div>
                    <div>
                        <img src='@image' style='float: left; margin: 5px 5px 5px 5px'/>
                    </div>
                    <div>
                        <span style='font-size: 16px; color: #224499'>Digit:</span>
                        <span style='font-size: 18px'>@digit (true: @trueDigit)</span>
                    </div>
                </div>
                """
        plotFigure.add_tools(HoverTool(tooltips=tooltip))

        plotFigure.circle(
            'x', 'y',
            source=datasource,
            color=dict(field='digit', transform=colorMapping),
            line_alpha=0.6, fill_alpha=0.6, size=4
        )
        plotting.show(plotFigure)

        return plotFigure


    def make_umap(self, data):
        lastTest = self.predictUntilLayer(self.model, len(self.model.layers)-2, data[0])
        denseEst = self.model.predict(data[0])

        reducer = umap.UMAP()
        embedder = reducer.fit_transform(lastTest)

        self.umapPlot(embedder, data[0], np.argmax(denseEst, axis=1), data[1],
                 title='Input of last layer of Dense network')

        #embedder = umap.UMAP()#encoder=self.model, dims=self.dims)#ParametricUMAP
        #embeddingModelLast = embedder.fit_transform(images.reshape(-1, self.dims[0]*self.dims[1]))

        embedder.save(f'{self.path}umap')

        print(embedder._history)
        #fig, ax = plt.subplots()
        #ax.plot(embedder._history['loss'])
        #ax.set_ylabel('Cross Entropy')
        #ax.set_xlabel('Epoch')