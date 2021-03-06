import tensorflow as tf
import tensorflow_hub as hub

import tensorflow.python.ops.numpy_ops.np_config as np_config

from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras import layers, Model, models, activations, optimizers, losses, metrics, callbacks

import numpy as np
import pandas as pd

import matplotlib.pyplot as plt
import matplotlib.image as mpimg

from sklearn.preprocessing import MinMaxScaler, OneHotEncoder
from sklearn.model_selection import train_test_split
from sklearn.compose import make_column_transformer
from sklearn.metrics import confusion_matrix, accuracy_score, f1_score, precision_score, recall_score

import random
import os
import sys
import pathlib
import itertools
import datetime
import time

# from tensorflow_helper import helper

class helper():
    """
    `\ffrom tensorflow_helper import helper`\n

    \n# helper:\n
    -- `helper.SetGpuLimit()` | limits the memory growth\n\n
    -- `helper.SetSeed()`     | sets the random seed for numpy and tensorflow\n
    -- `helper.EnableNumpyBehavior()` | Enables numpy behavior\n
    helper.Plot:\n
        -- `helper.Plot.PlotConfusionMatrix(y_true, y_pred)` | Plots a confusion matrix from the data\n
        -- `helper.Plot.PlotDecisionBoundory(model, x, y)`   | Plots a decision boundory from the data<classifction>\n
        -- `helper.Plot.PlotHistory(history)`                | Plots the model loss/metrics curves\n
        -- `helper.Plot.PlotLearningRate(history, epochs or lrs)` | Ploted learning rate\n
    \n\n
    helper.Image:\n
        -- `helper.Image.PlotRandomImage(path, (target) or (class_names))` | Plots a random image from the data\n\n
        -- `helper.Image.CreateGen()` | Creates and returns a gen to the data\n
        -- `helper.Image.GetDataFromGenDir(data_gen, path)` | Returns data from directory\n
    \n\n
    helper.Model:\n
        -- `helper.Model.CreateModel(model_url)` | Returns a *sequential* transfer learning model\n
        -- `helper.Model.CompileModel(model, optimzier, loss)` | Compiles the model\n
        -- `helper.Model.UnfreezeLayers(unfreeze , base_line)`  | Unfreeze's the layers of the model\n
    \n\n
    helper.Callbacks:\n
        -- `helper.Callbacks.LearningRateCallback()`                | Return's a learning rate scheduler callback\n
        -- `helper.Callbacks.CreateTensorboardCallback(path, name)` | Return's a tensorboard callback\n
        -- `helper.Callbacks.CreateCheckpointCallback(path, name)`  | Return's a checkpoint callback\n
    \n\n
    helper.Math:\n
        -- `helper.Math.DivideArray(percentage, array)`    | Return's 2 arrays diveided by the percentage\n
        -- `helper.Math.CalculateModelResults(y_true, y_pred)`| Return's a dictionary list of: accuracy, precision, recall, f1-score\n
    helper
    """

    __version__ = f"\nHelper class version is: {0.40}v\n";

    class Image():
        """
        helper.Image:\n
            `helper.Image.PlotRandomImage(path, (target) or (class_names))` | Plots a random image from the data\n\n
            `helper.Image.CreateGen()` | Creates and returns a gen to the data\n
            `helper.Image.GetDataFromGenDir(data_gen, path)` | Returns data from directory\n
            `helper.Image.GetClassNames()` | Returns an array of all the files in the path\n
        \n\n
        """

        def PlotRandomImage(path=None, target=None, class_names=None, y_pred=None, y_prob=None, images=None, labels=None, images_recalse=None,index=None, data_edited=None,loop=1, loop_forever=False):
            """
            Requirements:\n

                path        | the path of the data (ex: "data/train")\n
                target      | the target of the data (ex: "pizza")\n
                    or      | set value for *target* or *class_names*\n
                class_names | names of all the classes\n

            Extra:\n
                
                y_pred        | required: class_names and images -> adds predictions of the model into the plot\n
                y_prob        | required: y_pred -> adds statistical chance of the model predictions\n
                images        | will use images as numbers to plot the image\n
                labels        | if images then labels need a value\n
                images_rescale| rescales the images -> defult: 1 if y_pred is false -> else 1./255 (ex: 255. or 1/255.)\n
                index         | set a chosen index -> defult: index will be randomly chosen\n
                data_edited   | if image was edited via layers | set the value to 'data_edited'\n
                loop          | number of images to show\n
                loop_forever  | if True -> will show inf images\n

            Returns:\n

                `Ploted image from the data`\n
            """
            if(images is None or y_pred is None):
                assert not(path is None), "\nPlotRandomImage( **path is missing** ) \npath | the path of the data (ex: 'data/train')\n";
            if(y_pred is not None and images is None or class_names is None):
                assert False, "\nPlotRandomImage( **images/class_names is missing** )\n";
            if(y_prob is not None and y_pred is None):
                assert False, "\nPlotRandomImage( **y_pred is missing** ) y_pred | required: class_names and images -> adds predictions of the model into the plot\n";

            if(y_pred is None and images is not None):
                images_recalse = 1.;
            elif(y_pred is not None):
                images_recalse = 1/255.;

            if(loop_forever):
                loop=sys.maxsize;
            
            def setImage(data, returnIndex=False):
                if(returnIndex):
                    if(index is None):
                        return random.randint(0, len(data) - 1);
                    return index;
                if(index is None):
                    return random.choice(data);
                return data[index];

            def setTitle(target, use_data_edited=False, index=None):
                if(y_prob is not None and y_pred is not None):
                    return f"pred: {class_names[y_pred[index]]} | prob: {y_prob[y_pred[index]]}";
                if(y_pred is not None):
                    return f"pred: {class_names[y_pred[index]]}";
                if(use_data_edited):
                    return f"True label: {target}";
                return f"Edited image: {target}";

            def show():
                plt.axis("off");
                plt.show();

            def plot(img, target, index=None):
                plt.matshow(img);
                plt.title(setTitle(target, index=index));
                show();
                if(data_edited is not None):
                    plt.matshow(tf.squeeze(data_edited(tf.expand_dims(img, axis=0))));
                    plt.title(setTitle(target, True, index));
                    show();

            if((path is not None) and (((target is not None) or (class_names is not None))) and (images is None) and (y_pred is None)):
                for j in range(loop):
                    if(class_names is not None):
                        target = random.choice(class_names);

                    target_path = f"{path}\{target}";
                    images_path = os.listdir(target_path);

                    chosen_image = setImage(images_path);

                    img = mpimg.imread(f"{target_path}/{chosen_image}");

                    plot(img, target);

            elif(images is not None):
                for j in range(loop):               
                    i = setImage(images, returnIndex=True);

                    if(y_pred is not None):
                        plot(images[i] / images_recalse, class_names[y_pred[i]], index);
                    elif(labels is None):
                        plot(images[i] / images_recalse, i, i);
                    else:
                        plot(images[i] / images_recalse, class_names[tf.argmax(labels[i], 0)], i);


        def CreateGen(edited=False, rescale=1./255, depth=0.2, rotation_range=None, shear_range=None, zoom_range=None, height_shift_range=None, width_shift_range=None, horizontal_flip=True):
            """
            Extra:\n

                edited             |  if True -> will edit the picture with: rotation_range, shear_range, ...\n
                rescale            |  rescale of image<normalized> -> defult: 1./255\n
                depth              |  will set the value of all variabels <that are None> to depth -> defult: 0.2\n
                rotation_range     |  change the rotation of the image\n
                shear_range        |  change the shear of the image\n
                zoom_range         |  change the zoom of the image\n
                height_shift_range |  shift the heigth of the image\n
                width_shift_range  |  shift the width of the image\n
                horizontal_flip    |  flip the image horizontaly\n

            Returns:\n

                `Creates and returns a gen to the data`\n
            """
            if(edited):
                def setVar(var=None):
                    if(var is None):
                        return depth;
                
                rotation_range = setVar(rotation_range);
                shear_range = setVar(shear_range);
                zoom_range = setVar(zoom_range);
                height_shift_range = setVar(height_shift_range);
                width_shift_range = setVar(width_shift_range);

                return ImageDataGenerator(
                    rescale=rescale,
                    rotation_range=rotation_range,
                    shear_range=shear_range,
                    zoom_range=zoom_range,
                    height_shift_range=height_shift_range,
                    width_shift_range=width_shift_range,
                    horizontal_flip=horizontal_flip
                );
            return ImageDataGenerator(
                rescale=rescale
                );

        def CreateEditedLayer(rescale=1., only_rescale=False, depth=0.2, rotation_range=None, zoom_range=None, height_shift_range=None, width_shift_range=None, flip="horizontal"):
            """
            Extra:\n

                rescale            |  rescale of image<normalized> -> defult: 1.\n
                only_rescale       |  if True -> will only rescale the image\n
                depth              |  will set the value of all variabels <that are None> to depth -> defult: 0.2\n
                rotation_range     |  change the rotation of the image\n
                zoom_range         |  change the zoom of the image\n
                height_shift_range |  shift the heigth of the image\n
                width_shift_range  |  shift the width of the image\n
                flip               |  defult: "horizontal" -> flip's the image horizontaly\n

            Returns:\n

                `Creates and returns an edited layer`\n
            """
            if(not(only_rescale)):
                def setVar(var=None):
                    if(var is None):
                        return depth;
                
                rotation_range = setVar(rotation_range);
                zoom_range = setVar(zoom_range);
                height_shift_range = setVar(height_shift_range);
                width_shift_range = setVar(width_shift_range);

                return models.Sequential([
                    layers.experimental.preprocessing.Rescaling(rescale),
                    layers.experimental.preprocessing.RandomFlip(flip),
                    layers.experimental.preprocessing.RandomRotation(rotation_range),
                    layers.experimental.preprocessing.RandomZoom(zoom_range),
                    layers.experimental.preprocessing.RandomHeight(height_shift_range),
                    layers.experimental.preprocessing.RandomWidth(width_shift_range)
                ]);
            return models.Sequential([
                    layers.experimental.preprocessing.Rescaling(rescale)
                ]);



        def GetDataFromGenDir(data_gen, path, batch_size=32, target_size=(224,224), class_mode="categorical", seed=42):
            """
            Requirements:\n

                data_gen | ImageDataGenerator --> use helper.Image.CreateGen(**args**) to create a gen\n
                path     | path of data\n

            Extra:\n

                batch_size   |  the batch size of the images -> defult: 32\n
                target_size  |  the the target size of the images -> defult: (<height>224, <wdith>224)\n
                class_mode   |  "sparse", "categorical", "binary", "input" ,"None" -> defult: "categorical"\n
                seed         |  seed of the data -> defult: 42\n

            Returns:\n

                Returns data from directory\n
            """
            return data_gen.flow_from_directory(
                directory=path,
                batch_size=batch_size,
                target_size=target_size,
                class_mode=class_mode,
                seed=seed);
        

    class Model():
        """
        helper.Model:\n
            helper.Model.CreateModel(model_url)               | Returns a *sequential* transfer learning model\n
            helper.Model.CompileModel(model, optimzier, loss) | Compiles the model\n
            helper.Model.UnfreezeLayers(freeze , base_line)   | Unfreeze's the layers of the model <tf.keras.applcations>\n
        """

        def CreateModelURL(model_url=None, input_shape=None, n_classes=None, activation=None, trainable=False, build=False):
            """
            Requirements:\n

                model_url   | model link from: tenosrflow-hub\n
                input_shape | **array** of input shape of model\n
                n_classes   | number of classes for the output layer(length of classes names)\n

            Extra:\n
                
                activation  | activation methed of output layer -> defult: sigmoid/softmax <based on n_classes>\n
                trainable   | control if the layer is trainable of not -> defult: False\n
                build       | some models require build --> if they do: set build to True -> defult:False\n

            Returns:\n

                `Returns a *sequential* transfer learning model`\n
            """
            assert not(model_url is None), "\n**model_url is missing** \nmodel_url | model link from: tenosrflow-hub\n";
            assert not(input_shape is None), "\n*input_shape is missing** \ninput_shape | input shape of model\n";
            assert not(n_classes is None),  "\n**n_classes is missing** \nn_classes   | number of classes for the output layer(length of classes names)\n";

            if((activation is None)):
                if(n_classes < 2):
                    activation = activations.sigmoid;
                else:
                    activation = activations.softmax;

            def setLayer():
                if(not(build)):
                    return hub.KerasLayer(
                        model_url,
                        trainable=False,
                        name="keras_layer",
                        input_shape=input_shape);
                return hub.KerasLayer(
                    model_url,
                    trainable=False,
                    name="keras_layer");

            model = models.Sequential([
                setLayer(),
                layers.Dense(n_classes, activation=activation)
            ]);

            if(build):
                model.build(input_shape);

            return model;


        def CompileModel(model=None, optimizer=optimizers.Adam(), loss="categorical"):
            """
            Requirements:\n

                model     | functional API or sequential API\n
                optimizer | optimizer -> defult: tf.keras.optimizer.Adam()\n
                loss      | "mae", "mse", "binary", "categorical", "sparse"\n

            Returns:\n

                `Compiles the model`\n
            """
            assert not(model is None), "CompileModel( **model is None** ) | API or Sequential\n";

            def setLossAndMetrics():
                if(loss == "mae"):
                    return losses.mae, metrics.mae;
                elif(loss == "mse"):
                    return losses.mse, metrics.mse;
                elif(loss == "binary"):
                    return losses.BinaryCrossentropy(), ["accuracy"];
                elif(loss == "categorical"):
                    return losses.CategoricalCrossentropy(), ["accuracy"];
                elif(loss == "sparse"):
                    return losses.SparseCategoricalCrossentropy(), ["accuracy"];

                assert False, "CompileModel(loss = Faild to find loss) | 'mae', 'mse', 'binary', 'categorical', 'sparse'";
            
            model_loss, model_metrics = setLossAndMetrics();

            model.compile(
                optimizer=optimizer,
                loss=model_loss,
                metrics=model_metrics
                );

        def UnfreezeLayers(unfreeze=20, base_line=None, print_layers=False):
            """
            Requirements:\n

                unfreeze  | the percentage of the amount of layers that will be unfreezed (ex: 20 will be 20%)\n
                base_line | the output of tf.keras.appliactions.//\n

            Extra:\n

                print_layers | prints layers info -> defult: False\n

            Returns:\n

                `Unfreeze's the layers of the model <tf.keras.applcations>`\n\n
            """
            assert not(unfreeze is None),  "UnfreezeLayers( **freeze is None** ) \nunfreeze  | the percentage of the amount of layers that will be unfreezed\n";
            assert not(base_line is None), "UnfreezeLayers( **base_line is None** ) \nbase_line | the output of tf.keras.appliactions.//\n";

            base_line.trainable = True;

            if(unfreeze > 1):
                unfreeze /= 100;
            elif((unfreeze < 0) or (unfreeze > 100)):
                assert False, "\n**unfreeze** needs to be between: 0-1 | or | 0-100\n";
            
            freeze = int((1 - unfreeze) * (len(base_line.layers)));

            for layers in base_line.layers[freeze:]:
                layers.trainable = False;
            
            if(print_layers):
                for i, layer in enumerate(base_line.layers):
                    print(f"{i} | {layer.name} | {layer.trainable}");

    class Plot():
        """
        \nhelper plot:\n
            `helper.Plot.PlotConfusionMatrix(y_true, y_pred)`     | Plotes a confusion matrix from the data\n
            `helper.Plot.PlotDecisionBoundory(model, x, y)`       | Plots a decision boundory from the data<classifction>\n
            `helper.Plot.PlotHistory(history)`                    | Ploted model loss/metrics curves\n
            `helper.Plot.PlotLearningRate(history, epochs or lrs)`| Ploted learning rate\n
        """

        def PlotConfusionMatrix(y_true=None, y_pred=None, class_names=None, figsize=(10, 7), size=17, text_size=10, show_text=True, show_text_norm=False, save_png_path=None):
            """
            Requirements:\n

                y_true | true label\n
                y_pred | predicted label\n

            Extra:\n

                class_names    | list of class names in the data set | if class_names is None -> will use an array of numbers\n
                figsize        | plot window heigth and length | defult: (10, 7)\n
                size           | size of title/axis text\n
                text_size      | size of text in the confusion matrix\n
                show_text      | if True -> will show text on confusion matrix\n
                show_text_norm | if True -> will show norm_text with text\n
                save_png_path  | will save the image on the given path(ex: "file/confusion_matrix") | defult: None\n

            Returns:\n

                `Ploted confusion matrix of the data set`\n
            """
            #if y_true or y_pred is None | return error
            assert not(y_true is None), "\n**y_true is missing** \ny_true | true label\n";
            assert not(y_pred is None), "\n**y_pred is missing** \ny_pred | predicted label\n";

            y_pred = tf.round(y_pred);

            cm = confusion_matrix(y_true, y_pred);
            cm_norm = cm.astype("float") / cm.sum(axis=1)[:, np.newaxis];

            n_classes = cm.shape[0];

            fig, ax = plt.subplots(figsize=figsize);

            cax = ax.matshow(cm, cmap=plt.cm.Blues);
            fig.colorbar(cax);

            def setLabel():
                if(class_names is None):
                    return np.arange(n_classes);
                return class_names;

            ax.set(
                title="Confusion Matrix",
                xlabel="Predicted label",
                ylabel="True label",
                xticks=np.arange(n_classes),
                yticks=np.arange(n_classes),
                xticklabels=setLabel(),
                yticklabels=setLabel()
            );

            ax.xaxis.tick_bottom();

            ax.xaxis.label.set_size(size);
            ax.yaxis.label.set_size(size);
            ax.title.set_size(size);

            plt.xticks(rotation=70);
            
            def setText(i, j):
                if(show_text_norm):
                    return f"{cm[i,j]}({cm_norm[i, j]}%)"
                return f"{cm[i,j]}";

            def setTextColor(i, j):
                thresehold = (cm.max() + cm.min()) / 2;
                if(thresehold > cm[i, j]):
                    return "black";
                return "white";

            if(show_text):
                for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
                    plt.text(
                        j, i, setText(i, j),
                        horizontalalignment="center",
                        size=text_size,
                        color=setTextColor(i, j)
                        );

            plt.show();

            if(save_png_path is not None):
                fig.savefig(f"{save_png_path}.png");

        
        def PlotDecisionBoundory(model, x, y, cmap=plt.cm.RdYlBu):
            """
            Requirements:\n

                model | using the model to make predictions\n
                x     | data\n
                y     | labels\n

            Extra:\n

                cmap | color of the graph -> defult: plt.cm.RdYlBu\n

            Returns:\n

                `Ploted model predictions <with classification>`\n
            """
            assert not(model is None), "\n**model is missing** \nmodel | using the model to make predictions\n";
            assert not(x is None), "\n**x is missing** \nx | data\n";
            assert not(y is None), "\n**y is missing** \ny | label\n";

            x_min, x_max = min(x[:, 0]) - 0.1, max(x[:, 0]) + 0.1;
            y_min, y_max = min(x[:, 1]) - 0.1, max(x[:, 1]) + 0.1;

            xx, yy = np.meshgrid(
                np.linspace(x_min, x_max, 100),
                np.linspace(y_min, y_max, 100)
            );

            x_in = np.c_[xx.ravel(), yy.ravel()];

            predict = model.predict(x_in);

            if(len(predict[0]) > 1):
                print("doing multiclass-classifiction!");
                predict = np.argmax(predict, axis=1).reshape(xx.shpae);
            else:
                print("doing binary-callsifiction!");
                predict = np.round(predict).reshape(xx.shape);
            
            plt.contourf(xx, yy, predict, alpha=0.7, cmap=cmap);
            plt.scatter(x[:, 0], x[:, 1], c=y, s=40, cmap=cmap);
            plt.xlim(xx.min(), yy.max());
            plt.ylim(yy.min(), yy.max());
            plt.show();
        
        def PlotHistory(history=None, save_path=None, show=True):
            """
            Requirements:\n

                history | value of model.fit (ex: history = model.fit(...))\n
                save_path | the path to save the data (if None will not save)\n
                show | use plt.show? defult is True\n

            Returns:\n

                `Ploted model loss/metrics curves`\n
            """
            assert not(history is None), "\n**history is missing** \nhistory | value of model.fit (ex: history = model.fit(...))\n";

            pd.DataFrame(history.history).plot();
            
            if(show):
                plt.show();

            if(save_path is not None):
                plt.savefig(f"{save_path}.png");

        def PlotLearningRate(history=None, epochs=None, lrs=None):
            """
            Requirements:\n

                history | value of model.fit (ex: history = model.fit(...))\n
                epochs  | number of epochs that the model has trained on\n
                    or\n
                lrs     | set a lrs manually -> if (epochs is not None) -> lrs = 1e-4 * 10 ** (tf.range(epochs) /20)\n

            Returns:\n

                `Ploted learning rate`\n
            """
            assert not(history is None), "\n**history is missing** \nhistory | value of model.fit (ex: history = model.fit(...))\n";
            if(lrs is not None):
                assert not(epochs is None), "\n**epochs/lrs is missing** \n";

            def setLRS():
                if(lrs is None):
                    return 1e-4 * 10 ** (tf.range(epochs) / 20);
                return lrs;
            
            plt.semilogx(setLRS(), history.history["loss"]);
            plt.show();
    
    class Callbacks():

        """
        helper.Callbacks:\n
            `helper.Callbacks.LearningRateCallback()`                | Return's a learning rate scheduler callback\n
            `helper.Callbacks.CreateTensorboardCallback(path, name)` | Return's a tensorboard callback\n
            `helper.Callbacks.CreateCheckpointCallback(path, name)`  | Return's a checkpoint callback\n
        """


        def CreateTensorboardCallback(path=None, path_name=None, time=datetime.datetime.now().strftime("%Y-%m-%d-%H-%M")):
            """
            Requirements:\n

                path      | folder_path(ex: folder/tensorboard)\n
                path_name | name of folder in path(ex: res_net_v2_50)\n

            Extra:\n

                time | defult: year-month-day-hour-min -> can be set manually\n

            Returns:\n

                `Return's a tensorboard callback`\n
            """
            assert not(path is None), "\n**path is missing** \npath | folder_path(ex: folder/tensorboard)\n";
            assert not(path_name is None), "\n**path is missing** \npath_name | name of folder in path(ex: res_net_v2_50)\n";

            target_path = f"{path}/{path_name}/{time}";
 
            return callbacks.TensorBoard(target_path);

        def CreateCheckpointCallback(path=None, name=None, class_mode=None, get_checkpoint_path=False, save_weights_only=True, save_best_only=True):
            """
            Requirements:\n

                path | folder_path(ex: folder/chekcpoint)\n

            Extra:\n

                name                | recommended: path name -> name of folder inside the file<path>(ex: res_net_v2_50)\n
                class_mode          | "None" | "mae" | "mse" | "accuracy" | "categorical" | "sparse" | "rmse" | custom...\n
                get_checkpoint_path | if True -> returns the checkpoint path via path and returns the value\n
                save_weights_only   | defult and recommended: True\n
                save_best_only      | save only the best epoch <via val_loss> -> defult: True\n

            Returns:\n

                `Return's a checkpoint callback`\n
            """
            assert not(path is None), "CreateCheckpointCallback( **path is missing** ) \npath | folder_path(ex: folder/chekcpoint)\n";
            
            def setMonitor():
                if(class_mode == None or class_mode == "None" or class_mode == "mae" or class_mode == "mse"):
                    return "val_loss";
                elif(class_mode == "accuracy"):
                    return "val_accuracy";
                elif(class_mode == "categorical"):
                    return "val_categorical_accuracy";
                elif(class_mode == "sparse" or class_mode == "int"):
                    return "val_sparse_categorical_accuracy";
                elif(class_mode == "rmse"):
                    return "val_root_mean_squared_error";
                else:
                    return class_mode;

            def GetPath():
                if(name is None):
                    return f"{path}.ckpt";
                return f"{path}/{name}.ckpt";
            
            target_path = GetPath();

            if(get_checkpoint_path):
                return target_path;

            return callbacks.ModelCheckpoint(
                target_path,
                verbose=1,
                save_weights_only=save_weights_only,
                save_best_only=save_best_only,
                save_freq="epoch",
                monitor=setMonitor());
        
        def LearningRateCallback(lrs=lambda epoch: 1e-4 * 10 ** (epoch / 20)):
            """
            Extra:\n
                lrs | Learning rate of the callback -> defult: 1e-4 * 10 ** (epoch / 20)\n

            Returns:\n

                `Return's a learning rate scheduler callback`\n
            """
            return callbacks.LearningRateScheduler(lrs);


    class Math:
        """
        \nhelper.Math:\n
            `helper.Math.DivideArray(percentage, array)`       | Return's 2 array's diveided by the percentage\n
            `helper.Math.CalculateModelResults(y_true, y_pred)`| Return's a dictionary list of: accuracy, precision, recall, f1-score\n
        """
        def DivideArray(percentage=None, array=None):
            """
            Requirements:\n

                percentage | value bettewn 0 - 1 as a float or 1 - 100 as an integer\n
                array      | numpy/python array\n

            Returns:\n

                `Return's 2 arrays diveided by the percentage`\n
            """
            assert not(percentage is None), "DivideArray( **percentange is None** ) \npercentage | value bettewn 0 - 1 as a float or 1 - 100 as an integer\n";
            assert not(array is None), "DivideArray( **array is None** ) \narray | numpy/python array\n";

            if(percentage > 100 or percentage < 0):
                assert False, "*percentage* need's to be set between 0 - 1 | or | 1 - 100";
            elif(percentage >= 1):
                percentage /=100;
            
            num_of_values = int(percentage * len(array));

            return array[:num_of_values], array[num_of_values:];

        def CalculateModelResults(y_true=None, y_pred=None, mode="binary"):
            """
            Requirements:\n

                y_true | a list/np-array/tenosr of true labels\n
                y_pred | a list/np-array/tenosr of predicted labels(as integages)\n

            Extra:\n

                mode | "binary", "categorical"\n

            Returns:\n

                `Return's a dictionary list of: accuracy, precision, recall, f1-score`\n
            """
            assert not(y_true is None), "helper.Math.CalculateModelResults( **y_true is None** ) \ny_true | a list/np-array/tenosr of true labels\n";
            assert not(y_pred is None), "helper.Math.CalculateModelResults( **y_pred is None** ) \ny_pred | a list/np-array/tenosr of predicted labels(as integages)\n";

            def SetAverage():
                if(mode == "binary"):
                    return "binary";
                elif(mode == "categorical"):
                    return "micro";
                assert False, "Mode was not found \nmode | 'binary', 'categorical'\n";
            return {
                "accuracy": accuracy_score(y_true, y_pred),
                "precision": precision_score(y_true, y_pred, average=SetAverage()),
                "recall": recall_score(y_true, y_pred, average=SetAverage()),
                "f1-score": f1_score(y_true, y_pred, average=SetAverage())
            };

    class Time:
        """
        Time:\n

            `helper.Time.timer` | a class to count time\n
        """
        class Timer:
            """
            helper.Time.Timer:\n
                helper.Time.Timer.CreateTimer() | Creates and returns a new timer object\n
                helper.Time.Timer.GetTime() | Amount of time that passed since timer was created\n

            ## Example of use:

                timer = helper.Time.Timer();\n
                timer_0 = timer.CreateTimer();\n
                timer.Sleep(0.25);\n
                total_time = timer.GetTime(timer_0);\n
                -- total time is 0.25\n
            """
            def __init__(self):
                self.times = [];
                
            def CreateTimer(self):
                """
                Returns:\n

                    `Creates and returns a new timer object`\n
                """
                start_time = time.time();
                self.times.append(start_time);
                return len(self.times)-1;
                
            def GetTime(self, index=0, round_time=True, convert_to_string=False):
                """
                Args:\n

                    index | output of timer.CreateTimer()
                    round_time | round output -> example: 0.25005353 into 0.25 | defult: True
                    convert_to_string | return output as string -> example: 0.25 into "0.25s", defult: False

                Returns:\n

                    `Amount of time that passed since timer was created`\n
                """
                current_time = time.time();
                result_time = current_time - self.times[index];
                if(round_time):
                    result_time = round(result_time, 2);
                if(convert_to_string):
                    result_time = f"{result_time}s";
                return result_time;
                
            def Sleep(self, sec=None):
                """
                Args:\n

                    sec | Stop time for ... sec\n

                Returns:\n

                    `Freezing time for a number of sec -> time.sleep(sec)`\n
                """
                time.sleep(sec);

    def SetGpuLimit(condition=True):
        """
        Extras:\n

            condition | if False: there will be no limit for the poor RAM -> defult: True\n

        Returns:\n

            `limits the memory growth`\n
        """
        gpu = tf.config.list_physical_devices("GPU");
        try:
            tf.config.experimental.set_memory_growth(gpu[0], condition);
            print(f"\nset_memory_growth is: {condition}\n");
        except:
            print("\n-Invalid device or cannot modify virtual devices once initialized-\n");
            pass;


    def SetSeed(seed=42):
        """
        Extras:\n

            seed | set the value of the seed -> defult: 42

        Returns:\n
            `sets the random seed for numpy and tensorflow`\n
        """
        tf.random.set_seed(seed);
        np.random.seed(seed);

    def EnableNumpyBehavior():
        """
        Returns: \n
            `Enables numpy behavior  |  from:` \n
            `tensorflow.python.ops.numpy_ops.np_config`
        """
        np_config.enable_numpy_behavior();


    def Set(condition=True, seed=42):
        """
        Calles the functions:\n
            `helper.SetGpuLimit()`\n
            `helper.SetSeed()`\n
            `helper.EnableNumpyBehavior()`\n
        """
        helper.SetGpuLimit(condition);
        helper.SetSeed(seed);
        helper.EnableNumpyBehavior();

    class Experimental:
        """
        \nhelper.Experimental -> experiencing new functions\n
            helper.Experimental.Image\n
                `helper.helper.Experimental.Image.GetClassNames(path)` |  Returns a list of all the files in path\n
        """
        class Image:
            def GetClassNames(path, show_class_names=False):
                """
                Requirements:\n

                    path | path of folder location\n

                Extra:\n

                    show_class_names | prints the returend variable {class_names}\n

                Returns:\n

                    `Returns an array of all the files in the path`\n
                """
                assert not(path is None), "GetClassNames( **path is missing** ) \npath | path of folder location\n";

                class_names = np.array(os.listdir(path));

                if(show_class_names):
                    print(class_names);

                return class_names;