import matplotlib.pyplot as plt
import seaborn as sns
plt.style.use('ggplot')
plt.rcParams['font.size'] = 15

import tensorflow as tf
import tensorflow.keras as keras
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.layers import Embedding
from keras.layers import Flatten

# construct your deep_leraning model
def get_deep_learning_model(params):
    # Define your model using the model params
    pass
    return model

def get_model_performance(best_model_file_name, patience, epoch, params):
    """ 
    Typically I alter patience
    Attributes
    ----------
    best_model_file_name : str
        File name to save the model
    patience : int
        number of epoch to try for improvment after the last model improvement
    epoch : int
        number of epoch in total
    params : <this is a placeholder for your model paramteres you might 
              want to use while defining the model in get_deep_learning_model>
    """
    keras.backend.clear_session()
    model = get_deep_learning_model(params)
    callbacks=[
        keras.callbacks.EarlyStopping(monitor="val_loss", 
                                      patience=patience, 
                                      verbose=1, 
                                      mode="min", 
                                      restore_best_weights=True),
        keras.callbacks.ModelCheckpoint(filepath=best_model_file_name, 
                                        verbose=1, 
                                        save_best_only=True)]
    model.compile(loss='binary_crossentropy',
                  optimizer='adam', 
                  metrics=[tf.keras.metrics.Precision(), tf.keras.metrics.Recall()])    
    history = model.fit(X_train, 
                        y_train, 
                        epochs=20,
                        validation_data=0.2, 
                        callbacks=callbacks)
    return model, history                
                    
def plot_performance(history):
    """ 
    Plots the Training and Validation Loss
    
    Attributes
    ----------
    history : Keras History Object
        History.history attribute is a record of training loss values 
        and metrics values at successive epochs, as well as validation 
        loss values and validation metrics values
    """
    len_keys = len(list(history.history.keys()))
    for i in range(0, len_keys/2):  
        metric_to_plot = list(history.history.keys())[i]
        val_metric_to_plot = list(history.history.keys())[i+3]
    plt.plot(range(1, max(history.epoch) + 2), history.history[metric_to_plot], ".:", label="Training loss")
    plt.plot(range(1, max(history.epoch) + 2), history.history[val_metric_to_plot], ".:", label="Validation loss")
    plt.title('Training and Validation Loss')
    plt.xlim([1,max(history.epoch) + 2])
    plt.xticks(range(1, max(history.epoch) + 2))
    plt.legend()
    plt.show()
    
model, history = get_model_performance('best_model_file_name.sav', vocab_size, max_length)
plot_performance(history)
