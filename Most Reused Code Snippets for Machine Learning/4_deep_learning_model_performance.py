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
def get_deep_learning_model(vocab_size, max_length):
    model = Sequential()
    model.add(Embedding(vocab_size, 10, input_length=max_length))
    model.add(Flatten())
    model.add(Dense(1, activation='sigmoid'))
    return model

def get_model_performance(best_model_file_name, vocab_size, max_length):
    keras.backend.clear_session()
    model = get_deep_learning_model(vocab_size, max_length) # Call the model here
    callbacks=[
        keras.callbacks.EarlyStopping(monitor="val_loss", 
                                      patience=15, 
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
                        validation_data=(X_test, y_test), 
                        callbacks=callbacks)
    return model, history                
                    
def plot_performance(history):
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

model, history = get_model_performance('best_model_file_name', vocab_size, max_length)
plot_performance(history)
