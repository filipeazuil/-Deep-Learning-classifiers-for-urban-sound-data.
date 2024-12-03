import keras_tuner as kt
from tensorflow.keras import layers, models, Input, regularizers
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.regularizers import l2
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Dense, Flatten, Dropout, MaxPooling1D, Conv1D, BatchNormalization
from tensorflow.keras.optimizers import SGD
import pickle

def build_model_mfcc(hp):
    model = Sequential()

    # Input layer
    model.add(Input(shape=(40, 174, 1)))

    # First convolutional block with L2 regularization
    model.add(Conv2D(
        filters=hp.Int('conv1_filters', min_value=16, max_value=64, step=16),  # Tunable
        kernel_size=5,  # Tunable
        activation='relu',
        kernel_regularizer=regularizers.l2(hp.Float('l2', min_value=0.01, max_value=0.1, step=0.01))  # Tunable
    ))
    model.add(MaxPooling2D(pool_size=(2, 2)))

    # Second convolutional block with L2 regularization
    model.add(Conv2D(
        filters=hp.Int('conv2_filters', min_value=32, max_value=64, step=32),  # Tunable
        kernel_size=3,  # Tunable
        activation='relu',
        kernel_regularizer=regularizers.l2(hp.Float('l2', min_value=0.01, max_value=0.1, step=0.01))  # Tunable
    ))
    model.add(MaxPooling2D(pool_size=(2, 2)))

    # Flatten the output
    model.add(Flatten())

    # Dense layer with L2 regularization
    model.add(Dense(
        units=hp.Int('dense_units', min_value=16, max_value=64, step=16),  # Tunable
        activation='relu',
        kernel_regularizer=regularizers.l2(hp.Float('l2', min_value=0.01, max_value=0.1, step=0.01))  # Tunable
    ))

    # Dropout layer for additional regularization
    model.add(Dropout(rate=hp.Float('dropout', min_value=0.2, max_value=0.4, step=0.1)))  # Tunable

    # Output layer
    model.add(Dense(10, activation='softmax'))

    # Compile the model
    learning_rate = hp.Float('learning_rate', min_value=1e-4, max_value=1e-3, sampling='log')  # Tunable
    optimizer = SGD(learning_rate=learning_rate, momentum=0.9)
    model.compile(optimizer=optimizer, loss='categorical_crossentropy', metrics=['accuracy'])

    return model

# Define the model-building function for Hyperband
def build_model_1d(hp):
    num_classes = 10  # Predicting 10 classes
    input_shape = (4000 , 1)

    # Create a sequential model
    model = Sequential()

    # First convolutional block with tunable hyperparameters
    model.add(Conv1D(filters=hp.Int('conv_1_filters', min_value=32, max_value=128, step=32), 
                     kernel_size=3, activation='relu', 
                     input_shape=input_shape, 
                     kernel_regularizer=l2(hp.Float('l2_lambda', min_value=0.0001, max_value=0.1, step=0.0001))))
    model.add(BatchNormalization())
    model.add(MaxPooling1D(pool_size=2))

    # Second convolutional block with tunable filters
    model.add(Conv1D(filters=hp.Int('conv_2_filters', min_value=64, max_value=256, step=64), 
                     kernel_size=3, activation='relu', 
                     kernel_regularizer=l2(hp.Float('l2_lambda', min_value=0.0001, max_value=0.1, step=0.0001))))
    model.add(BatchNormalization())
    model.add(MaxPooling1D(pool_size=2))

    # Third convolutional block with tunable filters
    model.add(Conv1D(filters=hp.Int('conv_3_filters', min_value=128, max_value=512, step=128), 
                     kernel_size=3, activation='relu', 
                     kernel_regularizer=l2(hp.Float('l2_lambda', min_value=0.0001, max_value=0.1, step=0.0001))))
    model.add(BatchNormalization())
    model.add(MaxPooling1D(pool_size=2))

    # Flatten and add fully connected layers with L2 regularization
    model.add(Flatten())
    model.add(Dense(hp.Int('dense_units', min_value=64, max_value=256, step=64), activation='relu', kernel_regularizer=l2(hp.Float('l2_lambda', min_value=0.0001, max_value=0.1, step=0.0001))))
    model.add(Dropout(hp.Float('dropout_rate', min_value=0.2, max_value=0.5, step=0.1)))  # Tunable dropout rate

    # Output layer
    model.add(Dense(num_classes, activation='softmax', kernel_regularizer=l2(hp.Float('l2_lambda', min_value=0.0001, max_value=0.1, step=0.0001))))
    
    # Learning rate tuning
    optimizer_adam = Adam(learning_rate=hp.Float('learning_rate', min_value=1e-5, max_value=1e-2, sampling='log'))
    
    # Compile the model
    model.compile(optimizer=optimizer_adam,
                  loss='categorical_crossentropy',
                  metrics=['accuracy'])
    return model

# Define the model-building function for Hyperband
def build_model_mel(hp):
    # Create a sequential model
    model = Sequential()

    # Input layer
    model.add(Input(shape=(40, 174, 1)))

    # First convolutional block with tunable hyperparameters
    model.add(Conv2D(
        filters=hp.Int('conv_1_filters', min_value=32, max_value=64, step=32),
        kernel_size=(5, 5),
        activation='relu',
        kernel_regularizer=regularizers.l2(hp.Float('l2_lambda', min_value=0.001, max_value=0.01, step=0.0001))
    ))

    # Second convolutional block with tunable hyperparameters
    model.add(Conv2D(
        filters=hp.Int('conv_2_filters', min_value=16, max_value=64, step=16),
        kernel_size=(3, 3),
        activation='relu',
        kernel_regularizer=regularizers.l2(hp.Float('l2_lambda', min_value=0.001, max_value=0.01, step=0.0001))
    ))
    model.add(MaxPooling2D(pool_size=(2, 2)))

    # Flatten the output
    model.add(Flatten())

    # Dense layer with tunable hyperparameters
    model.add(Dense(
        units=hp.Int('dense_units', min_value=64, max_value=128, step=64),
        activation='relu',
        kernel_regularizer=regularizers.l2(hp.Float('l2_lambda', min_value=0.001, max_value=0.01, step=0.0001))
    ))

    # Dropout layer with tunable dropout rate
    model.add(Dropout(rate=hp.Float('dropout_rate', min_value=0.3, max_value=0.4, step=0.1)))

    # Output layer
    model.add(Dense(10, activation='softmax'))

    optimizer = SGD(learning_rate=hp.Float('learning_rate', min_value=5e-5, max_value=1e-3, sampling='log'), momentum=0.9)

    # Compile the model
    model.compile(optimizer=optimizer, loss='categorical_crossentropy', metrics=['accuracy'])

    return model

def save_pkl(data, path):
    with open(path, "wb") as saved_data:
        pickle.dump(data, saved_data)
    saved_data.close()

def load_pkl(path):
    to_return = None
    with open(path, "rb") as loaded_data:
        to_return = pickle.load(loaded_data)
    loaded_data.close()
    return to_return

def grad_val(model, x, k):

    with tf.GradientTape(persistent=True) as tape:
        inputs = tf.cast(x, dtype=tf.float64)
        tape.watch(inputs)
        results = model(inputs)
        results_k = results[0,k]
        results_k=tf.convert_to_tensor(results_k, dtype=tf.float32)
        
    gradients = tape.gradient(results_k, inputs)
    del tape
    return [grad.numpy() for grad in gradients], results

def robustness_val(x_list, r_list):
    r_norm = np.sqrt(np.sum(np.fromiter([np.linalg.norm(r_input)**2 for r_input in r_list], dtype=np.float32)))
    x_norm = np.sqrt(np.sum(np.fromiter([np.linalg.norm(x_input)**2 for x_input in x_list], dtype=np.float32)))
    return r_norm / x_norm

