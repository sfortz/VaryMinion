import tensorflow as tf
from tensorflow.keras import layers
import vary_minion_losses_tensorflow as minion_losses

# create a LSTM
# adapted from https://keras.io/guides/working_with_rnns/
def get_LSTM_model(multi=False, alpha_size=128, nb_classes=1, nb_col=128, nb_unit=10):
    # the model takes as input and output only one kind of such entities (Sequential) -> takes a list of Events and outputs
    # the configuration to which the trace belongs. In between, only the result of the neuron functions are aggregated.
    # model = tf.keras.Sequential([
    # tf.keras.layers.Dense(10, activation='relu'),
    # tf.keras.layers.Dense(10, activation='relu'),
    # tf.keras.layers.Dense(nb_classes)
    # ])

    model = tf.keras.Sequential()
    model.add(layers.Embedding(alpha_size, alpha_size, mask_zero=True))
    model.add(layers.Bidirectional(layers.LSTM(nb_unit,activation='relu')))
    # model.add(layers.Bidirectional(layers.LSTM(nb_unit)))

    # https://towardsdatascience.com/deep-learning-which-loss-and-activation-functions-should-i-use-ac02f1c56aa8
    if multi:
        # sigmoid is well suited when we predict multiple label for multiple classes:
        # model.add(layers.Dense(nb_classes, activation='sigmoid'))
         #model.add(layers.Dense(nb_classes))
        # model.add(layers.Dense(nb_classes, activation='relu'))
        model.add(layers.Dense(nb_classes, activation='tanh'))
        # Binary Cross Entropy is well suited in this case:
        model.compile(optimizer='adam',
                      loss=tf.keras.losses.BinaryCrossentropy(from_logits=True),
                      #loss=tf.keras.losses.binary_crossentropy,
                      #loss=minion_losses.vary_manhattan_dist,
                      #loss=minion_losses.vary_weighted_jaccard,
                      metrics=['accuracy'])
                      #metrics=[tf.keras.metrics.TopKCategoricalAccuracy(k=3)])

    else:
        # softmax is well suited when we predict a single label for multiple classes:
        model.add(layers.Dense(nb_classes, activation='softmax'))

        # Categorical Cross Entropy is well suited in this case:
        model.compile(optimizer='adam',
                      #loss='mean_squared_error',
                      loss=tf.keras.losses.CategoricalCrossentropy(from_logits=True),
                      metrics=['accuracy'])

    return model
