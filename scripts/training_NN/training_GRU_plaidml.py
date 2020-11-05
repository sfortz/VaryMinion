from keras import layers
from keras.models import Sequential
import vary_minion_losses_plaidml as minion_losses


# create a GRU
# adapted from https://keras.io/guides/working_with_rnns/
def get_GRU_model(multi=False, alpha_size=128, nb_classes=1, nb_col=128, nb_unit=10):

    # we need to return a model compatible with the plaidml backend
    print("constructing a GRU model independent of tensorflow for plaidml")

    model = Sequential()
    model.add(layers.Embedding(alpha_size, alpha_size, input_length=nb_col))
    model.add(layers.Bidirectional(layers.GRU(nb_unit, activation='relu')))

    # https://towardsdatascience.com/deep-learning-which-loss-and-activation-functions-should-i-use-ac02f1c56aa8
    if multi:
        # sigmoid is well suited when we predict multiple label for multiple classes:
        model.add(layers.Dense(nb_classes, activation='tanh'))

        # Binary Cross Entropy is well suited in this case:
        model.compile(optimizer='adam',
                      #loss = 'binary_crossentropy',
                      loss=minion_losses.vary_manhattan_dist_indiv,
                      metrics=['accuracy'])

    else:
        # softmax is well suited when we predict a single label for multiple classes:
        model.add(layers.Dense(nb_classes, activation='softmax'))

        # Categorical Cross Entropy is well suited in this case:
        model.compile(optimizer='adam',
                      loss='categorical_crossentropy',
                      metrics=['accuracy'])

    return model
