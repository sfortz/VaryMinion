from keras import layers
from keras.models import Sequential

# create a RNN
# adapted from https://keras.io/guides/working_with_rnns/
def get_RNN_model(multi=False, alpha_size=128, nb_classes=1, nb_col=128, nb_unit=10):

    # we need to return a model compatible with the plaidml backend
    print("constructing a RNN model independent of tensorflow for plaidml")
    model = Sequential()
    model.add(layers.Embedding(alpha_size, alpha_size, input_length=nb_col))
    model.add(layers.SimpleRNN(nb_unit))

    # https://towardsdatascience.com/deep-learning-which-loss-and-activation-functions-should-i-use-ac02f1c56aa8
    if multi :
        # softmax is well suited when we predict multiple label for multiple classes:
        model.add(layers.Dense(nb_classes, activation='sigmoid'))

        # Binary Cross Entropy is well suited in this case:
        model.compile(optimizer='adam',
                      loss='binary_crossentropy',
                      metrics=['accuracy'])

    else :
        # softmax is well suited when we predict a single label for multiple classes:
        model.add(layers.Dense(nb_classes, activation='softmax'))

        # Categorical Cross Entropy is well suited in this case:
        model.compile(optimizer='adam',
                      loss='categorical_crossentropy',
                      metrics=['accuracy'])

    return model
