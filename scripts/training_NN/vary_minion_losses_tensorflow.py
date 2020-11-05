import tensorflow as tf
from tensorflow.keras import backend as kb
from keras import backend as K



# used for segementation, not sure that what we want here :(
def ioU_jaccard_distance(y_true, y_pred, smooth=100):
    """ This distance is copied verbatim from:
    https://github.com/keras-team/keras-contrib/blob/master/keras_contrib/losses/jaccard.py"""

    intersection = K.sum(K.abs(y_true * y_pred), axis=-1)
    sum_ = K.sum(K.abs(y_true) + K.abs(y_pred), axis=-1)
    print("intersection")
    kb.print_tensor(intersection)
    print("union")
    kb.print_tensor(sum_)

    jac = (intersection + smooth) / (sum_ - intersection + smooth)
    return (1 - jac) * smooth


def dummy_indicator(y_predicted, thres=0.0):
    """
    This indicator return a tensor with values 0 if values in y_predicted < thres, else 1.
    """

    y_predicted = K.tf.where(K.greater(y_predicted, thres),K.ones_like(y_predicted), K.zeros_like(y_predicted))

    return y_predicted

# TODO: remove this method not suitable as a loss
def vary_boolean_jaccard(y_true, y_predicted):
    """
    This function implements the classic jaccard distance over an indicator functions that converts float predictions
    into boolean values.
    """
    y_predicted = dummy_indicator(y_predicted)
    y_true = K.cast(y_true, tf.float32)
    intersection = K.sum(K.abs(y_true - y_predicted), axis=-1)
    union = K.sum(K.ones_like(y_predicted), axis=-1) # number of possible matches (tensors y_predicted, y_true have the same number
    # of values)
    j_index = K.tf.divide(intersection, union)

    return j_index

# a different form of the jaccard distance which considers minimum and maximum values of the tensors 
def vary_weighted_jaccard(y_true, y_predicted):
    """ This function implements the weighted jaccard distance also known as Soergel:
    https://en.wikipedia.org/wiki/Jaccard_index. Since it works on real and positive numbers no indicator is needed
    to translate probabilities into Boolean values. """

    # we convert actual labels to float to ease comparisons
    y_true = tf.cast(y_true, tf.float32)
    num = kb.sum(kb.minimum(y_true, y_predicted), axis=-1)
    denom = kb.sum(kb.maximum(y_true, y_predicted), axis=-1)
    j_index = tf.divide(num, denom)

    ones = tf.ones_like(j_index)
    return ones - j_index

# bounded version of the weighted jaccard distance putting negative values to 0
def vary_weighted_jaccard_rectified(y_true, y_predicted):
    """ This function implements the weighted jaccard distance also known as Soergel:
    https://en.wikipedia.org/wiki/Jaccard_index. Since it works on real and positive numbers no indicator is needed
    to translate probabilities into Boolean values. This function "rectifies" negative values to zero. """

    # rectification of negative values to zero
    zeros = tf.zeros_like(y_predicted)
    y_predicted = kb.maximum(y_predicted, zeros)
    # we convert actual labels to float to ease comparisons
    y_true = tf.cast(y_true, tf.float32)

    num = kb.sum(kb.minimum(y_true, y_predicted), axis=-1)
    denom = kb.sum(kb.maximum(y_true, y_predicted), axis=-1)
    j_index = tf.divide(num, denom)

    ones = tf.ones_like(j_index)
    return ones - j_index

# the manhattan distance between two tensors
def vary_manhattan_dist(y_true,y_predicted):
    """
    Implements Manatthan distance a loss
    """
    y_actual_float = tf.cast(y_true, tf.float32)
    manh_dist = kb.sum(kb.abs(y_actual_float - y_predicted), axis=-1)
    return manh_dist

# TODO remove this method
def vary_manhattan_dist_indiv(y_true,y_predicted):
    """
    Implements Manatthan distance a loss. Reports detailed losses per classes. DO NOT USE
    """
    y_actual_float = tf.cast(y_true, tf.float32)
    manh_dist = kb.abs(y_actual_float - y_predicted)
    return manh_dist
