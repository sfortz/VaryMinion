from os import environ
environ["KERAS_BACKEND"] = "plaidml.keras.backend"  # NB:  PlaidML backend has to be imported before keras to work
# (sorry PEP8 ;))
import numpy as np
from keras import backend as K
import plaidml
import plaidml.keras.backend as pkb


def ioU_jaccard_distance(y_true, y_predicted, smooth=100):
    """ This distance is copied verbatim from:
    https://github.com/keras-team/keras-contrib/blob/master/keras_contrib/losses/jaccard.py"""

    # conversion to be sure floating point 64 bits valued are not used => unsupported by plaidML.
    y_true = pkb.cast(y_true, dtype=plaidml.DType.FLOAT32)
    y_predicted = pkb.cast(y_predicted, dtype=plaidml.DType.FLOAT32)

    intersection = K.sum(K.abs(y_true * y_predicted), axis=-1)
    sum_ = K.sum(K.abs(y_true) + K.abs(y_predicted), axis=-1)
    jac = (intersection + smooth) / (sum_ - intersection + smooth)
    return (1 - jac) * smooth


def dummy_indicator(y_predicted, thres=0.0):
    """
    This indicator return a tensorflow with values 0 if values in y_predicted < thres, else 1.
    """
    y_predicted = pkb.cast(y_predicted, dtype='float32')
    np_thres = np.full_like(y_predicted,thres)

    y_predicted = pkb.greater(y_predicted, np_thres)
    y_predicted = pkb.cast(y_predicted, dtype='float32')

    return y_predicted

# TODO: remove this method not suitable as a loss
def vary_boolean_jaccard(y_true, y_predicted):
    """
    This function implements the classic jaccard distance over an indicator functions that converts float predictions
    into boolean values.
    """

    y_predicted = dummy_indicator(y_predicted)
    y_predicted = pkb.cast(y_predicted, 'int32')
    y_true = pkb.cast(y_true, 'int32')

    intersection = pkb.sum(pkb.equal(y_predicted, y_true), axis=-1)
    # union = pkb.sum(pkb.ones_like(y_predicted)) # number of possible matches (tensors y_predicted, y_true have the same number
    #     # of values)
    union = pkb.sum(pkb.ones_like(y_predicted), axis=-1)  # number of possible matches (tensors y_predicted, y_true have the same number
    # of values)
    j_index = intersection / union
    ones_j = pkb.ones_like(j_index)
    dist = ones_j - j_index

    return dist

# TODO remove this method
def vary_boolean_jaccard_indiv(y_true, y_predicted):
    """
    This function implements the classic jaccard distance over an indicator functions that converts float predictions
    into boolean values. For each sample, it provides an array of losses for each class.
    """

    y_predicted = dummy_indicator(y_predicted)
    y_predicted = pkb.cast(y_predicted, 'int32')
    y_true = pkb.cast(y_true, 'int32')

    intersection = pkb.equal(y_predicted, y_true)
    # union = pkb.sum(pkb.ones_like(y_predicted)) # number of possible matches (tensors y_predicted, y_true have the same number
    #     # of values)
    union = pkb.ones_like(y_predicted)  # number of possible matches (tensors y_predicted, y_true have the same number
    # of values)
    j_index = intersection / union
    ones_j = pkb.ones_like(j_index)
    dist = ones_j - j_index

    return dist


def vary_weighted_jaccard(y_true, y_predicted):
    """ This function implements the weighted jaccard distance also known as Soergel:
    https://en.wikipedia.org/wiki/Jaccard_index. Since it works on real and positive numbers no indicator is needed
    to translate probabilities into Boolean values. """

    # conversion to be sure floating point 64 bits valued are not used => unsupported by plaidML.
    y_true = pkb.cast(y_true, dtype=plaidml.DType.FLOAT32)
    y_predicted = pkb.cast(y_predicted, dtype=plaidml.DType.FLOAT32)

    num = pkb.sum(pkb.minimum(y_true, y_predicted), axis=-1)
    denom = pkb.sum(pkb.maximum(y_true, y_predicted), axis=-1)

    j = num / denom
    ones = pkb.ones_like(j)
    return ones - j

# TODO remove this method
def vary_weighted_jaccard_indiv(y_true, y_predicted):
    """ This function implements the weighted jaccard distance also known as Soergel:
    https://en.wikipedia.org/wiki/Jaccard_index. Since it works on real and positive numbers no indicator is needed
    to translate probabilities into Boolean values. This function "rectifies" negative values to zero.  This  function
    return a loss for each class (one array of losses per sample) """

    # conversion to be sure floating point 64 bits valued are not used => unsupported by plaidML.
    y_true = pkb.cast(y_true, dtype=plaidml.DType.FLOAT32)
    y_predicted = pkb.cast(y_predicted, dtype=plaidml.DType.FLOAT32)

    num = pkb.minimum(y_true, y_predicted)
    denom = pkb.maximum(y_true, y_predicted)

    j = num / denom
    ones = pkb.ones_like(j)
    return ones - j


def vary_weighted_jaccard_rectified(y_true, y_predicted):
    """ This function implements the weighted jaccard distance also known as Soergel:
    https://en.wikipedia.org/wiki/Jaccard_index. Since it works on real and positive numbers no indicator is needed
    to translate probabilities into Boolean values. This function "rectifies" negative values to zero. """

    # conversion to be sure floating point 64 bits valued are not used => unsupported by plaidML.
    y_true = pkb.cast(y_true, dtype=plaidml.DType.FLOAT32)
    y_predicted = pkb.cast(y_predicted, dtype=plaidml.DType.FLOAT32)

    # rectification of negative values to zero
    zeros = pkb.zeros_like(y_predicted)
    y_predicted = pkb.maximum(y_predicted, zeros)

    num = pkb.sum(K.minimum(y_true, y_predicted), axis=-1)
    denom = pkb.sum(K.maximum(y_true, y_predicted), axis=-1)
    j = num / denom
    ones = pkb.ones_like(j)
    return ones - j


# TODO remove this method
def vary_weighted_jaccard_rectified_indiv(y_true, y_predicted):
    """ This function implements the weighted jaccard distance also known as Soergel:
    https://en.wikipedia.org/wiki/Jaccard_index. Since it works on real and positive numbers no indicator is needed
    to translate probabilities into Boolean values. This function "rectifies" negative values to zero.  This  function
    return a loss for each class (one array of losses per sample)"""

    # conversion to be sure floating point 64 bits valued are not used => unsupported by plaidML.
    y_true = pkb.cast(y_true, dtype=plaidml.DType.FLOAT32)
    y_predicted = pkb.cast(y_predicted, dtype=plaidml.DType.FLOAT32)

    # rectification of negative values to zero
    zeros = pkb.zeros_like(y_predicted)
    y_predicted = pkb.maximum(y_predicted, zeros)

    num = pkb.minimum(y_true, y_predicted)
    denom = pkb.maximum(y_true, y_predicted)
    j = num / denom
    ones = pkb.ones_like(j)
    return ones - j


def vary_manhattan_dist(y_true,y_predicted):
    """
    Implements Manhattan distance a loss. It reports one loss value per sample.
    """
    y_actual_float = pkb.cast(y_true, dtype=plaidml.DType.FLOAT32)
    manh_dist = pkb.sum(pkb.abs(y_actual_float - y_predicted), axis=-1)
    return manh_dist


def vary_manhattan_dist_indiv(y_true,y_predicted):
    """
    Implements Manhattan distance a loss. It reports individual loss values: one array per sample with loss values for each class
    """
    y_actual_float = pkb.cast(y_true, dtype=plaidml.DType.FLOAT32)
    manh_dist = pkb.abs(y_actual_float - y_predicted)
    return manh_dist

