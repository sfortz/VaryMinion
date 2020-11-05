from os import environ

environ["KERAS_BACKEND"] = "plaidml.keras.backend"  # NB:  PlaidML backend has to be imported before keras to work
# (sorry PEP8 ;))
# import plaidml
import keras.backend as K
import plaidml.keras.backend as pkb
import unittest
import vary_minion_losses_plaidml as minion_losses
import numpy as np


class MyTestCase(unittest.TestCase):

    def test_IoU_jaccard_loss(self):
        y_true = np.array([[1.0, 1.0, 1.0]])
        y_pred = np.array([[1.0, 1.0, 1.0]])

        y_true = y_true.astype(
            'float32')  # Default Dtype for numpy array is float64, yet plaidml does not support 64 bits floats.
        y_pred = y_pred.astype(
            'float32')  # Default Dtype for numpy array is float64, yet plaidml does not support 64 bits floats.

        val = K.eval(minion_losses.ioU_jaccard_distance(y_true, y_pred))
        # in this case the distance should be 0

        expected = np.array([0.0])

        self.assertTrue(np.array_equiv(val, expected))

    def test_IoU_jaccard_loss_multi(self):
        y_true = np.array([[1.0, 1.0, 1.0],[1.0, 1.0, 1.0]])
        y_pred = np.array([[0.0, 0.0, 0.0], [0.0, 1.0, 1.0]])

        y_true = y_true.astype(
            'float32')  # Default Dtype for numpy array is float64, yet plaidml does not support 64 bits floats.
        y_pred = y_pred.astype(
            'float32')  # Default Dtype for numpy array is float64, yet plaidml does not support 64 bits floats.

        val = K.eval(minion_losses.ioU_jaccard_distance(y_true, y_pred))

        expected = np.array([2.91262269, 0.97087026])
        expected = expected.astype('float32')

        self.assertTrue(np.array_equiv(expected, val), msg=f"expected: {expected}, got: {val}")


    def test_boolean_jaccard(self):
        y_true = np.array([[1.0, 1.0, 1.0]])
        y_pred = np.array([[0.0, 0.0, 1.0]])
        y_true = y_true.astype(
            'float32')  # Default Dtype for numpy array is float64, yet plaidml does not support 64 bits floats.
        y_pred = y_pred.astype(
            'float32')  # Default Dtype for numpy array is float64, yet plaidml does not support 64 bits floats.

        val = pkb.eval(minion_losses.vary_boolean_jaccard(y_true, y_pred))
        expected = np.array([0.6666667])
        expected = expected.astype('float32')

        self.assertTrue(np.array_equiv(expected, val), msg=f"expected: {expected}, got: {val}")

    def test_boolean_jaccard_2D(self):
        y_true = np.array([[1.0, 1.0, 1.0]])
        y_pred = np.array([[0.0, 0.0, 1.0], [0.0, 0.0, 0.15]])
        y_true = y_true.astype(
            'float32')  # Default Dtype for numpy array is float64, yet plaidml does not support 64 bits floats.
        y_pred = y_pred.astype(
            'float32')  # Default Dtype for numpy array is float64, yet plaidml does not support 64 bits floats.

        val = pkb.eval(minion_losses.vary_boolean_jaccard(y_true, y_pred))
        expected = np.array([0.6666667, 0.6666667])
        expected = expected.astype('float32')

        self.assertTrue(np.array_equal(expected, val))

    def test_boolean_jaccard_indiv(self):
        y_true = np.array([[1.0, 1.0, 1.0]])
        y_pred = np.array([[0.0, 0.0, 1.0], [0.0, 0.0, 0.15]])
        y_true = y_true.astype(
            'float32')  # Default Dtype for numpy array is float64, yet plaidml does not support 64 bits floats.
        y_pred = y_pred.astype(
            'float32')  # Default Dtype for numpy array is float64, yet plaidml does not support 64 bits floats.

        val = pkb.eval(minion_losses.vary_boolean_jaccard_indiv(y_true, y_pred))
        expected = np.array([[1.0, 1.0, 0.0], [1.0, 1.0, 0.0]])
        expected = expected.astype('float32')

        self.assertTrue(np.array_equal(expected, val))

    def test_dummy_indicator_non_zero(self):
        y_true = np.array([[0.0, 1.0, 1.0]])
        y_pred = np.array([[-0.5, 0.22, 0.47]])
        y_true = y_true.astype(
            'float32')  # Default Dtype for numpy array is float64, yet plaidml does not support 64 bits floats.
        y_pred = y_pred.astype(
            'float32')  # Default Dtype for numpy array is float64, yet plaidml does not support 64 bits floats.
        indicator = pkb.eval(minion_losses.dummy_indicator(y_pred))

        self.assertTrue(np.array_equal(y_true, indicator))

    def test_dummy_indicator(self):
        y_true = np.array([[0.0, 1.0, 1.0]])
        y_pred = np.array([[-0.5, 0.22, 0.47]])
        y_true = y_true.astype(
            'float32')  # Default Dtype for numpy array is float64, yet plaidml does not support 64 bits floats.
        y_pred = y_pred.astype(
            'float32')  # Default Dtype for numpy array is float64, yet plaidml does not support 64 bits floats.
        indicator = pkb.eval(minion_losses.dummy_indicator(y_pred))
        cmp = np.array_equal(y_true, indicator)
        self.assertTrue(cmp)

    def test_weighted_jaccard(self):
        y_true = np.array([1.0, 1.0, 1.0])
        y_pred = np.array([[1.0, 1.0, 0.0]])
        y_true = y_true.astype(
            'float32')  # Default Dtype for numpy array is float64, yet plaidml does not support 64 bits floats.
        y_pred = y_pred.astype(
            'float32')  # Default Dtype for numpy array is float64, yet plaidml does not support 64 bits floats.
        val = pkb.eval(minion_losses.vary_weighted_jaccard(y_true, y_pred))

        expected = np.array([0.33333337])
        expected = expected.astype('float32')

        self.assertTrue(np.array_equal(expected, val), msg=f" expected:  [0.0, 0.0, 0.0], got: {val}")

    def test_weighted_jaccard_indiv(self):
        y_true = np.array([1.0, 1.0, 1.0])
        y_pred = np.array([[1.0, 1.0, 0.0]])
        y_true = y_true.astype(
            'float32')  # Default Dtype for numpy array is float64, yet plaidml does not support 64 bits floats.
        y_pred = y_pred.astype(
            'float32')  # Default Dtype for numpy array is float64, yet plaidml does not support 64 bits floats.
        val = pkb.eval(minion_losses.vary_weighted_jaccard_indiv(y_true, y_pred))

        expected = np.array([[0.0, 0.0, 1.0]])
        expected = expected.astype('float32')

        self.assertTrue(np.array_equal(expected, val), msg=f" expected:  {expected}, got: {val}")

    def test_weighted_jaccard_rectified(self):
        y_true = np.array([1.0, 1.0, 1.0])
        y_pred = np.array([[1.0, 0.0, 0.0], [1.0, 0.0, 0.0]])
        y_true = y_true.astype(
            'float32')  # Default Dtype for numpy array is float64, yet plaidml does not support 64 bits floats.
        y_pred = y_pred.astype(
            'float32')  # Default Dtype for numpy array is float64, yet plaidml does not support 64 bits floats.
        val = pkb.eval(minion_losses.vary_weighted_jaccard_rectified(y_true, y_pred))
        expected = np.array([0.6666667, 0.6666667])
        expected = expected.astype('float32')

        self.assertTrue(np.array_equal(expected, val))

    def test_weighted_jaccard_rectified_negative(self):
        y_true = np.array([1.0, 1.0, 1.0])
        y_pred = np.array([[1.0, -0.5, 0.0], [1.0, 0.0, 0.0]])
        y_true = y_true.astype(
            'float32')  # Default Dtype for numpy array is float64, yet plaidml does not support 64 bits floats.
        y_pred = y_pred.astype(
            'float32')  # Default Dtype for numpy array is float64, yet plaidml does not support 64 bits floats.
        val = pkb.eval(minion_losses.vary_weighted_jaccard_rectified(y_true, y_pred))

        expected = np.array([0.6666667, 0.6666667])
        expected = expected.astype('float32')

        self.assertTrue(np.array_equal(expected, val))

    def test_weighted_jaccard_rectified_negative_indiv(self):
        y_true = np.array([1.0, 1.0, 1.0])
        y_pred = np.array([[1.0, -0.5, 0.0], [1.0, 0.0, 0.0]])
        y_true = y_true.astype(
            'float32')  # Default Dtype for numpy array is float64, yet plaidml does not support 64 bits floats.
        y_pred = y_pred.astype(
            'float32')  # Default Dtype for numpy array is float64, yet plaidml does not support 64 bits floats.
        val = pkb.eval(minion_losses.vary_weighted_jaccard_rectified_indiv(y_true, y_pred))

        expected = np.array([[0.0, 1.0, 1.0], [0.0, 1.0, 1.0]])
        expected = expected.astype('float32')

        self.assertTrue(np.array_equal(expected, val))

    def test_manhattan(self):
        y_true = np.array([1, 1, 0])
        y_pred = np.array([[0.0, 0.0, 0.0]])
        # y_true = y_true.astype('float32') # Defaut Dtype for numpy array is float64, yet plaidml does not support 64 bits floats.
        y_pred = y_pred.astype(
            'float32')  # Defaut Dtype for numpy array is float64, yet plaidml does not support 64 bits floats.
        val = pkb.eval(minion_losses.vary_manhattan_dist(y_true, y_pred))
        expected = np.array([2.0])
        self.assertTrue(np.array_equal(expected, val))

    def test_manhattan_2(self):
        y_true = np.array([1, 1, 0])
        y_pred = np.array([[0.0, 0.0, 0.0], [0.0, 0.5, 0.0], [0.0, 1.0, 0.0], [1.0, 1.0, 0.0]])
        # y_true = y_true.astype('float32') # Defaut Dtype for numpy array is float64, yet plaidml does not support 64 bits floats.
        y_pred = y_pred.astype(
            'float32')  # Defaut Dtype for numpy array is float64, yet plaidml does not support 64 bits floats.
        val = pkb.eval(minion_losses.vary_manhattan_dist(y_true, y_pred))
        expected = np.array([2.0, 1.5, 1.0, 0.0])

        self.assertTrue(np.array_equal(expected,
                                       val))  # equiv => shape may differ but value are bit.  Difference between float32 and float64.

    def test_manhattan_indiv(self):
        y_true = np.array([1, 1, 0])
        y_pred = np.array([[0.0, 0.0, 0.0], [0.0, 0.5, 0.0], [0.0, 1.0, 0.0], [1.0, 1.0, 0.0]])
        # y_true = y_true.astype('float32') # Defaut Dtype for numpy array is float64, yet plaidml does not support 64 bits floats.
        y_pred = y_pred.astype(
            'float32')  # Defaut Dtype for numpy array is float64, yet plaidml does not support 64 bits floats.
        val = pkb.eval(minion_losses.vary_manhattan_dist_indiv(y_true, y_pred))

        expected = np.array([[1.0, 1.0, 0.0], [1.0, 0.5, 0.0], [1.0, 0.0, 0.0], [0.0, 0.0, 0.0]])

        self.assertTrue(np.array_equal(expected,
                                       val))  # equiv => shape may differ but value are bit.  Difference between float32 and float64.


if __name__ == '__main__':
    unittest.main()
