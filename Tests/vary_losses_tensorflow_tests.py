import unittest
import tensorflow as tf
import tensorflow.keras.backend as kb
import vary_minion_losses_tensorflow as minion_losses

class MyTestCase(unittest.TestCase):


    def test_IoU_jaccard_loss(self):
        y_true = tf.constant([1.0, 1.0, 1.0])
        y_pred = tf.constant([[1.0, 1.0, 1.0]])
        val = minion_losses.ioU_jaccard_distance(y_true, y_pred)
        # in this case the distance should be 2/3 (0.6666666666666667)

        expected = tf.constant([0.0], dtype=tf.float32)

        self.assertTrue(tf.math.equal(val, expected))

    def test_IoU_jaccard_loss_multi(self):
        y_true = tf.constant([1.0, 1.0, 1.0])
        y_pred = tf.constant([[0.0, 0.0, 0.0], [0.0, 1.0, 1.0]])
        val = minion_losses.ioU_jaccard_distance(y_true, y_pred)
        # in this case the distance should be 2/3 (0.6666666666666667)
        print(val)
        expected = tf.constant([2.9126227, 0.9708762], dtype=tf.float32)
        cmp = tf.math.equal(expected, val)
        self.assertTrue(kb.all(cmp))

    def test_weighted_jaccard(self):
        y_true = tf.constant([1.0, 1.0, 1.0])
        y_pred = tf.constant([1.0, 0.0, 0.0])
        val = minion_losses.vary_weighted_jaccard(y_true, y_pred)
        # in this case the distance should be 2/3 (0.6666666)
        expected = tf.constant([0.6666666])
        self.assertTrue(tf.math.equal(val, expected))

    def test_weighted_jaccard_rectified(self):
        y_true = tf.constant([1.0, 1.0, 1.0])
        y_pred = tf.constant([1.0, 0.0, 0.0])
        val = minion_losses.vary_weighted_jaccard_rectified(y_true, y_pred)
        # in this case the distance should be 2/3 (0.6666666)
        expected = tf.constant([0.6666666])
        self.assertTrue(tf.math.equal(val, expected))


    def test_weighted_jaccard_rectified_multi(self):
        y_true = tf.constant([1.0, 1.0, 1.0])
        y_pred = tf.constant([[1.0, 1.0, 0.0], [1.0, 1.0, 0.0]])
        val = minion_losses.vary_weighted_jaccard_rectified(y_true, y_pred)
        # in this case the distance should be 1/3 (0.3333333)
        #print(tf.keras.backend.eval(val))
        expected = tf.constant([0.3333333, 0.3333333])

        cmp = tf.math.equal(val, expected)
        self.assertTrue(tf.keras.backend.all(cmp))

    def test_weighted_jaccard_rectified_negative(self):
        y_true = tf.constant([1.0, 1.0, 1.0])
        y_pred = tf.constant([[1.0, -0.74, 0.0], [1.0, 0.36, 0.0]])
        val = minion_losses.vary_weighted_jaccard_rectified(y_true, y_pred)
        # in this case the distance should be 2/3 (0.6666666)
        expected = tf.constant([0.6666666, 0.5466666])

        cmp = tf.math.equal(val, expected)
        self.assertTrue(tf.keras.backend.all(cmp))


    def test_dummy_indicator(self):
        y_true = tf.constant([[0.0, 1.0, 1.0]], dtype=tf.float32)
        y_pred = tf.constant([[-0.5, 0.22, 0.47]])
        indicator = minion_losses.dummy_indicator(y_pred)
        cmp = tf.math.equal(y_true, indicator)
        print("ind:" + str(indicator))
        #print("cmp: "+ str(cmp))
        self.assertTrue(tf.keras.backend.all(cmp))


    def test_boolean_jaccard(self):
        y_true = tf.constant([1.0, 1.0, 1.0])
        y_pred = tf.constant([[0.0, 0.0, 1.0]])
        val = minion_losses.vary_boolean_jaccard(y_true, y_pred)
        # in this case the distance should be 2/3 (0.6666666666666667)

        expected = tf.constant([0.666666687],dtype=tf.float32)
        #kb.print_tensor(val)
        self.assertTrue(tf.math.equal(val, expected))

    def test_boolean_jaccard_multi(self):
        y_true = tf.constant([1.0, 1.0, 1.0])
        y_pred = tf.constant([[0.0, 0.0, 1.0], [0.0, 1.0, 1.0]])
        val = minion_losses.vary_boolean_jaccard(y_true, y_pred)
        # in this case the distance should be 2/3 (0.6666666666666667)

        expected = tf.constant([0.666666687, 0.333333343],dtype=tf.float32)
        cmp = tf.math.equal(expected, val)
        self.assertTrue(kb.all(cmp))

    def test_boolean_jaccard_multi_float_values(self):
        y_true = tf.constant([1.0, 1.0, 1.0])
        y_pred = tf.constant([[0.0, 0.0, 0.28], [0.0, 0.77, 0.02]])
        val = minion_losses.vary_boolean_jaccard(y_true, y_pred)
        # in this case the distance should be 2/3 (0.6666666666666667)

        expected = tf.constant([0.666666687, 0.333333343],dtype=tf.float32)
        print(val)
        cmp = tf.math.equal(expected, val)
        self.assertTrue(kb.all(cmp))

    def test_manahattan(self):
        y_true = tf.constant([[1.0, 1.0, 1.0]])
        y_pred = tf.constant([[0.0, 0.0, 1.0], [0.0, 0.0, 1.0]])
        val = minion_losses.vary_manhattan_dist(y_true, y_pred)
        # in this case the distance should be 2/3 (0.6666666666666667)

        expected = tf.constant([[2.0, 2.0]], dtype=tf.float32)
        cmp = tf.math.equal(val, expected)
        self.assertTrue(tf.keras.backend.all(cmp))

    def test_manahattan_indiv(self):
        y_true = tf.constant([[1.0, 1.0, 1.0]])
        y_pred = tf.constant([[0.0, 0.0, 1.0], [1.0, 0.0, 1.0]])
        val = minion_losses.vary_manhattan_dist_indiv(y_true, y_pred)

        expected = tf.constant([[1.0, 1.0, 0.0], [0.0, 1.0, 0.0]], dtype=tf.float32)
        cmp = tf.math.equal(val, expected)
        self.assertTrue(tf.keras.backend.all(cmp))

if __name__ == '__main__':
    unittest.main()
