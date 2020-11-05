import unittest
import training_Model_tensorflow as train_model


class MyTestCase(unittest.TestCase):

    def test_device_information(self):
        train_model.get_tf_device_type()
        print(train_model.TENSORFLOW_DEVICE)
        self.assertIsNotNone(train_model.TENSORFLOW_DEVICE, msg='Tensorflow device cannot be None')


if __name__ == '__main__':
    unittest.main()
