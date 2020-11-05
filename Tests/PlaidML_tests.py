import unittest
import training_Model_plaidml as train_model

class MyTestCase(unittest.TestCase):
    def test_device_information(self):
        train_model.get_device_information()
        print(train_model.PLAIDML_IDS)
        self.assertIsNotNone(train_model.PLAIDML_IDS, msg="The retrieved processor type cannot be None")


if __name__ == '__main__':
    unittest.main()
