import unittest
import torch
import os
import shutil
from src.datasets.palm_dataset import PalmPrintDataset
from src.datasets.own_dataset import OwnDataset
from src.datasets.factory import DatasetFactory
from src.datasets.base import BaseDataset

class TestPalmPrintDataset(unittest.TestCase):
    def setUp(self):
        self.config = {
            'image_size': [128, 128]
        }
        self.dummy_data_dir = 'dummy_test_data_dir'
        
    def tearDown(self):
        if os.path.exists(self.dummy_data_dir):
            shutil.rmtree(self.dummy_data_dir)

    def test_dataset_initialization_dry_run(self):
        dataset = PalmPrintDataset(self.dummy_data_dir, self.config, is_train=True)
        self.assertEqual(len(dataset), 100, "Dummy dataset phải có độ dài 100")
        
    def test_dataset_getitem_dry_run(self):
        dataset = PalmPrintDataset(self.dummy_data_dir, self.config, is_train=False)
        img, label = dataset[0]
        self.assertEqual(img.shape, (3, 128, 128), "Image shape không khớp với config")
        self.assertIsInstance(label, int, "Label phải là kiểu int")
        
    def test_transforms_exist(self):
        dataset_train = PalmPrintDataset(self.dummy_data_dir, self.config, is_train=True)
        dataset_test = PalmPrintDataset(self.dummy_data_dir, self.config, is_train=False)
        self.assertIsNotNone(dataset_train.transform)
        self.assertIsNotNone(dataset_test.transform)
        self.assertTrue(len(dataset_train.transform.transforms) > len(dataset_test.transform.transforms))

class TestOwnDataset(unittest.TestCase):
    def setUp(self):
        self.config = {
            'image_size': [32, 32]
        }
        self.dummy_data_dir = 'dummy_own_data_dir'
        
    def tearDown(self):
        if os.path.exists(self.dummy_data_dir):
            shutil.rmtree(self.dummy_data_dir)

    def test_own_dataset_initialization_dry_run(self):
        dataset = OwnDataset(self.dummy_data_dir, self.config, is_train=True)
        self.assertEqual(len(dataset), 100, "Dummy OwnDataset phải có độ dài 100")
        
    def test_own_dataset_getitem_dry_run(self):
        dataset = OwnDataset(self.dummy_data_dir, self.config, is_train=False)
        img, label = dataset[0]
        self.assertEqual(img.shape, (3, 32, 32), "Image shape không khớp với config")
        self.assertIsInstance(label, int, "Label phải là kiểu int")
        
    def test_own_transforms_exist(self):
        dataset_train = OwnDataset(self.dummy_data_dir, self.config, is_train=True)
        dataset_test = OwnDataset(self.dummy_data_dir, self.config, is_train=False)
        self.assertIsNotNone(dataset_train.transform)
        self.assertIsNotNone(dataset_test.transform)
        self.assertTrue(len(dataset_train.transform.transforms) > len(dataset_test.transform.transforms))

class TestDatasetFactory(unittest.TestCase):
    def setUp(self):
        self.config = {
            'image_size': [64, 64]
        }
        self.dummy_data_dir = 'dummy_factory_dir'
        
    def test_factory_create_palmprint(self):
        dataset = DatasetFactory.create('PalmPrintDataset', self.dummy_data_dir, self.config)
        self.assertIsInstance(dataset, PalmPrintDataset)
        self.assertTrue(issubclass(type(dataset), BaseDataset))
        
    def test_factory_create_owndataset(self):
        dataset = DatasetFactory.create('OwnDataset', self.dummy_data_dir, self.config)
        self.assertIsInstance(dataset, OwnDataset)
        
    def test_factory_invalid_dataset(self):
        with self.assertRaises(ValueError):
            DatasetFactory.create('NonExistentDataset', self.dummy_data_dir, self.config)

if __name__ == '__main__':
    unittest.main()
