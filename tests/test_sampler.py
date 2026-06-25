import unittest
from src.datasets.sampler import PKSampler, RandomClassSampler, WeightedClassSampler, get_sampler

class TestSamplers(unittest.TestCase):
    def setUp(self):
        # Giả lập 5 classes, mỗi class 10 samples -> Tổng 50 samples
        self.labels = [i // 10 for i in range(50)]

    def test_pk_sampler_basic(self):
        p = 3
        k = 4
        sampler = PKSampler(self.labels, p=p, k=k)
        
        # Batch size phải là p * k
        self.assertEqual(sampler.batch_size, 12)
        
        for batch in sampler:
            self.assertEqual(len(batch), 12)
            batch_labels = [self.labels[idx] for idx in batch]
            unique_classes = set(batch_labels)
            self.assertEqual(len(unique_classes), p)
            for c in unique_classes:
                self.assertEqual(batch_labels.count(c), k)

    def test_random_class_sampler(self):
        num_classes = 2
        sampler = RandomClassSampler(self.labels, num_classes_per_batch=num_classes)
        
        for batch in sampler:
            batch_labels = [self.labels[idx] for idx in batch]
            unique_classes = set(batch_labels)
            self.assertEqual(len(unique_classes), num_classes)
            # Vì mỗi class có 10 mẫu, 2 class sẽ có đúng 20 mẫu
            self.assertEqual(len(batch), 20)

    def test_weighted_class_sampler(self):
        batch_size = 16
        sampler = WeightedClassSampler(self.labels, batch_size=batch_size)
        
        for batch in sampler:
            self.assertEqual(len(batch), batch_size)

    def test_get_sampler_factory(self):
        sampler1 = get_sampler('pk_sampler', self.labels, batch_size=32, p=8, k=4)
        self.assertIsInstance(sampler1, PKSampler)
        
        sampler2 = get_sampler('random_class', self.labels, batch_size=32, num_classes=4)
        self.assertIsInstance(sampler2, RandomClassSampler)
        
        sampler3 = get_sampler('weighted', self.labels, batch_size=32)
        self.assertIsInstance(sampler3, WeightedClassSampler)

if __name__ == '__main__':
    unittest.main()
