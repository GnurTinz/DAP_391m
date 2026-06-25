import unittest
from src.datasets.sampler import PKSampler

class TestPKSampler(unittest.TestCase):
    def test_pk_sampler_basic(self):
        # Giả lập 5 classes, mỗi class 10 samples -> Tổng 50 samples
        labels = [i // 10 for i in range(50)]
        
        p = 3
        k = 4
        sampler = PKSampler(labels, p=p, k=k)
        
        # Batch size phải là p * k
        self.assertEqual(sampler.batch_size, 12)
        
        for batch in sampler:
            self.assertEqual(len(batch), 12)
            
            # Kiểm tra xem có đúng P class và mỗi class K mẫu không
            batch_labels = [labels[idx] for idx in batch]
            unique_classes = set(batch_labels)
            
            # Do replace=True có thể xảy ra nếu số class không đủ, 
            # nhưng ở đây có 5 class > P=3 nên unique_classes phải bằng P
            self.assertEqual(len(unique_classes), p)
            
            for c in unique_classes:
                self.assertEqual(batch_labels.count(c), k)

if __name__ == '__main__':
    unittest.main()
