import unittest
from src.engine.loss_scheduler import LossSchedulerManager, ConstantScheduler, LinearAnnealingScheduler, StepScheduler

class TestLossScheduler(unittest.TestCase):
    def test_constant_scheduler(self):
        scheduler = ConstantScheduler(0.5)
        self.assertEqual(scheduler.get(0), 0.5)
        self.assertEqual(scheduler.get(10), 0.5)

    def test_linear_annealing_scheduler(self):
        scheduler = LinearAnnealingScheduler(start_epoch=0, end_epoch=10, start_value=0.0, end_value=1.0)
        self.assertEqual(scheduler.get(0), 0.0)
        self.assertEqual(scheduler.get(5), 0.5)
        self.assertEqual(scheduler.get(10), 1.0)
        self.assertEqual(scheduler.get(15), 1.0)

    def test_step_scheduler(self):
        scheduler = StepScheduler(start_epoch=5, value=0.8)
        self.assertEqual(scheduler.get(0), 0.0)
        self.assertEqual(scheduler.get(4), 0.0)
        self.assertEqual(scheduler.get(5), 0.8)
        self.assertEqual(scheduler.get(10), 0.8)

    def test_manager(self):
        config = {
            'kl': {'type': 'linear', 'start_epoch': 0, 'end_epoch': 10, 'start_value': 0.0, 'end_value': 1.0},
            'con': {'type': 'step', 'start_epoch': 5, 'value': 0.5},
            'rec': {'type': 'constant', 'value': 1.0}
        }
        manager = LossSchedulerManager(config)
        
        weights_epoch_0 = manager.get_weights(0)
        self.assertEqual(weights_epoch_0['kl'], 0.0)
        self.assertEqual(weights_epoch_0['con'], 0.0)
        self.assertEqual(weights_epoch_0['rec'], 1.0)
        
        weights_epoch_5 = manager.get_weights(5)
        self.assertEqual(weights_epoch_5['kl'], 0.5)
        self.assertEqual(weights_epoch_5['con'], 0.5)
        self.assertEqual(weights_epoch_5['rec'], 1.0)

if __name__ == '__main__':
    unittest.main()
