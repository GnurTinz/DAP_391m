import math

class ConstantScheduler:
    def __init__(self, value):
        self.value = float(value)
        
    def get(self, epoch, step=None):
        return self.value

class LinearAnnealingScheduler:
    def __init__(self, start_epoch, end_epoch, start_value, end_value):
        self.start_epoch = int(start_epoch)
        self.end_epoch = int(end_epoch)
        self.start_value = float(start_value)
        self.end_value = float(end_value)
        
    def get(self, epoch, step=None):
        if epoch <= self.start_epoch:
            return self.start_value
        if epoch >= self.end_epoch:
            return self.end_value
        progress = (epoch - self.start_epoch) / float(self.end_epoch - self.start_epoch)
        return self.start_value + progress * (self.end_value - self.start_value)

class LinearStepScheduler:
    """
    Tăng dần tuyến tính nhưng tính theo từng Step (batch) thay vì Epoch.
    Rất thích hợp cho Warmup.
    """
    def __init__(self, start_step, end_step, start_value, end_value):
        self.start_step = int(start_step)
        self.end_step = int(end_step)
        self.start_value = float(start_value)
        self.end_value = float(end_value)
        
    def get(self, epoch, step=None):
        if step is None or step <= self.start_step:
            return self.start_value
        if step >= self.end_step:
            return self.end_value
        progress = (step - self.start_step) / float(self.end_step - self.start_step)
        return self.start_value + progress * (self.end_value - self.start_value)

class StepScheduler:
    def __init__(self, start_epoch, value):
        self.start_epoch = int(start_epoch)
        self.value = float(value)
        
    def get(self, epoch, step=None):
        if epoch < self.start_epoch:
            return 0.0
        return self.value

class CyclicScheduler:
    """
    Tăng giảm liên tục theo hình sin (Cosine Annealing tuần hoàn).
    Dựa trên step (batch) thay vì epoch để có chu kỳ linh hoạt.
    """
    def __init__(self, start_step, cycle_length, min_value, max_value):
        self.start_step = int(start_step)
        self.cycle_length = int(cycle_length)
        self.min_value = float(min_value)
        self.max_value = float(max_value)
        
    def get(self, epoch, step=None):
        if step is None or step < self.start_step:
            return self.min_value
            
        phase = ((step - self.start_step) % self.cycle_length) / self.cycle_length
        # Hàm -cos đi từ -1 -> 1 -> -1. 
        # Chuyển đổi về khoảng [0, 1]
        val = 0.5 * (1 - math.cos(phase * 2 * math.pi))
        return self.min_value + val * (self.max_value - self.min_value)


class LossSchedulerManager:
    """
    Manages multiple loss schedules based on a configuration dictionary.
    """
    def __init__(self, schedules_config):
        self.schedules = {}
        for loss_name, cfg in schedules_config.items():
            stype = cfg.get('type', 'constant').lower()
            if stype == 'constant':
                self.schedules[loss_name] = ConstantScheduler(cfg.get('value', 0.0))
            elif stype == 'linear':
                self.schedules[loss_name] = LinearAnnealingScheduler(
                    cfg.get('start_epoch', 0),
                    cfg.get('end_epoch', 10),
                    cfg.get('start_value', 0.0),
                    cfg.get('end_value', 1.0)
                )
            elif stype == 'linear_step':
                self.schedules[loss_name] = LinearStepScheduler(
                    cfg.get('start_step', 0),
                    cfg.get('end_step', 1000),
                    cfg.get('start_value', 0.0),
                    cfg.get('end_value', 1.0)
                )
            elif stype == 'step':
                self.schedules[loss_name] = StepScheduler(
                    cfg.get('start_epoch', 0),
                    cfg.get('value', 1.0)
                )
            elif stype == 'cyclic':
                self.schedules[loss_name] = CyclicScheduler(
                    cfg.get('start_step', 0),
                    cfg.get('cycle_length', 100),
                    cfg.get('min_value', 0.0),
                    cfg.get('max_value', 1.0)
                )
            else:
                raise ValueError(f"Unknown scheduler type: {stype}")
                
    def get_weights(self, epoch, step=None):
        """
        Returns a dictionary of {loss_name: current_weight} for the given epoch and step.
        """
        return {name: scheduler.get(epoch, step) for name, scheduler in self.schedules.items()}
