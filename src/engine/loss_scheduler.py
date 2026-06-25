class ConstantScheduler:
    def __init__(self, value):
        self.value = float(value)
        
    def get(self, epoch):
        return self.value

class LinearAnnealingScheduler:
    def __init__(self, start_epoch, end_epoch, start_value, end_value):
        self.start_epoch = int(start_epoch)
        self.end_epoch = int(end_epoch)
        self.start_value = float(start_value)
        self.end_value = float(end_value)
        
    def get(self, epoch):
        if epoch <= self.start_epoch:
            return self.start_value
        if epoch >= self.end_epoch:
            return self.end_value
        progress = (epoch - self.start_epoch) / float(self.end_epoch - self.start_epoch)
        return self.start_value + progress * (self.end_value - self.start_value)

class StepScheduler:
    def __init__(self, start_epoch, value):
        self.start_epoch = int(start_epoch)
        self.value = float(value)
        
    def get(self, epoch):
        if epoch < self.start_epoch:
            return 0.0
        return self.value

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
            elif stype == 'step':
                self.schedules[loss_name] = StepScheduler(
                    cfg.get('start_epoch', 0),
                    cfg.get('value', 1.0)
                )
            else:
                raise ValueError(f"Unknown scheduler type: {stype}")
                
    def get_weights(self, epoch):
        """
        Returns a dictionary of {loss_name: current_weight} for the given epoch.
        """
        return {name: scheduler.get(epoch) for name, scheduler in self.schedules.items()}
