from .registry import register_scheduler

class LRScheduler:
    def __init__(self, optimizer): 
        self.optimizer = optimizer
        self.base_lr = optimizer.lr

    def step(self, epoch, loss=None): 
        raise NotImplementedError
    
    def get_lr(self):
        return self.optimizer.lr
    
@register_scheduler("multisteplr")
class MultiStepLR(LRScheduler):
    def __init__(self, optimizer, milestones, gamma=0.1):
        super().__init__(optimizer)
        self.milestones = set(milestones)
        self.gamma = gamma

    def step(self, epoch, loss=None):
        if epoch in self.milestones:
            new_lr = self.optimizer.lr * self.gamma
            print(f"Adjusting learning rate to {new_lr:.6f} at epoch {epoch}")
            self.optimizer.lr = new_lr


@register_scheduler("plateaulr")
class PlateauLR(LRScheduler): 
    def __init__(self, optimizer, mode="min", factor=0.5, patience=5, threshold=1e-4, min_lr=1e-6):
        super().__init__(optimizer)
        self.mode = mode
        self.factor = factor
        self.patience = patience
        self.threshold = threshold
        self.min_lr = min_lr
        self.best_loss = float("inf") if mode == "min" else -float("inf")
        self.num_bad_epochs = 0

    def step(self, epoch, loss=None):
        if loss is None:
            return 
        
        if self.mode == "min": 
            is_better = loss < self.best_loss - self.threshold
        else: 
            is_better = loss > self.best_loss + self.threshold
        
        if is_better: 
            self.best_loss = loss
            self.num_bad_epochs = 0
        else:
            self.num_bad_epochs += 1
        
        if self.num_bad_epochs >= self.patience: 
            new_lr = max(self.optimizer.lr * self.factor, self.min_lr)
            if new_lr < self.optimizer.lr: 
                print(f"Reducing learning rate to {new_lr:.6f}")
                self.optimizer.lr = new_lr
                self.num_bad_epochs = 0
