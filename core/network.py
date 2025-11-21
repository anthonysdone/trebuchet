from .registry import LAYERS, OPS, OPTIMS
from .layer import Layer
import sys

class Network: 
    def __init__(self, layers=[]):
        self.layers = []
        if layers:
            for spec in layers:
                self.add(spec)
    
    def add(self, spec):
        layer = self.build_layer(spec)
        self.layers.append(layer)
        return layer

    def build_layer(self, spec):
        if isinstance(spec, Layer):
            return spec
        if isinstance(spec, str):
            return LAYERS[spec]()
        if isinstance(spec, dict):
            name = spec["name"]
            kwargs = spec.get("params", {})
            return LAYERS[name](**kwargs)
        raise ValueError(f"[Network]: Invalid layer specification: {spec}")
    
    def train(self): 
        for layer in self.layers:
            layer.train()
    
    def eval(self):
        for layer in self.layers:
            layer.eval()
    
    def forward(self, x):
        for layer in self.layers:
            x = layer(x)
        return x
    
    __call__ = forward

    def parameters(self):
        params = []
        for layer in self.layers:
            params.extend(layer.parameters())
        return params
    
    def zero_grad(self):
        for param in self.parameters():
            param.zero_grad()
    
    def loss(self, pred, target, loss_name):
        loss_fn = OPS[loss_name]
        return loss_fn(pred, target)
    
    def train_step(self, x, y, loss_name, optimizer):
        preds = self.forward(x)
        loss = self.loss(preds, y, loss_name)
        loss.backward()
        optimizer.step()
        return {"pred": preds, "loss": loss}    
    
    def default_callback(self, epoch, batch_idx, total_batches, stats):
        loss_val = stats["loss"].data if hasattr(stats["loss"], "data") else stats["loss"]
        progress = (batch_idx + 1) / total_batches
        bar_length = 30
        filled = int(bar_length * progress)
        bar = "█" * filled + "░" * (bar_length - filled)

        sys.stdout.write(f"\rEpoch {epoch + 1} [{bar}] {batch_idx + 1}/{total_batches} - Loss: {loss_val:.4f}")
        sys.stdout.flush()

        if batch_idx + 1 == total_batches:
            sys.stdout.write("\n")
    
    def fit(self, dataloader, loss_name, optimizer, epochs=1, callback=None):
        history = []
        for epoch in range(epochs):
            epoch_loss = 0.0
            batch_count = 0

            batches = list(dataloader)
            total_batches = len(batches)

            for batch_idx, (x_batch, y_batch) in enumerate(batches):
                stats = self.train_step(x_batch, y_batch, loss_name, optimizer)
                epoch_loss += stats["loss"].data
                batch_count += 1
                if callback is None:
                    self.default_callback(epoch, batch_idx, total_batches, stats)
            history.append(epoch_loss / batch_count if batch_count > 0 else 0)
        return history