from core import device
from .registry import LAYERS, OPS, OPTIMS, SCHEDULERS
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
    
    def fit(self, dataloader, loss_name, optimizer, epochs=1, callback=None, scheduler=None):
        history = []
        for epoch in range(epochs):
            epoch_loss = 0.0
            batch_count = 0

            # Get fresh generator/iterable for this epoch
            batches_list = list(dataloader() if callable(dataloader) else dataloader) # type: ignore
            total_batches = len(batches_list)

            for batch_idx, (x_batch, y_batch) in enumerate(batches_list):
                stats = self.train_step(x_batch, y_batch, loss_name, optimizer)
                loss_val = stats["loss"].data
                epoch_loss += float(loss_val) if hasattr(loss_val, 'item') else float(loss_val)
                batch_count += 1
                if callback is None:
                    self.default_callback(epoch, batch_idx, total_batches, stats)
                del x_batch, y_batch, stats
            del batches_list
            
            history.append(epoch_loss / batch_count if batch_count > 0 else 0)

            if scheduler is not None: 
                scheduler.step(epoch, loss=history[-1])

        return history
    
    def save(self, filepath): 
        import pickle

        state = {
            "layers": [],
            "training": self.layers[0].training if self.layers else True
        }

        for layer in self.layers: 
            layer_state = {
                "type": layer.__class__.__name__,
                "params": {}
            }
            
            if hasattr(layer, "parameters") and callable(layer.parameters):
                params = layer.parameters()
                if params and isinstance(params, (list, tuple)):  # Only process if params is iterable
                    for i, param in enumerate(params): 
                        if device.xp.__name__ != "numpy": 
                            import numpy as np
                            layer_state["params"][f"param_{i}"] = np.array(param.data)
                        else: 
                            layer_state["params"][f"param_{i}"] = param.data.copy()
                
            if hasattr(layer, "__dict__"):
                attrs = {}
                for key, value in layer.__dict__.items():
                    if key not in ["weight", "bias", "gamma", "beta", "training"]: 
                        if isinstance(value, (int, float, str, bool, tuple, list, dict)):
                            attrs[key] = value
                        elif device.xp.__name__ != "numpy" and hasattr(value, "__array__"):
                            import numpy as np
                            attrs[key] = np.array(value)
                        elif isinstance(value, device.xp.ndarray):
                            attrs[key] = value.copy()
                layer_state["attrs"] = attrs
            state["layers"].append(layer_state)
        
        with open(filepath, "wb") as f:
            pickle.dump(state, f)
        
        print(f"Model saved to {filepath}")

    def load(self, filepath):
        import pickle
        
        with open(filepath, 'rb') as f:
            state = pickle.load(f)
        
        if len(state['layers']) != len(self.layers):
            raise ValueError(f"Architecture mismatch: saved model has {len(state['layers'])} layers, current model has {len(self.layers)} layers")
        
        for i, (layer, layer_state) in enumerate(zip(self.layers, state['layers'])):
            if layer.__class__.__name__ != layer_state['type']:
                raise ValueError(f"Layer {i} type mismatch: expected {layer_state['type']}, got {layer.__class__.__name__}")
            
            if hasattr(layer, 'parameters') and callable(layer.parameters):
                params = layer.parameters()
                if params and isinstance(params, (list, tuple)):  # Only process if params is iterable
                    for j, param in enumerate(params):
                        param_key = f'param_{j}'
                        if param_key in layer_state['params']:
                            loaded_data = layer_state['params'][param_key]
                            # Convert back to device backend
                            if device.xp.__name__ != 'numpy':
                                loaded_data = device.xp.array(loaded_data)
                            param.data = loaded_data
            
            if 'attrs' in layer_state:
                for key, value in layer_state['attrs'].items():
                    if hasattr(layer, key):
                        if device.xp.__name__ != 'numpy' and isinstance(value, device.xp.ndarray):
                            value = device.xp.array(value)
                        setattr(layer, key, value)
        
        if state['training']:
            self.train()
        else:
            self.eval()
        
        print(f"Model loaded from {filepath}")