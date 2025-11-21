from core import device, Tensor


def get_xor_data():
    xp = device.xp
    x = Tensor(xp.array([[0,0],[0,1],[1,0],[1,1]]))
    y = Tensor(xp.array([[0],[1],[1],[0]]))
    return x, y
    
def get_mnist_data():
    import urllib.request
    import gzip
    import os

    os.makedirs("./data/mnist", exist_ok=True)

    # Using a working mirror for MNIST data
    base_url = "https://storage.googleapis.com/tensorflow/tf-keras-datasets/"
    files = {
        "train_images": "train-images-idx3-ubyte.gz",
        "train_labels": "train-labels-idx1-ubyte.gz",
        "test_images": "t10k-images-idx3-ubyte.gz",
        "test_labels": "t10k-labels-idx1-ubyte.gz",
    }

    for key, filename in files.items():
        path = f"./data/mnist/{filename}"
        if not os.path.exists(path):
            print(f"Downloading {filename}...")
            urllib.request.urlretrieve(base_url + filename, path)
    
    def load_images(filename):
        import numpy as np
        with gzip.open(f"./data/mnist/{filename}", "rb") as f:
            data = np.frombuffer(f.read(), np.uint8, offset=16)
        data = data.reshape(-1, 28*28).astype(np.float32) / 255.0
        # Convert to target backend if needed
        if device.xp.__name__ != 'numpy':
            data = device.xp.array(data)
        return data
    
    def load_labels(filename):
        import numpy as np
        with gzip.open(f"./data/mnist/{filename}", "rb") as f:
            data = np.frombuffer(f.read(), np.uint8, offset=8)
        data = data.astype(np.int32)
        # Convert to target backend if needed
        if device.xp.__name__ != 'numpy':
            data = device.xp.array(data)
        return data
    
    x_train = Tensor(load_images(files["train_images"]))
    y_train = Tensor(load_labels(files["train_labels"]))
    x_test = Tensor(load_images(files["test_images"]))
    y_test = Tensor(load_labels(files["test_labels"]))

    return (x_train, y_train), (x_test, y_test)

def dataloader(x, y, batch_size=4):
    if batch_size is None:
        return [(x, y)]
    
    n_samples = x.data.shape[0]
    batches = []

    for i in range(0, n_samples, batch_size):
        end = min(i + batch_size, n_samples)
        x_batch = Tensor(x.data[i:end])
        y_batch = Tensor(y.data[i:end])
        batches.append((x_batch, y_batch))
    
    return batches