from core import device, Tensor

def get_imagenet_data():
    import os
    import numpy as np
    from PIL import Image
    from datasets import load_dataset
    from huggingface_hub import login
    
    token = os.getenv('HF_TOKEN')
    if not token:
        print("\n" + "="*70)
        print("HuggingFace authentication required for ImageNet-1k")
        print("="*70)
        print("\nSTEPS:")
        print("1. Go to: https://huggingface.co/settings/tokens")
        print("   - Create a new token with 'read' access")
        print("\n2. Go to: https://huggingface.co/datasets/ILSVRC/imagenet-1k")
        print("   - Click 'Access repository' and accept the terms")
        print("\n3. Enter your token below or set HF_TOKEN environment variable")
        print("="*70)
        token = input("\nEnter your HuggingFace token: ").strip()
        if not token:
            raise ValueError("Token is required to access ImageNet-1k dataset")
    
    print("Authenticating with HuggingFace...")
    login(token=token)
    print("Authentication successful!")
    
    print("\nLoading datasets (this may take a while on first run)...")
    train_dataset = load_dataset("ILSVRC/imagenet-1k", split="train", trust_remote_code=True, token=token)
    val_dataset = load_dataset("ILSVRC/imagenet-1k", split="validation", trust_remote_code=True, token=token)
    
    def process_image(img, size=227):
        """Resize and normalize image for AlexNet"""
        img = img.convert('RGB')
        img = img.resize((size, size), Image.BILINEAR) # type: ignore
        img_array = np.array(img).astype(np.float32) / 255.0
        # ImageNet normalization
        mean = np.array([0.485, 0.456, 0.406])
        std = np.array([0.229, 0.224, 0.225])
        img_array = (img_array - mean) / std
        img_array = np.transpose(img_array, (2, 0, 1))  # HWC to CHW
        return img_array
    
    print("Processing training images...")
    x_train_list = []
    y_train_list = []
    
    for idx, example in enumerate(train_dataset):
        if idx % 10000 == 0:
            print(f"  Processed {idx} training images...")
        img = process_image(example['image'])
        x_train_list.append(img)
        y_train_list.append(example['label'])
    
    print("Processing validation images...")
    x_val_list = []
    y_val_list = []
    
    for idx, example in enumerate(val_dataset):
        if idx % 5000 == 0:
            print(f"  Processed {idx} validation images...")
        img = process_image(example['image'])
        x_val_list.append(img)
        y_val_list.append(example['label'])
    
    x_train = np.array(x_train_list, dtype=np.float32)
    y_train = np.array(y_train_list, dtype=np.int32)
    x_val = np.array(x_val_list, dtype=np.float32)
    y_val = np.array(y_val_list, dtype=np.int32)
    
    print(f"Training set: {x_train.shape}, Validation set: {x_val.shape}")
    
    if device.xp.__name__ != 'numpy':
        x_train = device.xp.array(x_train)
        y_train = device.xp.array(y_train)
        x_val = device.xp.array(x_val)
        y_val = device.xp.array(y_val)
    
    return (Tensor(x_train), Tensor(y_train)), (Tensor(x_val), Tensor(y_val))


def get_cifar10_data():
    import urllib.request
    import tarfile
    import pickle
    import os
    import numpy as np

    os.makedirs("./data/cifar10", exist_ok=True)

    url = "https://www.cs.toronto.edu/~kriz/cifar-10-python.tar.gz"
    filepath = "./data/cifar10/cifar-10-python.tar.gz"

    if not os.path.exists(filepath):
        print("Downloading CIFAR-10...")
        urllib.request.urlretrieve(url, filepath)
    
    if not os.path.exists("./data/cifar10/cifar-10-batches-py"):
        print("Extracting...")
        with tarfile.open(filepath, "r:gz") as tar:
            tar.extractall(path="./data/cifar10/")
    
    def load_batch(filename):
        with open(filename, "rb") as f:
            batch = pickle.load(f, encoding="bytes")
        data = batch[b"data"].astype(np.float32) / 255.0
        labels = np.array(batch[b"labels"], dtype=np.int32)
        data = data.reshape(-1, 3, 32, 32)
        return data, labels
    
    x_train_list = []
    y_train_list = []
    for i in range(1, 6):
        data, labels = load_batch(f"./data/cifar10/cifar-10-batches-py/data_batch_{i}")
        x_train_list.append(data)
        y_train_list.append(labels)

    x_train = np.concatenate(x_train_list, axis=0)
    y_train = np.concatenate(y_train_list, axis=0)

    x_test, y_test = load_batch("./data/cifar10/cifar-10-batches-py/test_batch")

    if device.xp.__name__ != 'numpy':
        x_train = device.xp.array(x_train)
        y_train = device.xp.array(y_train)
        x_test = device.xp.array(x_test)
        y_test = device.xp.array(y_test)
    
    return (Tensor(x_train), Tensor(y_train)), (Tensor(x_test), Tensor(y_test))

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

def get_xor_data():
    xp = device.xp
    x = Tensor(xp.array([[0,0],[0,1],[1,0],[1,1]]))
    y = Tensor(xp.array([[0],[1],[1],[0]]))
    return x, y

def dataloader(x, y, batch_size=4, shuffle=False):
    """Returns a callable that generates batches. Call it to get a fresh generator."""
    def generate_batches():
        if batch_size is None:
            yield (x, y)
            return
        
        n_samples = x.data.shape[0]
        indices = device.xp.arange(n_samples)
        
        if shuffle:
            # Shuffle indices
            if device.xp.__name__ == 'mlx.core':
                import numpy as np
                indices_np = np.arange(n_samples)
                np.random.shuffle(indices_np)
                indices = device.xp.array(indices_np)
            else:
                device.xp.random.shuffle(indices)

        for i in range(0, n_samples, batch_size):
            end = min(i + batch_size, n_samples)
            batch_indices = indices[i:end]
            # Create tensors without requiring gradients to avoid building computation graph
            x_batch = Tensor(x.data[batch_indices], req_grad=False)
            y_batch = Tensor(y.data[batch_indices], req_grad=False)
            yield (x_batch, y_batch)
    
    return generate_batches