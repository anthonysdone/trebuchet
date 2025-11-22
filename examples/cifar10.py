import time
import sys
import os
import gc

from core import device, scheduler
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from core import Network, optim, Linear, Relu
from plugins.conv import Conv2d, MaxPool2d, Flatten
from examples.dataset import get_cifar10_data, dataloader

def train_cifar10(epochs=10, lr=.001, batch_size=64):

    device.use_gpu()
    print(f"Using device: {device.xp.__name__}") 
    
    net = Network()

    # Conv Block 1 
    net.add(Conv2d(3, 64, kernel_size=3, padding=1))
    net.add(Relu())
    net.add(MaxPool2d(kernel_size=2, stride=2))

    # Conv Block 2
    net.add(Conv2d(64, 128, kernel_size=3, padding=1))
    net.add(Relu())
    net.add(MaxPool2d(kernel_size=2, stride=2))

    # Conv Block 3
    net.add(Conv2d(128, 128, kernel_size=3, padding=1))
    net.add(Relu())
    net.add(MaxPool2d(kernel_size=2, stride=2))

    # MLP
    net.add(Flatten())
    net.add(Linear(128 * 4 * 4, 256))
    net.add(Relu())
    net.add(Linear(256, 10))

    (x_train, y_train), (x_test, y_test) = get_cifar10_data()

    optimizer = optim.Adam(net.parameters(), lr=lr)

    # Use scheduler from scheduler module with more aggressive settings
    lr_scheduler = scheduler.PlateauLR(optimizer, patience=3, factor=0.5, min_lr=1e-6)

    start = time.time()
    history = net.fit(
        dataloader(x_train, y_train, batch_size=batch_size, shuffle=True),
        loss_name="cross_entropy_loss",
        optimizer=optimizer,
        epochs=epochs,
        scheduler=lr_scheduler)
    dur = time.time() - start

    print(f"\nTraining completed in {dur:.2f} seconds.")
    print(f"Final Loss: {history[-1]:.4f}")

    net.eval()

    correct = 0
    total = 0

    for x_batch, y_batch in dataloader(x_test, y_test, batch_size=batch_size)(): 
        logits = net(x_batch).data
        predicted = logits.argmax(axis=1)
        correct += (predicted == y_batch.data).sum()
        total += y_batch.data.shape[0]
    
    accuracy = correct / total
    print(f"Accuracy: {accuracy * 100:.2f}%")

if __name__ == "__main__":
    # Reduced batch size for M4 MacBook with 16GB RAM
    train_cifar10(epochs=40, batch_size=32, lr=0.001)