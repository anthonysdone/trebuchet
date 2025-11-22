import time
import sys
import os

from core import device, scheduler
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from core import Network, optim, Linear, Relu
from plugins.conv import Conv2d, MaxPool2d, Flatten, LocalResponseNorm, Dropout
from examples.dataset import get_imagenet_data, dataloader

def train_alexnet(epochs=90, lr=0.01, save_path=None, batch_size=128):

    device.use_gpu()
    print(f"Using device: {device.xp.__name__}")

    net = Network()

    # Conv Block 1
    net.add(Conv2d(3, 96, kernel_size=11, stride=4, padding=2))
    net.add(Relu())
    net.add(LocalResponseNorm(size=5, alpha=1e-4, beta=0.75, k=2))
    net.add(MaxPool2d(kernel_size=3, stride=2))

    # Conv Block 2
    net.add(Conv2d(96, 256, kernel_size=5, padding=2))
    net.add(Relu())
    net.add(LocalResponseNorm(size=5, alpha=1e-4, beta=0.75, k=2))
    net.add(MaxPool2d(kernel_size=3, stride=2))

    # Conv Block 3
    net.add(Conv2d(384, 384, kernel_size=3, padding=1))
    net.add(Relu())
    net.add(MaxPool2d(kernel_size=3, stride=2))

    # MLP
    net.add(Flatten())
    net.add(Linear(384 * 6 * 6, 4096))
    net.add(Relu())
    net.add(Dropout(p=0.5))
    net.add(Linear(4096, 4096))
    net.add(Relu())
    net.add(Dropout(p=0.5))
    net.add(Linear(4096, 1000))

    (x_train, y_train), (x_test, y_test) = get_imagenet_data()

    optimizer = optim.SGD(net.parameters(), lr=lr, momentum=0.9, weight_decay=5e-4)

    lr_scheduler = scheduler.MultiStepLR(optimizer, milestones=[30, 60, 80], gamma=0.1)

    start = time.time()
    history = net.fit(
        dataloader(x_train, y_train, batch_size=batch_size, shuffle=True),
        loss_name="cross_entropy_loss",
        optimizer=optimizer,
        epochs=epochs,
        scheduler=lr_scheduler
    )
    dur = time.time() - start

    print(f"\nTraining completed in {dur/60:.2f} minutes.")
    print(f"Final Loss: {history[-1]:.4f}")

    if save_path:
        os.makedirs(os.path.dirname(save_path) if os.path.dirname(save_path) else ".", exist_ok=True)
        net.save(save_path)

    net.eval()

    correct_top1 = 0
    correct_top5 = 0
    total = 0

    for x_batch, y_batch in dataloader(x_test, y_test, batch_size=batch_size)(): 
        logits = net(x_batch).data
        predicted_top5 = logits.argsort(axis=1)[:, -5:]
        predicted_top1 = logits.argmax(axis=1)

        correct_top1 += (predicted_top1 == y_batch.data).sum()
        correct_top5 += sum([1 if y_batch.data[i] in predicted_top5[i] else 0 for i in range(y_batch.data.shape[0])])
        total += y_batch.data.shape[0]
    
    top1_acc = correct_top1 / total
    top5_acc = correct_top5 / total

    print(f"Top-1 Accuracy: {top1_acc * 100:.2f}%")
    print(f"Top-5 Accuracy: {top5_acc * 100:.2f}%")

if __name__ == "__main__":
    train_alexnet(
        epochs=90, 
        batch_size=256, 
        lr=0.01, 
        save_path="./models/alexnet_model.pkl"
    )