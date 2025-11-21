import time
from core import Network, Linear, Relu, optim
from datasets import get_mnist_data, dataloader


def train_mnist(epochs=3, lr=.0005, batch_size=32):
    net = Network()
    net.add(Linear(28*28, 256))
    net.add(Relu())
    net.add(Linear(256, 64))
    net.add(Relu())
    net.add(Linear(64, 10))

    (x_train, y_train), (x_test, y_test) = get_mnist_data()

    optimizer = optim.Adam(net.parameters(), lr=lr)

    start = time.time()
    history = net.fit(
        dataloader(x_train, y_train, batch_size=batch_size), 
        loss_name="cross_entropy_loss",  
        optimizer=optimizer, 
        epochs=epochs)
    dur = time.time() - start

    print(f"\nTraining completed in {dur:.2f} seconds.")
    print(f"Final Loss: {history[-1]:.4f}")

    net.eval()
    # Network now outputs logits, so just take argmax (no softmax needed for classification)
    logits = net(x_test).data
    predicted_classes = logits.argmax(axis=1)
    accuracy = (predicted_classes == y_test.data.flatten()).mean()
    print(f"Accuracy: {accuracy * 100:.2f}%")

if __name__ == "__main__":
    train_mnist()