import time
from core import Network, Linear, Tanh, Sigmoid, optim
from datasets import get_xor_data, dataloader


def train_xor(epochs=5000, lr=.01):
    net = Network()
    net.add(Linear(2, 4))
    net.add(Tanh())
    net.add(Linear(4, 1))
    net.add(Sigmoid())

    x, y = get_xor_data()

    optimizer = optim.SGD(net.parameters(), lr=lr)

    start = time.time()
    history = net.fit(
        dataloader(x, y), 
        loss_name="bce_loss", 
        optimizer=optimizer, 
        epochs=epochs)
    dur = time.time() - start

    print(f"\nTraining completed in {dur:.2f} seconds.")
    print(f"Final Loss: {history[-1]:.4f}")

    net.eval()
    preds = net(x).data
    print("\nInputs:\n", x.data)
    print("Targets:\n", y.data)
    print("Predictions:\n", preds)

if __name__ == "__main__":
    train_xor()