import time
from core import *
from core.layer import Linear, Sigmoid, Tanh
from core.optim import SGD

def simple_dataloader(x, y, batch_size=4):
    yield (x, y)

def train_xor(epochs=5000, lr=.01):
    net = Network()
    net.add(Linear(2, 4))
    net.add(Tanh())
    net.add(Linear(4, 1))
    net.add(Sigmoid())

    xp = device.xp
    x = Tensor(xp.array([[0,0],[0,1],[1,0],[1,1]]))
    y = Tensor(xp.array([[0],[1],[1],[0]]))

    optimizer = SGD(net.parameters(), lr=lr)

    start = time.time()
    history = net.fit(
        simple_dataloader(x, y, batch_size=4), 
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