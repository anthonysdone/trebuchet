import time
from core import Network, Linear, Sigmoid, optim
from core import device
from examples.dataset import get_mnist_data, dataloader


def train_mnist(epochs=5, lr=.001, batch_size=10):
    
    net = Network()
    net.add(Linear(28*28, 128))
    net.add(Sigmoid())
    net.add(Linear(128, 64))
    net.add(Sigmoid())
    net.add(Linear(64, 10))

    # if you want to test gpu vs cpu try this very large mlp
    # device.use_gpu()
    # net = Network()
    # net.add(Linear(28*28, 50*50))
    # net.add(Relu())
    # net.add(Linear(50*50, 40*40))
    # net.add(Relu())
    # net.add(Linear(40*40, 30*30))
    # net.add(Relu())
    # net.add(Linear(30*30, 20*20))
    # net.add(Relu())
    # net.add(Linear(20*20, 10*10))
    # net.add(Relu())
    # net.add(Linear(10*10, 10))

    (x_train, y_train), (x_test, y_test) = get_mnist_data()

    optimizer = optim.SGD(net.parameters(), lr=lr)

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