import torch
from d2l import torch as d2l
from torch import nn
from torch.nn import functional as F
import matplotlib.pyplot as plt

class NonlinearData(d2l.DataModule):
    def __init__(self, n, batch_size):
        super().__init__()
        self.save_hyperparameters()
        f = lambda x: 2 * torch.sin(x) + x ** 0.8
        self.x_train, _ = torch.sort(torch.rand(n) * 5)
        self.y_train = f(self.x_train) + torch.randn(n)
        self.x_val = torch.arange(0, 5, 5.0 / n)
        self.y_val = f(self.x_val)

    def get_dataloader(self, train):
        arrays = (self.x_train, self.y_train) if train else (self.x_val, self.y_val)
        return self.get_tensorloader(arrays, train)


n = 3
data = NonlinearData(n, batch_size=3)


def plot_kernel_reg(y_hat):
    d2l.plot(data.x_val, [data.y_val, y_hat.detach().numpy()], 'x', 'y', legend=['Truth', 'Pred'],
             xlim=[0, 5], ylim=[-1, 5])
    d2l.plt.plot(data.x_train, data.y_train, 'o', alpha=0.5)
    plt.show()


def diff(queries, keys):
    print(f"Queris Reshape: {queries.reshape((-1, 1))}")
    print(f"keys.reshape: {keys.reshape((1, -1))}")
    return queries.reshape((-1, 1)) - keys.reshape((1, -1))


def attention_pool(query_key_diffs, values):
    attention_weights = F.softmax(-query_key_diffs ** 2 / 2, dim=1)
    return torch.matmul(attention_weights, values), attention_weights


class NWKernelRegression(d2l.Module):
    def __init__(self, keys, values, lr):
        super(NWKernelRegression, self).__init__()
        self.save_hyperparameters()
        self.w = torch.ones(1, requires_grad=True)

    def forward(self, queries):
        y_hat, self.attention_weights = attention_pool(
            diff(queries, self.keys) * self.w, self.values)
        return y_hat

    def loss(self, y_hat, y):
        l = (y_hat.reshape(-1) - y.reshape(-1)) ** 2 / 2
        return l.mean()

    def configure_optimizers(self):
        return d2l.SGD([self.w], self.lr)


if __name__ == "__main__":
    model = NWKernelRegression(data.x_train, data.y_train, lr=1)
    model.board.display = False
    trainer = d2l.Trainer(max_epochs=5)
    trainer.fit(model, data)
    plot_kernel_reg(model.forward(data.x_val))
