from sklearn.datasets import make_moons
import torch
from torch import nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset


class _MLP(nn.Module):
    def __init__(self, in_dim: int, hidden_dim: int, out_dim: int):
        super().__init__()
        self.layers = nn.Sequential(
            nn.Linear(in_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, out_dim),
        )

    def forward(self, x):
        return self.layers(x)


###############################################


def _get_moon_data(unsqueeze_y=False):
    X, y = make_moons(n_samples=512, noise=0.05, random_state=354)
    X = torch.tensor(X, dtype=torch.float32)
    y = torch.tensor(y, dtype=int)
    if unsqueeze_y:  # better when the training regimen uses l1 loss, rather than x-ent
        y = y.unsqueeze(-1)
    return DataLoader(TensorDataset(X, y), batch_size=128, shuffle=True)


def _check_equal(tensor1, tensor2):
    if torch.allclose(tensor1, tensor2, rtol=1e-3, atol=1e-5):
        print("Congrats! You've passed the test.")
    else:
        print("Your optimizer returns different results from the pytorch solution.")


###############################################


def _train_with_opt(model, opt):
    dl = _get_moon_data()
    for i, (X, y) in enumerate(dl):
        opt.zero_grad()
        loss = F.cross_entropy(model(X), y)
        loss.backward()
        opt.step()


def test_sgd(SGD):
    test_cases = [
        dict(lr=0.1, momentum=0.0, dampening=0.0, weight_decay=0.0),
        dict(lr=0.1, momentum=0.7, dampening=0.0, weight_decay=0.0),
        dict(lr=0.1, momentum=0.5, dampening=0.5, weight_decay=0.0),
        dict(lr=0.1, momentum=0.5, dampening=0.5, weight_decay=0.05),
        dict(lr=0.2, momentum=0.8, dampening=0.0, weight_decay=0.05),
    ]
    for opt_config in test_cases:
        torch.manual_seed(819)
        model = _MLP(2, 32, 2)
        opt = torch.optim.SGD(model.parameters(), **opt_config)
        _train_with_opt(model, opt)
        w0_correct = model.layers[0].weight

        torch.manual_seed(819)
        model = _MLP(2, 32, 2)
        opt = SGD(model.parameters(), **opt_config)
        _train_with_opt(model, opt)
        w0_submitted = model.layers[0].weight

        print("\nTesting configuration: ", opt_config)
        _check_equal(w0_correct, w0_submitted)


def test_rmsprop(RMSprop):
    test_cases = [
        dict(lr=0.1, alpha=0.9, eps=0.001, weight_decay=0.0, momentum=0.0),
        dict(lr=0.1, alpha=0.95, eps=0.0001, weight_decay=0.05, momentum=0.0),
        dict(lr=0.1, alpha=0.95, eps=0.0001, weight_decay=0.05, momentum=0.5),
        dict(lr=0.1, alpha=0.95, eps=0.0001, weight_decay=0.05, momentum=0.0),
    ]
    for opt_config in test_cases:
        torch.manual_seed(819)
        model = _MLP(2, 32, 2)
        opt = torch.optim.RMSprop(model.parameters(), **opt_config)
        _train_with_opt(model, opt)
        w0_correct = model.layers[0].weight

        torch.manual_seed(819)
        model = _MLP(2, 32, 2)
        opt = RMSprop(model.parameters(), **opt_config)
        _train_with_opt(model, opt)
        w0_submitted = model.layers[0].weight

        print("\nTesting configuration: ", opt_config)
        _check_equal(w0_correct, w0_submitted)


def test_adam(Adam):
    test_cases = [
        dict(lr=0.1, betas=(0.8, 0.95), eps=0.001, weight_decay=0.0),
        dict(lr=0.1, betas=(0.8, 0.9), eps=0.001, weight_decay=0.05),
        dict(lr=0.2, betas=(0.9, 0.95), eps=0.01, weight_decay=0.08),
    ]
    for opt_config in test_cases:
        torch.manual_seed(819)
        model = _MLP(2, 32, 2)
        opt = torch.optim.Adam(model.parameters(), **opt_config)
        _train_with_opt(model, opt)
        w0_correct = model.layers[0].weight

        torch.manual_seed(819)
        model = _MLP(2, 32, 2)
        opt = Adam(model.parameters(), **opt_config)
        _train_with_opt(model, opt)
        w0_submitted = model.layers[0].weight

        print("\nTesting configuration: ", opt_config)
        _check_equal(w0_correct, w0_submitted)
