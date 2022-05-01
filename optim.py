import torch
import optim_tests as tests


class _SGD:
    def __init__(
        self, params, lr: float, momentum: float, dampening: float, weight_decay: float
    ):
        self.params = list(params)
        self.lr = lr
        self.wd = weight_decay
        self.mu = momentum
        self.tau = dampening
        self.b = [None for _ in self.params]

    def zero_grad(self):
        for p in self.params:
            p.grad = None

    def step(self):
        with torch.no_grad():
            for i, p in enumerate(self.params):
                assert p.grad is not None
                g = p.grad + self.wd * p
                if self.mu:
                    if self.b[i] is not None:
                        self.b[i] = self.mu * self.b[i] + (1.0 - self.tau) * g
                    else:
                        self.b[i] = g
                    g = self.b[i]
                p -= self.lr * g


class _RMSprop:
    def __init__(
        self,
        params,
        lr: float,
        alpha: float,
        eps: float,
        weight_decay: float,
        momentum: float,
    ):
        self.params = list(params)
        self.lr = lr
        self.alpha = alpha
        self.eps = eps
        self.wd = weight_decay
        self.mu = momentum

        self.v = [torch.zeros_like(p) for p in self.params]
        self.b = [torch.zeros_like(p) for p in self.params]

    def zero_grad(self):
        for p in self.params:
            p.grad = None

    def step(self):
        with torch.no_grad():
            for i, p in enumerate(self.params):
                assert p.grad is not None
                g = p.grad + self.wd * p
                self.v[i] = self.alpha * self.v[i] + (1.0 - self.alpha) * g**2
                if self.mu > 0:
                    self.b[i] = self.mu * self.b[i] + g / (self.v[i].sqrt() + self.eps)
                    p -= self.lr * self.b[i]
                else:
                    p -= self.lr * g / (self.v[i].sqrt() + self.eps)


class _Adam:
    def __init__(
        self,
        params,
        lr: float,
        betas: tuple[float, float],
        eps: float,
        weight_decay: float,
    ):
        self.params = list(params)
        self.lr = lr
        self.beta1, self.beta2 = betas
        self.eps = eps
        self.wd = weight_decay

        self.m = [torch.zeros_like(p) for p in self.params]
        self.v = [torch.zeros_like(p) for p in self.params]
        self.t = 0

    def zero_grad(self):
        for p in self.params:
            p.grad = None

    def step(self):
        self.t += 1
        with torch.no_grad():
            for i, p in enumerate(self.params):
                assert p.grad is not None
                g = p.grad + self.wd * p
                self.m[i] = self.beta1 * self.m[i] + (1.0 - self.beta1) * g
                self.v[i] = self.beta2 * self.v[i] + (1.0 - self.beta2) * g**2
                m_hat = self.m[i] / (1.0 - self.beta1**self.t)
                v_hat = self.v[i] / (1.0 - self.beta2**self.t)
                p -= self.lr * m_hat / (v_hat.sqrt() + self.eps)


if __name__ == "__main__":
    tests.test_sgd(_SGD)
    tests.test_rmsprop(_RMSprop)
    tests.test_adam(_Adam)
