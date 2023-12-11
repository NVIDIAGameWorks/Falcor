import torch

from largesteps.geometry import compute_matrix
from largesteps.optimize import AdamUniform
from largesteps.parameterize import from_differential, to_differential


class LargeSteps(torch.optim.Optimizer):
    """
    V: A list of parameters to optimize. It contains a tensor of shape (N, 3) representing the mesh vertices.
    F: An int32 tensor of shape (M, 3) representing the mesh indices.
    """
    def __init__(self, V: torch.Tensor, F: torch.Tensor, lr=0.1, betas=(0.9, 0.999), lambda_value=0.1):
        self.V = V[0]
        self.F = F
        self.M = compute_matrix(self.V, self.F, lambda_value)
        self.u = to_differential(self.M, self.V.detach()).clone().detach().requires_grad_(True)
        defaults = dict(F=self.F, lr=lr, betas=betas)
        self.optimizer = AdamUniform([self.u], lr=lr, betas=betas)
        super(LargeSteps, self).__init__(V, defaults)

    def step(self):
        # Build compute graph from u to V_next
        V_next = from_differential(self.M, self.u, 'Cholesky')

        # Propagate gradients from V_next to u
        V_next.backward(self.V.grad)

        # Step u
        self.optimizer.step()

        # Update V
        V_next = from_differential(self.M, self.u, 'Cholesky')
        self.V.data.copy_(V_next.data)

    def zero_grad(self):
        super(LargeSteps, self).zero_grad()
        self.optimizer.zero_grad()
