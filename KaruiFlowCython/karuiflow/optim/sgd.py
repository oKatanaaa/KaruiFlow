from .optimizer import Optimizer, required
from karuiflow.core import Parameter, tensor


class SGD(Optimizer):
    def __init__(self, params, lr=required, momentum=0.):
        if lr is not required and lr < 0.0:
            raise ValueError("Invalid learning rate: {}".format(lr))
        if momentum < 0.0:
            raise ValueError("Invalid momentum value: {}".format(momentum))

        defaults = dict(lr=lr, momentum=momentum)
        super(SGD, self).__init__(params, defaults)
        self.__init_momentum_buffers()

    def __init_momentum_buffers(self):
        for group in self.param_groups:
            momentum_buffers = []
            for p in group['params']:
                if p.requires_grad:
                    momentum_buffers.append(Parameter.from_tensor(p))
                else:
                    momentum_buffers.append(None)
            group['momentum_buffers'] = momentum_buffers

    def step(self):
        """Performs a single optimization step.
        """
        loss = None
        for group in self.param_groups:
            params = group['params']
            momentum = group['momentum']
            lr = group['lr']
            momentum_buffers = group['momentum_buffers']

            for p, b in zip(params, momentum_buffers):
                if p.grad is None:
                    continue

                # Update buffer
                b_update = b * tensor(momentum, dtype='float32') + tensor(1 - momentum, dtype='float32') * p.grad
                b.assign(b_update)

                # Perform gradient step
                p += b_update * tensor(lr, dtype='float32')

        return loss
