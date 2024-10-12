import numpy as np
import mindspore as ms
import mindspore.ops as ops
from mindspore.common.initializer import initializer
from mindspore.nn import Optimizer
_adam_opt = ops.MultitypeFuncGraph("adam_opt")
host_assign = ops.Assign()
host_assign.set_device("CPU")
host_cast = ops.Cast()
host_cast.set_device("CPU")
device_cast = ops.Cast()

@_adam_opt.register("Function", "Tensor", "Tensor", "Tensor", "Tensor", "Number", "Tensor", "Tensor", "Tensor",
                    "Tensor", "Bool", "Bool")
def _update_run_kernel(opt, beta1, beta2, eps, lr, weight_decay, param, m, v, gradient, decay_flags, optim_filter):
    """
    Update parameters by AdamWeightDecay op.
    """
    success = True
    if optim_filter:
        param32 = host_cast(param, ms.float32)
        gradient = device_cast(gradient, ms.float32)
        if decay_flags:
            next_param = opt(param32, m, v, lr, beta1, beta2, eps, weight_decay, gradient)
        else:
            next_param = opt(param32, m, v, lr, beta1, beta2, eps, 0.0, gradient)
        ret = host_assign(param, host_cast(ops.depend(param32, next_param), ops.dtype(param)))
        return ops.depend(success, ret)
    return success

class AdamWeightDecayOp(Optimizer):
    def __init__(self, params, learning_rate=1e-3, beta1=0.9, beta2=0.999, eps=1e-6, weight_decay=0.0):
        super(AdamWeightDecayOp, self).__init__(learning_rate, params, weight_decay)
        self.beta1 = ms.Tensor(np.array([beta1]).astype(np.float32))
        self.beta2 = ms.Tensor(np.array([beta2]).astype(np.float32))
        self.eps = ms.Tensor(np.array([eps]).astype(np.float32))
        self.moments1 = self.clone_param32(prefix="adam_m", init='zeros')
        self.moments2 = self.clone_param32(prefix="adam_v", init='zeros')
        self.opt = ops.AdamWeightDecay()
        self.hyper_map = ops.HyperMap()
        self.opt.set_device("CPU")

    def construct(self, gradients):
        """AdamWeightDecayOp"""
        lr = self.get_lr()
        if self.is_group:
            if self.is_group_lr:
                optim_result = self.map_reverse(ops.partial(_adam_opt, self.opt, self.beta1, self.beta2, self.eps),
                                                lr, self.weight_decay, self.parameters, self.moments1, self.moments2,
                                                gradients, self.decay_flags, self.optim_filter)
            else:
                optim_result = self.map_reverse(ops.partial(_adam_opt, self.opt, self.beta1, self.beta2, self.eps, lr),
                                                self.weight_decay, self.parameters, self.moments1, self.moments2,
                                                gradients, self.decay_flags, self.optim_filter)
        else:
            optim_result = self.map_reverse(ops.partial(_adam_opt, self.opt, self.beta1, self.beta2, self.eps, lr,
                                                        self.weight_decay), self.parameters, self.moments1, self.moments2,
                                            gradients, self.decay_flags, self.optim_filter)
        return optim_result

    def clone_param32(self, prefix, init=None):
        new = []
        for old_param in self.parameters:
            param_init = init
            if init is None:
                param_init = old_param.init
            new_state = old_param.clone()
            new_state.set_dtype(ms.float32)
            new_state.set_data(initializer(param_init, shape=old_param.shape, dtype=ms.float32))
            new_state.name = prefix + '.' + new_state.name
            new.append(new_state)
        return ms.ParameterTuple(new)
