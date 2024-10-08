from mindspore import Tensor
import numpy as np
base = Tensor([[0, 1], [2, 3]])
base.is_contiguous()
# True
t = base.transpose(1, 0) # t is a view of base. No data movement happened here.
print(t.is_contiguous())
# False
# To get a contiguous tensor, call `.contiguous()` to enforce
# copying data when `t` is not contiguous.
c = t.contiguous()
print(c.is_contiguous())
# True