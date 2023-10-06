import torch
import functools

assert_equal = functools.partial(torch.testing.assert_close, rtol=0, atol=0)
