

import numpy as np


def comp_kraus(kraus_ops, dims):

    dims = list(dims)
    if len(dims) != 2:
        raise ValueError("dims must contain the input and output dims")

    input_dim = dims[0]
    output_dim = dims[1]

    for op in kraus_ops:
        if op.shape != (output_dim, input_dim):
            raise ValueError("Kraus operator shape does not match provided dims")

    complement = []
    for row in range(output_dim):
        pieces = []
        for op in kraus_ops:
            pieces.append(op[row, :])
        block = np.stack(pieces)
        complement.append(block)
    return complement
