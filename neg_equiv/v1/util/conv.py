
import numpy as np
from util.conv_utils import conv_output_length
from util.conv_utils import conv_input_length
from util.conv_utils import get_pad

def conv(x, filters, strides, padding):

    n, h, w, c = np.shape(x)
    fh, fw, fc, fo = np.shape(filters)
    sh, sw = strides
    
    ho = conv_output_length(h, fh, padding.lower(), strides[0])
    wo = conv_output_length(w, fw, padding.lower(), strides[1])
    output_shape = (n, ho, wo, fo)
    
    ph = get_pad(padding.lower(), fh)
    pw = get_pad(padding.lower(), fw)

    x = np.pad(x, [[0, 0], [ph, ph], [pw, pw], [0, 0]], mode='constant')
    xs = []
    for i in range(ho):
        for j in range(wo):
            slice_row = slice(i * sh, i * sh + fh)
            slice_col = slice(j * sw, j * sw + fw)
            xs.append(np.reshape(x[:, slice_row, slice_col, :], (n, 1, fh * fw * fc)))

    x_aggregate = np.concatenate(xs, axis=1)
    x_aggregate = np.reshape(x_aggregate, (n * ho * wo, fh * fw * fc))

    filters = np.reshape(filters, (fh * fw * fc, fo))        

    Z = np.dot(x_aggregate, filters)
    Z = np.reshape(Z, (n, ho, wo, fo))

    return Z
    
    
