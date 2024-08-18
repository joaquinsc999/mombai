import tensorflow as tf

def bspline_basis(x, knots, i, k):
    # Implementación del cálculo básico de B-spline
    if k == 0:
        return tf.where((knots[i] <= x) & (x < knots[i + 1]), 1.0, 0.0)
    else:
        w1 = (x - knots[i]) / (knots[i + k] - knots[i] + 1e-8)
        w2 = (knots[i + k + 1] - x) / (knots[i + k + 1] - knots[i + 1] + 1e-8)
        
        return w1 * bspline_basis(x, knots, i, k - 1) + w2 * bspline_basis(x, knots, i + 1, k - 1)


def KANBspline(x, knots, k, coefs, weights_fixed_activation, weigths_spline):
    n = knots.shape[0] - 1 - k
    b_splines = []

    for i in range(n):
        coef = coefs[:, i]
        spline = coef * bspline_basis(x, knots, i, k)
        whole_activation = tf.multiply(tf.nn.swish(x), weights_fixed_activation) + tf.multiply(spline, weigths_spline)
        b_splines.append(whole_activation)
    
    return tf.reduce_sum(b_splines, axis=0)
