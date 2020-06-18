import numpy as np
import six
import sys
PICKLE_KWARGS = {'encoding': 'latin1'} if six.PY3 else {}

rtol = 1e-4
atol = 1e-08
TOLERANCE = 1e-4
np.set_printoptions(threshold=50, precision=4)


def raw_mnist(path):
    return (cleaned_mnist(path))


def cleaned_mnist(path):
    data = np.genfromtxt(path, delimiter=',')
    X = data[:, :-1]
    Y = data[:, -1]
    Y = Y.astype(int)
    return X, Y


def isAllClose(a, b, tol=0.01):
    LIST_TYPE = type([])
    if(type(a) == LIST_TYPE or type(b) == LIST_TYPE):
        assert len(a) == len(b)
        for i, j in zip(a, b):
            if(not np.allclose(i, j, atol=tol)):
                return False
        return True
    return np.allclose(a, b, atol=tol)


def closeness_test(value, reference, name):
    if not isinstance(value, np.ndarray):
        errmsg = "%s is not an array" % name
        raise TypeError(errmsg)
    if not value.dtype == reference.dtype:
        errmsg = "%s is of type %s when it should be %s" % (name, value.dtype, reference.dtype)
        raise TypeError(errmsg)
    if not value.shape == reference.shape:
        errmsg = "%s is of shape %s when it should be %s" % (name, value.shape, reference.shape)
        raise ValueError(errmsg)
    if not np.allclose(value, reference, rtol=rtol, atol=atol):
        errmsg = "Wrong value for %s" % name
        errmsg = errmsg + "\nSubmission value : \n"
        errmsg = errmsg + np.array2string(value)
        errmsg = errmsg + "\nReference value :\n"
        errmsg = errmsg + np.array2string(reference)
        raise ValueError(errmsg)
