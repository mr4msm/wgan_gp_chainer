import numpy

from chainer import cuda
from chainer import function
from chainer import function_node
from chainer.functions.math import matmul as _matmul
from chainer import utils
from chainer.utils import type_check
from chainer import variable


def _convert_value_to_string(value):
    if isinstance(value, variable.Variable):
        value = value.data

    if numpy.isscalar(value):
        if value < 0:
            return '({})'.format(value)
        else:
            return str(value)
    elif isinstance(value, (numpy.ndarray, cuda.ndarray)):
        return 'constant array'
    else:
        raise ValueError(
            'value must be scalar, ndarray, or Variable')


def _check_constant_type(value):
    if numpy.isscalar(value):
        return
    elif isinstance(value, (numpy.ndarray, cuda.ndarray)):
        return
    else:
        raise ValueError(
            'value must be scalar, ndarray, or Variable')


def _preprocess_const(x, value):
    xp = cuda.get_array_module(x)
    if not numpy.isscalar(value) and cuda.get_array_module(value) != xp:
        # TODO(unno): We can transfer arrays automatically
        raise TypeError('Cannot mix cupy.ndarray and numpy.ndarray')

    b = xp.broadcast(x, value)
    if b.shape != x.shape:
        raise ValueError('Failed to broadcast arrays')
    return utils.force_type(x.dtype, value)


class Neg(function_node.FunctionNode):

    @property
    def label(self):
        return '__neg__'

    def check_type_forward(self, in_types):
        type_check.expect(in_types.size() == 1)

    def forward(self, x):
        self.retain_inputs(())
        return utils.force_array(-x[0]),

    def backward(self, indexes, gy):
        return -gy[0],


def neg(self):  # -x
    """Element-wise negation.

    Returns:
        ~chainer.Variable: Output variable.
    """
    return Neg().apply((self,))[0]


_kern = None


def _get_kern():
    global _kern
    if _kern is None:
        _kern = cuda.elementwise(
            'T cond, T x', 'T y',
            'y = cond >= 0 ? x : (T)(-1. * x)', 'abs')
    return _kern


class Absolute(function_node.FunctionNode):

    @property
    def label(self):
        return '|_|'

    def check_type_forward(self, in_types):
        type_check.expect(in_types.size() == 1)
        type_check.expect(in_types[0].dtype.kind == 'f')

    def forward(self, inputs):
        self.retain_inputs((0,))
        x, = inputs
        return utils.force_array(abs(x)),

    def backward(self, indexes, grad_outputs):
        x = self.get_retained_inputs()[0].data
        return _AbsGrad(x).apply(grad_outputs)


class _AbsGrad(function_node.FunctionNode):

    def __init__(self, x):
        self.x = x

    def forward(self, inputs):
        gy, = inputs
        gy = gy.copy()
        gy[self.x < 0] *= -1.
        return gy,

    # def forward_gpu(self, inputs):
    #     gy, = inputs
    #     gy = _get_kern()(self.x, gy)
    #     return gy,

    def backward(self, indexes, grad_outputs):
        return _AbsGrad(self.x).apply(grad_outputs)


def absolute(self):
    """Element-wise absolute.

    Returns:
        ~chainer.Variable: Output variable.
    """
    return Absolute().apply((self,))[0]


class Add(function_node.FunctionNode):

    @property
    def label(self):
        return '_ + _'

    def check_type_forward(self, in_types):
        type_check.expect(in_types.size() == 2)
        type_check.expect(
            in_types[0].dtype == in_types[1].dtype,
            in_types[0].shape == in_types[1].shape
        )

    def forward(self, x):
        y = utils.force_array(x[0] + x[1])
        return y,

    def backward(self, indexes, gy):
        return gy[0], gy[0]


class AddConstant(function_node.FunctionNode):

    def __init__(self, value):
        self.value = value

    @property
    def label(self):
        return '_ + %s' % _convert_value_to_string(self.value)

    def check_type_forward(self, in_types):
        type_check.expect(in_types.size() == 1)

    def forward(self, x):
        value = _preprocess_const(x[0], self.value)
        return utils.force_array(x[0] + value),

    def backward(self, indexes, gy):
        return gy[0],


def add(self, rhs):  # lhs + rhs
    """Element-wise addition.

    Returns:
        ~chainer.Variable: Output variable.
    """
    if isinstance(rhs, variable.Variable):
        return Add().apply((self, rhs))[0]
    _check_constant_type(rhs)
    return AddConstant(rhs).apply((self,))[0]


class Sub(function_node.FunctionNode):

    @property
    def label(self):
        return '_ - _'

    def check_type_forward(self, in_types):
        type_check.expect(in_types.size() == 2)
        type_check.expect(
            in_types[0].dtype == in_types[1].dtype,
            in_types[0].shape == in_types[1].shape
        )

    def forward(self, x):
        return utils.force_array(x[0] - x[1]),

    def backward(self, indexes, gy):
        return gy[0], -gy[0]


def sub(self, rhs):  # lhs - rhs
    """Element-wise subtraction.

    Returns:
        ~chainer.Variable: Output variable.
    """

    if isinstance(rhs, variable.Variable):
        return Sub().apply((self, rhs))[0]
    _check_constant_type(rhs)
    return AddConstant(-rhs).apply((self,))[0]


class SubFromConstant(function_node.FunctionNode):

    def __init__(self, value):
        self.value = value

    @property
    def label(self):
        return '%s - _' % _convert_value_to_string(self.value)

    def check_type_forward(self, in_types):
        type_check.expect(in_types.size() == 1)

    def forward(self, x):
        self.retain_inputs(())
        value = _preprocess_const(x[0], self.value)
        return utils.force_array(value - x[0]),

    def backward(self, indexes, gy):
        return -gy[0],


def rsub(self, rhs):  # rhs - lhs
    """Element-wise subtraction.

    Returns:
        ~chainer.Variable: Output variable.
    """
    if isinstance(rhs, variable.Variable):
        return Sub().apply((rhs, self))[0]
    _check_constant_type(rhs)
    return SubFromConstant(rhs).apply((self,))[0]


class Mul(function_node.FunctionNode):

    @property
    def label(self):
        return '_ * _'

    def check_type_forward(self, in_types):
        type_check.expect(in_types.size() == 2)
        type_check.expect(
            in_types[0].dtype.kind == 'f',
            in_types[0].dtype == in_types[1].dtype,
            in_types[0].shape == in_types[1].shape
        )

    def forward(self, x):
        self.retain_inputs((0, 1))
        return utils.force_array(x[0] * x[1]),

    def backward(self, indexes, gy):
        xs = self.get_retained_inputs()
        return tuple(gy[0] * xs[1 - i] for i in indexes)


class MulConstant(function_node.FunctionNode):

    def __init__(self, value):
        self.value = value

    @property
    def label(self):
        return '_ * %s' % _convert_value_to_string(self.value)

    def check_type_forward(self, in_types):
        type_check.expect(in_types.size() == 1)

    def forward(self, x):
        value = _preprocess_const(x[0], self.value)
        return utils.force_array(value * x[0]),

    def backward(self, indexes, gy):
        return self.value * gy[0],


def mul(self, rhs):  # lhs * rhs
    """Element-wise multiplication.

    Returns:
        ~chainer.Variable: Output variable.
    """

    if isinstance(rhs, variable.Variable):
        return Mul().apply((self, rhs))[0]
    _check_constant_type(rhs)
    return MulConstant(rhs).apply((self,))[0]


class Div(function.Function):

    @property
    def label(self):
        return '_ / _'

    def check_type_forward(self, in_types):
        type_check.expect(in_types.size() == 2)
        type_check.expect(
            in_types[0].dtype.kind == 'f',
            in_types[0].dtype == in_types[1].dtype,
            in_types[0].shape == in_types[1].shape
        )

    def forward(self, x):
        return utils.force_array(x[0] / x[1]),

    def backward_cpu(self, x, gy):
        gx0 = utils.force_array(gy[0] / x[1])
        return gx0, utils.force_array(-gx0 * x[0] / x[1])

    def backward_gpu(self, x, gy):
        return cuda.elementwise(
            'T x0, T x1, T gy',
            'T gx0, T gx1',
            '''
               gx0 = gy / x1;
               gx1 = -gx0 * x0 / x1;
            ''', 'div_bwd')(x[0], x[1], gy[0])


def div(self, rhs):  # lhs / rhs
    """Element-wise division

    Returns:
        ~chainer.Variable: Output variable.
    """

    if isinstance(rhs, variable.Variable):
        return Div()(self, rhs)
    _check_constant_type(rhs)
    return MulConstant(1. / rhs).apply((self,))[0]


class DivFromConstant(function.Function):

    def __init__(self, value):
        self.value = value

    @property
    def label(self):
        return '_ / %s' % _convert_value_to_string(self.value)

    def check_type_forward(self, in_types):
        type_check.expect(in_types.size() == 1)
        type_check.expect(in_types[0].dtype.kind == 'f')

    def forward(self, x):
        value = _preprocess_const(x[0], self.value)
        return utils.force_array(value / x[0]),

    def backward_cpu(self, x, gy):
        value = _preprocess_const(x[0], self.value)
        return utils.force_array(-value * gy[0] / (x[0] ** 2)),

    def backward_gpu(self, x, gy):
        # TODO(beam2d): Make it not use the input
        value = _preprocess_const(x[0], self.value)
        gx = cuda.elementwise('T x, T gy, T value', 'T gx',
                              'gx = -value * gy / (x * x)',
                              'div_from_const_bwd')(x[0], gy[0], value)
        return gx,


def rdiv(self, rhs):  # rhs / lhs
    """Element-wise division.

    Returns:
        ~chainer.Variable: Output variable.
    """

    if isinstance(rhs, variable.Variable):
        return Div()(rhs, self)
    _check_constant_type(rhs)
    return DivFromConstant(rhs)(self)


class PowVarVar(function.Function):

    @property
    def label(self):
        return '_ ** _'

    def check_type_forward(self, in_types):
        type_check.expect(in_types.size() == 2)
        type_check.expect(
            in_types[0].dtype.kind == 'f',
            in_types[0].dtype == in_types[1].dtype,
            in_types[0].shape == in_types[1].shape
        )

    def forward_cpu(self, x):
        self.y = utils.force_array(x[0] ** x[1])
        return self.y,

    def forward_gpu(self, x):
        return x[0] ** x[1],

    def backward_cpu(self, x, gy):
        one = x[1].dtype.type(1)
        gx0 = utils.force_array(x[1] * (x[0] ** (x[1] - one)) * gy[0])
        gx1 = utils.force_array(numpy.log(x[0]) * self.y * gy[0])
        return gx0, gx1

    def backward_gpu(self, x, gy):
        return cuda.elementwise(
            'T x0, T x1, T gy', 'T gx0, T gx1',
            '''
               gx0 = x1 * pow(x0, x1 - 1) * gy;
               gx1 = log(x0) * pow(x0, x1) * gy;
            ''', 'pow_var_var_bwd')(x[0], x[1], gy[0])


class PowVarConst(function.Function):

    def __init__(self, value):
        self.value = value

    @property
    def label(self):
        return '_ ** %s' % _convert_value_to_string(self.value)

    def check_type_forward(self, in_types):
        type_check.expect(in_types.size() == 1)
        type_check.expect(in_types[0].dtype.kind == 'f')

    def forward(self, x):
        value = _preprocess_const(x[0], self.value)
        return utils.force_array(x[0] ** value, x[0].dtype),

    def backward_cpu(self, x, gy):
        val_1 = _preprocess_const(x[0], self.value - 1)
        gx = utils.force_type(x[0].dtype, self.value) * (x[0] ** val_1) * gy[0]
        return utils.force_array(gx),

    def backward_gpu(self, x, gy):
        value = _preprocess_const(x[0], self.value)
        gx = cuda.elementwise(
            'T x, T gy, T value', 'T gx',
            'gx = value * pow(x, value - 1) * gy',
            'pow_var_const_bwd')(x[0], gy[0], value)
        return gx,


def pow(self, rhs):  # lhs ** rhs
    """Element-wise power function.

    Returns:
        ~chainer.Variable: Output variable.
    """

    if isinstance(rhs, variable.Variable):
        return PowVarVar()(self, rhs)
    _check_constant_type(rhs)
    return PowVarConst(rhs)(self)


class PowConstVar(function.Function):

    def __init__(self, value):
        self.value = value

    @property
    def label(self):
        return '%s ** _' % _convert_value_to_string(self.value)

    def check_type_forward(self, in_types):
        type_check.expect(in_types.size() == 1)
        type_check.expect(in_types[0].dtype.kind == 'f')

    def forward(self, x):
        value = _preprocess_const(x[0], self.value)
        self.y = utils.force_array(value ** x[0])
        return self.y,

    def backward_cpu(self, x, gy):
        # TODO(beam2d): Make it not use the input
        value = _preprocess_const(x[0], self.value)
        return utils.force_array(
            numpy.log(value, dtype=x[0].dtype) * self.y * gy[0]),

    def backward_gpu(self, x, gy):
        value = _preprocess_const(x[0], self.value)
        gx = cuda.elementwise(
            'T x, T gy, T value', 'T gx',
            'gx = log(value) * pow(value, x) * gy',
            'pow_const_var_bwd')(x[0], gy[0], value)
        return gx,


def rpow(self, rhs):  # rhs ** lhs
    """Element-wise power function.

    Returns:
        ~chainer.Variable: Output variable.
    """

    if isinstance(rhs, variable.Variable):
        return PowVarVar()(rhs, self)
    _check_constant_type(rhs)
    return PowConstVar(rhs)(self)


class MatMulVarVar(_matmul.MatMul):

    @property
    def label(self):
        return '_ @ _'


class MatMulVarConst(function.Function):

    def __init__(self, value):
        self.value = value

    @property
    def label(self):
        return '_ @ %s' % _convert_value_to_string(self.value)

    def check_type_forward(self, in_types):
        type_check.expect(in_types.size() == 1)
        a_type = in_types[0]
        b_type = self.value

        type_check.expect(
            a_type.dtype.kind == 'f',
            b_type.dtype.kind == 'f',
            a_type.ndim >= 1,
            a_type.ndim == b_type.ndim,
        )

        ndim = type_check.eval(a_type.ndim)
        if ndim == 1:
            type_check.expect(a_type.shape == b_type.shape)
        else:
            a_idx = _matmul._get_check_index(False, False,
                                             row_idx=-2, col_idx=-1)
            b_idx = _matmul._get_check_index(False, True,
                                             row_idx=-2, col_idx=-1)
            type_check.expect(
                a_type.shape[:-2] == b_type.shape[:-2],
                a_type.shape[a_idx] == b_type.shape[b_idx],
            )

    def forward(self, x):
        self.retain_inputs(())
        self._x_shape = x[0].shape
        return utils.force_array(_matmul._matmul(x[0], self.value)),

    def backward(self, x, gy):
        if gy[0].ndim == 0:
            gx0 = gy[0] * self.value
        else:
            gx0 = _matmul._matmul(
                gy[0], self.value, transb=True, transout=False
            ).reshape(self._x_shape)
        return gx0,


class MatMulConstVar(function.Function):

    def __init__(self, value):
        self.value = value

    @property
    def label(self):
        return '%s @ _' % _convert_value_to_string(self.value)

    def check_type_forward(self, in_types):
        type_check.expect(in_types.size() == 1)
        a_type = self.value
        b_type = in_types[0]

        type_check.expect(
            a_type.dtype.kind == 'f',
            b_type.dtype.kind == 'f',
            a_type.ndim >= 1,
            a_type.ndim == b_type.ndim,
        )

        ndim = type_check.eval(a_type.ndim)
        if ndim == 1:
            type_check.expect(a_type.shape == b_type.shape)
        else:
            a_idx = _matmul._get_check_index(False, False,
                                             row_idx=-2, col_idx=-1)
            b_idx = _matmul._get_check_index(False, True,
                                             row_idx=-2, col_idx=-1)
            type_check.expect(
                a_type.shape[:-2] == b_type.shape[:-2],
                a_type.shape[a_idx] == b_type.shape[b_idx],
            )

    def forward(self, x):
        self.retain_inputs(())
        self._x_shape = x[0].shape
        return utils.force_array(_matmul._matmul(self.value, x[0])),

    def backward(self, x, gy):
        if gy[0].ndim == 0:
            gx1 = gy[0] * self.value
        else:
            gx1 = _matmul._matmul(
                self.value, gy[0], transa=True, transout=False
            ).reshape(self._x_shape)
        return gx1,


def matmul(self, rhs):  # lhs @ rhs
    """Matrix multiplication.

    Returns:
        ~chainer.Variable: Output variable.
    """

    if isinstance(rhs, variable.Variable):
        return MatMulVarVar()(self, rhs)
    _check_constant_type(rhs)
    return MatMulVarConst(rhs)(self)


def rmatmul(self, rhs):  # rhs @ lhs
    """Matrix multiplication.

    Returns:
        ~chainer.Variable: Output variable.
    """

    if isinstance(rhs, variable.Variable):
        return MatMulVarVar()(rhs, self)
    _check_constant_type(rhs)
    return MatMulConstVar(rhs)(self)


def install_variable_arithmetics():
    variable.Variable.__neg__ = neg
    variable.Variable.__abs__ = absolute
    variable.Variable.__add__ = add
    variable.Variable.__radd__ = add
    variable.Variable.__sub__ = sub
    variable.Variable.__rsub__ = rsub
    variable.Variable.__mul__ = mul
    variable.Variable.__rmul__ = mul
    variable.Variable.__div__ = div
    variable.Variable.__truediv__ = div
    variable.Variable.__rdiv__ = rdiv
    variable.Variable.__rtruediv__ = rdiv
    variable.Variable.__pow__ = pow
    variable.Variable.__rpow__ = rpow
    variable.Variable.__matmul__ = matmul
    variable.Variable.__rmatmul__ = rmatmul
