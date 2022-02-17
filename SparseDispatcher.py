import tensorflow as tf
import functools
from tensorflow.python.framework import function

def add_scope(scope=None, scope_fn=None):
  """Return a decorator which add a TF name/variable scope to a function.
  Note that the function returned by the decorator accept an additional 'name'
  parameter, which can overwrite the name scope given when the function is
  created.
  Args:
    scope (str): name of the scope. If None, the function name is used.
    scope_fn (fct): Either tf.name_scope or tf.variable_scope
  Returns:
    fct: the add_scope decorator
  """
  def decorator(f):

    @functools.wraps(f)
    def decorated(*args, **kwargs):
      name = kwargs.pop("name", None)  # Python 2 hack for keyword only args
      with scope_fn(name or scope or f.__name__):
        return f(*args, **kwargs)

    return decorated

  return decorator


def add_name_scope(scope=None):
  return add_scope(scope, scope_fn=tf.name_scope)

@function.Defun(
    python_grad_func=lambda x, dy: tf.convert_to_tensor(dy),
    shape_func=lambda op: [op.inputs[0].get_shape()])
def convert_gradient_to_tensor(x):
  """Identity operation whose gradient is converted to a `Tensor`.
  Currently, the gradient to `tf.concat` is particularly expensive to
  compute if dy is an `IndexedSlices` (a lack of GPU implementation
  forces the gradient operation onto CPU).  This situation occurs when
  the output of the `tf.concat` is eventually passed to `tf.gather`.
  It is sometimes faster to convert the gradient to a `Tensor`, so as
  to get the cheaper gradient for `tf.concat`.  To do this, replace
  `tf.concat(x)` with `convert_gradient_to_tensor(tf.concat(x))`.
  Args:
    x: A `Tensor`.
  Returns:
    The input `Tensor`.
  """
  return x


class SparseDispatcher(object):
    """Helper for implementing a mixture of experts.
  The purpose of this class is to create input minibatches for the
  experts and to combine the results of the experts to form a unified
  output tensor.
  There are two functions:
    dispatch - take an input Tensor and create input Tensors for each expert.
    combine - take output Tensors from each expert and form a combined output
      Tensor.  Outputs from different experts for the same batch element are
      summed together, weighted by the provided "gates".
  The class is initialized with a "gates" Tensor, which specifies which
  batch elements go to which experts, and the weights to use when combining
  the outputs.  Batch element b is sent to expert e iff gates[b, e] != 0.
  The inputs and outputs are all two-dimensional [batch, depth].
  Caller is responsible for collapsing additional dimensions prior to
  calling this class and reshaping the output to the original shape.
  See common_layers.reshape_like().
  Example use:
  gates: a float32 `Tensor` with shape `[batch_size, num_experts]`
  inputs: a float32 `Tensor` with shape `[batch_size, input_size]`
  experts: a list of length `num_experts` containing sub-networks.
    dispatcher = SparseDispatcher(num_experts, gates)
    expert_inputs = dispatcher.dispatch(inputs)
    expert_outputs = [experts[i](expert_inputs[i]) for i in range(num_experts)]
    outputs = dispatcher.combine(expert_outputs)
  The preceding code sets the output for a particular example b to:
  output[b] = Sum_i(gates[b, i] * experts[i](inputs[b]))
  This class takes advantage of sparsity in the gate matrix by including in the
  `Tensor`s for expert i only the batch elements for which `gates[b, i] > 0`.
  """

    def __init__(self, num_experts, gates):
        """Create a SparseDispatcher.
    Args:
      num_experts: an integer.
      gates: a `Tensor` of shape `[batch_size, num_experts]`.
    Returns:
      a SparseDispatcher
    """
        self._gates = gates
        self._num_experts = num_experts

        where = tf.cast(tf.where(tf.transpose(gates) > 0), dtype=tf.int32)
        self._expert_index, self._batch_index = tf.unstack(where, num=2, axis=1)
        self._part_sizes_tensor = tf.reduce_sum(tf.cast(gates > 0, dtype=tf.int32), [0])
        self._nonzero_gates = tf.gather(
            tf.reshape(self._gates, [-1]),
            self._batch_index * num_experts + self._expert_index)

    @add_name_scope()
    def dispatch(self, inp):
        """Create one input Tensor for each expert.
    The `Tensor` for a expert `i` contains the slices of `inp` corresponding
    to the batch elements `b` where `gates[b, i] > 0`.
    Args:
      inp: a `Tensor` of shape "[batch_size, <extra_input_dims>]`
    Returns:
      a list of `num_experts` `Tensor`s with shapes
        `[expert_batch_size_i, <extra_input_dims>]`.
    """
        inp = tf.gather(inp, self._batch_index)
        return tf.split(inp, self._part_sizes_tensor, 0, num=self._num_experts)

    @add_name_scope()
    def combine(self, expert_out, multiply_by_gates=True):
        """Sum together the expert output, weighted by the gates.
    The slice corresponding to a particular batch element `b` is computed
    as the sum over all experts `i` of the expert output, weighted by the
    corresponding gate values.  If `multiply_by_gates` is set to False, the
    gate values are ignored.
    Args:
      expert_out: a list of `num_experts` `Tensor`s, each with shape
        `[expert_batch_size_i, <extra_output_dims>]`.
      multiply_by_gates: a boolean
    Returns:
      a `Tensor` with shape `[batch_size, <extra_output_dims>]`.
    """
        # see comments on convert_gradient_to_tensor
        stitched = tf.concat(expert_out, 0)
        if multiply_by_gates:
            stitched *= tf.expand_dims(self._nonzero_gates, 1)
        combined = tf.math.unsorted_segment_sum(stitched, self._batch_index,
                                           tf.shape(self._gates)[0])
        return combined

    def expert_to_gates(self):
        """Gate values corresponding to the examples in the per-expert `Tensor`s.
    Returns:
      a list of `num_experts` one-dimensional `Tensor`s with type `tf.float32`
          and shapes `[expert_batch_size_i]`
    """
        return tf.split(
            self._nonzero_gates, self._part_sizes_tensor, 0, num=self._num_experts)

    def expert_to_batch_indices(self):
        """Batch indices corresponding to the examples in the per-expert `Tensor`s.
    Returns:
      a list of `num_experts` one-dimensional `Tensor`s with type `tf.int64`
          and shapes `[expert_batch_size_i]`
    """
        return tf.split(
            self._batch_index, self._part_sizes_tensor, 0, num=self._num_experts)

    @property
    def part_sizes(self):
        return self._part_sizes_tensor
