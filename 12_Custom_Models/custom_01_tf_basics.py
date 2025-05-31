import sklearn
import tensorflow as tf
from tensorflow import keras

# Common imports
import numpy as np

np.random.seed(42)
tf.random.set_seed(42)

import matplotlib as mpl
import matplotlib.pyplot as plt

# ## Tensors and operations
# ### Tensors
tf.constant([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]])  # matrix
tf.constant(42)  # scalar
t = tf.constant([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]])
t

t.shape
t.dtype

# ### Indexing
t[:, 1:]
t[..., 1, tf.newaxis]

# ### Ops
t + 10
tf.square(t)
t @ tf.transpose(t)

# ### Using `keras.backend`

from tensorflow import keras

K = keras.backend
K.square(K.transpose(t)) + 10

# ### From/To NumPy
a = np.array([2.0, 4.0, 5.0])
tf.constant(a)
t.numpy()
np.array(t)
tf.square(a)
np.square(t)
# ### Conflicting Types

try:
    tf.constant(2.0) + tf.constant(40)
except tf.errors.InvalidArgumentError as ex:
    print(ex)

try:
    tf.constant(2.0) + tf.constant(40.0, dtype=tf.float64)
except tf.errors.InvalidArgumentError as ex:
    print(ex)

t2 = tf.constant(40.0, dtype=tf.float64)
tf.constant(2.0) + tf.cast(t2, tf.float32)

# ### Strings

tf.constant(b"hello world")
tf.constant("café")
u = tf.constant([ord(c) for c in "café"])
u
b = tf.strings.unicode_encode(u, "UTF-8")
tf.strings.length(b, unit="UTF8_CHAR")
tf.strings.unicode_decode(b, "UTF-8")

# ### String arrays
p = tf.constant(["Café", "Coffee", "caffè", "咖啡"])
tf.strings.length(p, unit="UTF8_CHAR")
r = tf.strings.unicode_decode(p, "UTF8")
r

print(r)

# ### Ragged tensors
print(r[1])
print(r[1:3])

r2 = tf.ragged.constant([[65, 66], [], [67]])
print(tf.concat([r, r2], axis=0))

r3 = tf.ragged.constant([[68, 69, 70], [71], [], [72, 73]])
print(tf.concat([r, r3], axis=1))

tf.strings.unicode_encode(r3, "UTF-8")
r.to_tensor()

# ### Sparse tensors
s = tf.SparseTensor(
    indices=[[0, 1], [1, 0], [2, 3]],
    values=[1.0, 2.0, 3.0],
    dense_shape=[3, 4],
)

print(s)

tf.sparse.to_dense(s)

s2 = s * 2.0

try:
    s3 = s + 1.0
except TypeError as ex:
    print(ex)

s4 = tf.constant([[10.0, 20.0], [30.0, 40.0], [50.0, 60.0], [70.0, 80.0]])
tf.sparse.sparse_dense_matmul(s, s4)


s5 = tf.SparseTensor(
    indices=[[0, 2], [0, 1]], values=[1.0, 2.0], dense_shape=[3, 4]
)
print(s5)

try:
    tf.sparse.to_dense(s5)
except tf.errors.InvalidArgumentError as ex:
    print(ex)


s6 = tf.sparse.reorder(s5)
tf.sparse.to_dense(s6)

# ### Sets

set1 = tf.constant([[2, 3, 5, 7], [7, 9, 0, 0]])
set2 = tf.constant([[4, 5, 6], [9, 10, 0]])
tf.sparse.to_dense(tf.sets.union(set1, set2))
tf.sparse.to_dense(tf.sets.difference(set1, set2))

tf.sparse.to_dense(tf.sets.intersection(set1, set2))


# ### Variables
v = tf.Variable([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]])
v.assign(2 * v)
v[0, 1].assign(42)
v[:, 2].assign([0.0, 1.0])

try:
    v[1] = [7.0, 8.0, 9.0]
except TypeError as ex:
    print(ex)

v.scatter_nd_update(indices=[[0, 0], [1, 2]], updates=[100.0, 200.0])

sparse_delta = tf.IndexedSlices(
    values=[[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]], indices=[1, 0]
)
v.scatter_update(sparse_delta)

# ### Tensor Arrays
array = tf.TensorArray(dtype=tf.float32, size=3)
array = array.write(0, tf.constant([1.0, 2.0]))
array = array.write(1, tf.constant([3.0, 10.0]))
array = array.write(2, tf.constant([5.0, 7.0]))
array.read(1)
array.stack()
mean, variance = tf.nn.moments(array.stack(), axes=0)
mean
variance
