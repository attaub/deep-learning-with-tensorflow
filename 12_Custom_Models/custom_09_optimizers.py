import sys
import sklearn
import tensorflow as tf
from tensorflow import keras

# Common imports
import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt


# ## Custom Optimizers
# Defining custom optimizers is not very common, but in case you are one of the happy few who gets to write one, here is an example:


class MyMomentumOptimizer(keras.optimizers.Optimizer):
    def __init__(
        self,
        learning_rate=0.001,
        momentum=0.9,
        name="MyMomentumOptimizer",
        **kwargs
    ):
        """Call super().__init__() and use _set_hyper() to store hyperparameters"""
        super().__init__(name, **kwargs)
        self._set_hyper(
            "learning_rate", kwargs.get("lr", learning_rate)
        )  # handle lr=learning_rate
        self._set_hyper("decay", self._initial_decay)  #
        self._set_hyper("momentum", momentum)

    def _create_slots(self, var_list):
        """For each model variable, create the optimizer variable associated with it.
        TensorFlow calls these optimizer variables "slots".
        For momentum optimization, we need one momentum slot per model variable.
        """
        for var in var_list:
            self.add_slot(var, "momentum")

    @tf.function
    def _resource_apply_dense(self, grad, var):
        """Update the slots and perform one optimization step for one model variable"""
        var_dtype = var.dtype.base_dtype
        lr_t = self._decayed_lr(var_dtype)  # handle learning rate decay
        momentum_var = self.get_slot(var, "momentum")
        momentum_hyper = self._get_hyper("momentum", var_dtype)
        momentum_var.assign(
            momentum_var * momentum_hyper - (1.0 - momentum_hyper) * grad
        )
        var.assign_add(momentum_var * lr_t)

    def _resource_apply_sparse(self, grad, var):
        raise NotImplementedError

    def get_config(self):
        base_config = super().get_config()
        return {
            **base_config,
            "learning_rate": self._serialize_hyperparameter("learning_rate"),
            "decay": self._serialize_hyperparameter("decay"),
            "momentum": self._serialize_hyperparameter("momentum"),
        }


keras.backend.clear_session()
np.random.seed(42)
tf.random.set_seed(42)


model = keras.models.Sequential([keras.layers.Dense(1, input_shape=[8])])
model.compile(loss="mse", optimizer=MyMomentumOptimizer())
model.fit(X_train_scaled, y_train, epochs=5)
