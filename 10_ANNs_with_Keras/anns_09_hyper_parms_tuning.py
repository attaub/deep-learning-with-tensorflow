import tensorflow as tf
from tensorflow import keras
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.datasets import fetch_california_housing
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import RandomizedSearchCV
from scipy.stats import reciprocal

#################################################################

housing = fetch_california_housing()

X_train_full, X_test, y_train_full, y_test = train_test_split(
    housing.data, housing.target, random_state=42
)

X_train, X_valid, y_train, y_valid = train_test_split(
    X_train_full, y_train_full, random_state=42
)

scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_valid = scaler.transform(X_valid)
X_test = scaler.transform(X_test)

X_new = X_test[:3]

#################################################################
## Hyperparameter Tuning

keras.backend.clear_session()
np.random.seed(42)
tf.random.set_seed(42)


def build_model(n_hidden=1, n_neurons=30, learning_rate=3e-3, input_shape=[8]):
    model = keras.models.Sequential()
    model.add(keras.layers.InputLayer(input_shape=input_shape))
    for layer in range(n_hidden):
        model.add(keras.layers.Dense(n_neurons, activation="relu"))
    model.add(keras.layers.Dense(1))
    optimizer = keras.optimizers.SGD(learning_rate=learning_rate)
    model.compile(loss="mse", optimizer=optimizer)
    return model

#################################################################
#################################################################
keras_reg = keras.wrappers.scikit_learn.KerasRegressor(build_model)
#################################################################
#################################################################

keras_reg.fit(
    X_train,
    y_train,
    epochs=100,
    validation_data=(X_valid, y_valid),
#################################################################
    callbacks=[keras.callbacks.EarlyStopping(patience=10)],
#################################################################
)


mse_test = keras_reg.score(X_test, y_test)
y_pred = keras_reg.predict(X_new)


#################################################################
#################################################################
param_distribs = {
    "n_hidden": [0, 1, 2, 3],
    "n_neurons": np.arange(1, 100).tolist(),
    "learning_rate": reciprocal(3e-4, 3e-2).rvs(1000).tolist(),
}
#################################################################
#################################################################

#################################################################
#################################################################
rnd_search_cv = RandomizedSearchCV(
    keras_reg, param_distribs, n_iter=10, cv=3, verbose=2
)
rnd_search_cv.fit(
    X_train,
    y_train,
    epochs=100,
    validation_data=(X_valid, y_valid),
    callbacks=[keras.callbacks.EarlyStopping(patience=10)],
)
#################################################################
#################################################################


rnd_search_cv.best_params_
rnd_search_cv.best_score_
rnd_search_cv.best_estimator_
rnd_search_cv.score(X_test, y_test)
model = rnd_search_cv.best_estimator_.model
model

model.evaluate(X_test, y_test)
