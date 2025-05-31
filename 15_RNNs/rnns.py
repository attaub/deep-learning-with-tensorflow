import numpy as np


class SimpleRNN:
    def __init__(self, input_size, hidden_size, output_size):
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.output_size = output_size

        # weight and bias initialization
        self.Wx = np.random.randn(hidden_size, input_size) * 0.01
        self.Wh = np.random.randn(hidden_size, hidden_size) * 0.01
        self.Wy = np.random.randn(output_size, hidden_size) * 0.01

        self.bh = np.zeros((hidden_size, 1))
        self.by = np.zeros((output_size, 1))

    ##############################
    def forward(self, inputs):
        h = np.zeros((self.hidden_size, 1))
        self.inputs = inputs
        self.hs = []
        self.ys = []

        for x in inputs:
            x = x.reshape(-1, 1)
            h = np.tanh(np.dot(self.Wx, x) + np.dot(self.Wh, h) + self.bh)
            y = np.dot(self.Wy, h) + self.by

            self.hs.append(h)
            self.ys.append(y)

        return self.ys, h

    ##############################
    def backward(self, dL_dy, learning_rate=0.01):
        dWx = np.zeros_like(self.Wx)
        dWh = np.zeros_like(self.Wh)
        dWy = np.zeros_like(self.Wy)
        dbh = np.zeros_like(self.bh)
        dby = np.zeros_like(self.by)
        dh_next = np.zeros((self.hidden_size, 1))

        for t in reversed(range(len(self.inputs))):
            dy = dL_dy[t]
            dWy += np.dot(dy, self.hs[t].T)
            dby += dy

            dh = np.dot(self.Wy.T, dy) + dh_next
            dh_raw = (1 - self.hs[t] ** 2) * dh  # derivative of tanh
            dbh += dh_raw
            dWx += np.dot(dh_raw, self.inputs[t].reshape(1, -1))
            dWh += (
                np.dot(dh_raw, self.hs[t - 1].T)
                if t > 0
                else np.dot(dh_raw, np.zeros((self.hidden_size, 1)).T)
            )

            dh_next = np.dot(self.Wh.T, dh_raw)

        # updating parameters
        self.Wx -= learning_rate * dWx
        self.Wh -= learning_rate * dWh
        self.Wy -= learning_rate * dWy
        self.bh -= learning_rate * dbh
        self.by -= learning_rate * dby

    ##############################
    def fit(self, X, Y, epochs=100, learning_rate=0.01):
        for epoch in range(epochs):
            total_loss = 0
            for inputs, targets in zip(X, Y):
                outputs, _ = self.forward(inputs)
                loss = sum(
                    0.5 * np.sum((o - t.reshape(-1, 1)) ** 2)
                    for o, t in zip(outputs, targets)
                )
                total_loss += loss

                dL_dy = [
                    o - t.reshape(-1, 1) for o, t in zip(outputs, targets)
                ]
                self.backward(dL_dy, learning_rate)

            print(f"Epoch {epoch+1}/{epochs}, Loss: {total_loss:.4f}")

    #####################
    def predict(self, X):
        predictions = []
        for inputs in X:
            outputs, _ = self.forward(inputs)
            predictions.append(outputs)
        return predictions


#################################################################

if __name__ == "__main__":
    input_size = 3
    time_steps = 5
    hidden_size = 4
    output_size = 2

    rnn = SimpleRNN(input_size, hidden_size, output_size)
    inputs = [np.random.randn(input_size) for _ in range(time_steps)]
    targets = [np.random.randn(output_size) for _ in range(time_steps)]

    dataset_X = [inputs] * 10
    dataset_Y = [targets] * 10

    rnn.fit(dataset_X, dataset_Y, epochs=10)
    predictions = rnn.predict([inputs])

    for i, out in enumerate(predictions[0]):
        print(f"Time step {i}: {out.ravel()}")
