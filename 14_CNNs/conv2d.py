import numpy as np

class Conv2D:
    def __init__(self, num_filters, filter_size):
        self.num_filters = num_filters
        self.filter_size = filter_size
        self.filters = np.random.randn(num_filters, filter_size, filter_size) / 9  # Normalized init
    
    def iterate_regions(self, image):
        h, w = image.shape
        for i in range(h - self.filter_size + 1):
            for j in range(w - self.filter_size + 1):
                yield i, j, image[i:i+self.filter_size, j:j+self.filter_size]

    def forward(self, input):
        self.last_input = input
        h, w = input.shape
        output = np.zeros((h - self.filter_size + 1, w - self.filter_size + 1, self.num_filters))
        
        for i, j, region in self.iterate_regions(input):
            output[i, j] = np.sum(region * self.filters, axis=(1, 2))
        
        return output
    
    def backward(self, dL_dout, learning_rate):
        dL_dfilters = np.zeros(self.filters.shape)
        
        for i, j, region in self.iterate_regions(self.last_input):
            for f in range(self.num_filters):
                dL_dfilters[f] += dL_dout[i, j, f] * region
        
        self.filters -= learning_rate * dL_dfilters
        return None  # Simplified, full backprop requires more

class MaxPool2D:
    def __init__(self, size):
        self.size = size
    
    def iterate_regions(self, image):
        h, w, num_filters = image.shape
        new_h = h // self.size
        new_w = w // self.size
        
        for i in range(new_h):
            for j in range(new_w):
                yield i, j, image[i*self.size:(i+1)*self.size, j*self.size:(j+1)*self.size]
    
    def forward(self, input):
        self.last_input = input
        h, w, num_filters = input.shape
        output = np.zeros((h // self.size, w // self.size, num_filters))
        
        for i, j, region in self.iterate_regions(input):
            output[i, j] = np.max(region, axis=(0, 1))
        
        return output
    
class FullyConnected:
    def __init__(self, input_size, output_size):
        self.weights = np.random.randn(input_size, output_size) / input_size
        self.biases = np.zeros(output_size)
    
    def forward(self, input):
        self.last_input = input
        return np.dot(input, self.weights) + self.biases
    
    def backward(self, dL_dout, learning_rate):
        dL_dweights = np.dot(self.last_input.T, dL_dout)
        dL_dbiases = np.sum(dL_dout, axis=0)
        
        self.weights -= learning_rate * dL_dweights
        self.biases -= learning_rate * dL_dbiases
        return None  # Simplified, full backprop requires more

class CNN:
    def __init__(self):
        self.conv = Conv2D(8, 3)
        self.pool = MaxPool2D(2)
        self.fc = FullyConnected(13*13*8, 10)  # Assuming input is 28x28 (MNIST-like)
    
    def forward(self, input):
        output = self.conv.forward(input)
        output = self.pool.forward(output)
        output = output.flatten()
        output = self.fc.forward(output)
        return output

# Example usage
if __name__ == "__main__":
    sample_input = np.random.randn(28, 28)  # Example grayscale image
    cnn = CNN()
    output = cnn.forward(sample_input)
    print("CNN Output:", output)
