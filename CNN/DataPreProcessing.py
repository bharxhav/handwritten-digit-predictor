from tensorflow import keras


class DataPreProcessing:
    def __init__(self) -> None:
        pass

    def load_data(self):
        (xtrain, ytrain), (xtest, ytest) = keras.datasets.mnist.load_data()

        self.xtrain = xtrain
        self.xtest = xtest
        self.ytrain = ytrain
        self.ytest = ytest

    def normalize(self):
        self.xtrain /= 255.0
        self.xtest /= 255.0

        self.ytrain = keras.utils.to_categorical(self.ytrain)
        self.ytest = keras.utils.to_categorical(self.ytest)

    def get_data(self):
        return (self.xtrain, self.ytrain), (self.xtest, self.ytest)
