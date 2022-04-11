import tensorflowjs as tfjs


class ModelTraining:
    def __init__(self, model):
        self.model = model

    def train(self, xtrain, ytrain, xtest, ytest, epochs=5):
        self.model.fit(xtrain, ytrain, validation_data=(
            xtest, ytest), epochs=10)

    def save(self):
        self.model.save('../models/tf-model')
        tfjs.converters.save_keras_model(self.model, '../models/tfjs-model')
