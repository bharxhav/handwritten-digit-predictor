# Handwritten Digit Predictor

A server side Machine-Learning model that can recognize a digit that is drawn using mouse across thousands of written characters.

![Demo](./assets/HandwrittenDemo.gif)

---

## Backend

The model is made using CNN. Thanks to tensorflow, the model can be summarized as follows:

```python
# Convolution Neural Network
model = keras.Sequential()

# Layer 1: Conv 2D
model.add(keras.layers.Conv2D(32, (5, 5), padding='same', input_shape=[28, 28, 1]))
# Layer 2: Max Pool 2D
model.add(keras.layers.MaxPool2D(2, 2))
# Layer 3: Conv 2D
model.add(keras.layers.Conv2D(64, (5, 5), padding='same'))
# Layer 4: Max Pool
model.add(keras.layers.MaxPool2D((2, 2)))
# Layer 5: Flatten for Nodes
model.add(keras.layers.Flatten())
# Layer 6: Dense Layer
model.add(keras.layers.Dense(1024, activation='relu'))
# Layer 7: Dropout
model.add(keras.layers.Dropout(0.2))
# Layer 8: Output Layer
model.add(keras.layers.Dense(10, activation='softmax'))

# Compilation
model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
```

This model is later saved into TensorflowJS model, which is invoked in website's backend javascript.

---

## Credits

The UI is borrowed from [Benson Ruan's](https://medium.com/@bensonruan) drawing board tutorial.
