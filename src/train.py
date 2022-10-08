import numpy as np
import tensorflow as tf

from spiegelib import estimator
if __name__ == "__main__":
    trainFeatures = np.load("../data/dataset_uniform/STFT/train_features.npy")
    trainParams = np.load("../data/dataset_uniform/patch/train_patches.npy")
    testFeatures = np.load("../data/dataset_uniform/STFT/test_features.npy")
    testParams = np.load("../data/dataset_uniform/patch/test_patches.npy")

    # Create "STFT Images" with one channel for 2D CNN
    trainFeatures = trainFeatures.reshape(trainFeatures.shape[0], trainFeatures.shape[1], trainFeatures.shape[2], 1)
    testFeatures = testFeatures.reshape(testFeatures.shape[0], testFeatures.shape[1], testFeatures.shape[2], 1)
    print(trainFeatures.shape)
    print(testFeatures.shape)

    # Setup callbacks for trainings
    logger = estimator.TFEpochLogger()
    earlyStopping = tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=10)

    # Instantiate Conv6 Model with the input shape, output shape, and callbacks
    cnn = estimator.Conv8(trainFeatures.shape[1:],
                               trainParams.shape[-1],
                               callbacks=[logger, earlyStopping])

    cnn.add_training_data(trainFeatures, trainParams)
    cnn.add_testing_data(testFeatures, testParams)
    cnn.model.summary()
    cnn.fit(epochs=100)
    cnn.save_model('../data/models/cnn_vgg11_uniform.h5')
    logger.plot()