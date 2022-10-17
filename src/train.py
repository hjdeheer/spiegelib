import numpy as np
import tensorflow as tf

from spiegelib import estimator
from spiegelib.synth import SynthDawDreamer


if __name__ == "__main__":
    feature = "STFT"
    technique = "uniform"
    parameters = 9

    trainFeatures = np.load("../data/uniform_9_50k/STFT/train_features.npy")
    trainParams = np.load("../data/uniform_9_50k/patch/onehot16_train_patches.npy")
    testFeatures = np.load("../data/uniform_9_50k/STFT/test_features.npy")
    testParams = np.load("../data/uniform_9_50k/patch/onehot16_test_patches.npy")

    # Create "STFT Images" with one channel for 2D CNN
    trainFeatures = trainFeatures.reshape(trainFeatures.shape[0], trainFeatures.shape[1], trainFeatures.shape[2], 1)
    testFeatures = testFeatures.reshape(testFeatures.shape[0], testFeatures.shape[1], testFeatures.shape[2], 1)
    print(trainFeatures.shape)
    print(testFeatures.shape)

    # Setup callbacks for trainings
    logger = estimator.TFEpochLogger()
    earlyStopping = tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=5)
    tensorboard = tf.keras.callbacks.TensorBoard(log_dir="../data/models/conv6_STFT_uniform_9_50K")

    automatable_keys = np.load('../data/presets/automatableKeys.npy', allow_pickle=True)
    automatable_params = np.load('../data/presets/allParamsUpdated.npy', allow_pickle=True)[automatable_keys]

    # Instantiate Conv6 Model with the input shape, output shape, and callbacks
    cnn = estimator.Conv6(trainFeatures.shape[1:],
                               trainParams.shape[-1], automatable_keys=automatable_params, num_bins=16,
                               callbacks=[logger, earlyStopping, tensorboard])

    cnn.add_training_data(trainFeatures, trainParams, batch_size = 64)
    cnn.add_testing_data(testFeatures, testParams, batch_size = 64)
    cnn.model.summary()
    cnn.fit(epochs=100)
    cnn.save_model('../data/models/conv6_STFT_uniform_9_50K/model.h5')
    #logger.plot()

