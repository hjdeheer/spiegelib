#!/usr/bin/env python
"""
This class can be used to generate datasets from instances of synthesizers. This is
useful for creating large sets of data for training and validating deep learning models.
Examples
^^^^^^^^
Example generating 50000 training samples and 10000 testing samples from the
*Dexed* VST FM Synthesizer. Each sample is created by creating a random
patch configuration in *Dexed*, and then rendering a one second audio clip of
that patch. A 13-band MFCC is computed on the resulting audio. These audio features
and the synthesizer parameters used to synthesize the audio are saved in numpy files.
Audio features are standardized by removing the mean and scaling to unit variance. The
values used for scaling are saved after the first dataset generation so they
can be used on future data.
.. code-block:: python
    :linenos:
    import spiegelib as spgl
    synth = spgl.synth.SynthVST("/Library/Audio/Plug-Ins/VST/Dexed.vst",
                                note_length_secs=1.0,
                                render_length_secs=1.0)
    # Mel-frequency Cepstral Coefficients audio feature extractor.
    feature = spgl.features.MFCC(num_mfccs=13, frame_size=2048,
                                  hop_size=1024 time_major=True)
    features = [feature]
    # Setup generator for MFCC output and generate 50000 training examples
    # and 10000 testing examples
    generator = spgl.DatasetGenerator(synth, features,
                                      output_folder="./data_FM_mfcc",
                                      normalize=True)
    generator.generate(50000, file_prefix="train_")
    generator.generate(10000, file_prefix="test_")
    generator.save_scaler('data_scaler.pkl')
"""

import os

import numpy as np
import scipy.io.wavfile
from tqdm import trange
import tensorflow as tf
from spiegelib.features.features_base import FeaturesBase
from spiegelib.synth.synth_base import SynthBase
from scipy.ndimage import gaussian_filter1d

class DatasetGenerator():
    """
    Args:
        synth (Object): Synthesizer to generate test data from. Must inherit from
            :class:`spiegelib.synth.SynthBase`.
        features (list): List of Feature object to use for dataset generation. Feature obj must inherit from
            :class:`spiegelib.features.FeaturesBase`.
        output_folder (str, optional): Output folder for dataset, defaults to currect working directory.
        save_audio (bool, optional): Whether or not to save rendered audio files, defaults to False.
        scale (list, optional): list of whether or not to scale resulting feature vector. If the feature
            object does not have a scaler set, then this will train a data scaler based on the
            generated dataset and store them in the features object. Call :py:meth:`save_scaler`
            to store scaler settings. Defaults to a list of False
    Attributes:
        features_filename (str): filename for features output file, defaults to features.npy
        patches_filename (str): filename for patches output file, defaults to patches.npy
        audio_folder_name (str): folder name for the audio output if used. Will be automatically
            created within the output folder if saving audio. Defaults to audio
    """
    #TODO Add sampling technique param to constructor
    def __init__(self, synth, features, output_folder=os.getcwd(), save_audio=False, scale=None):
        """
        Contructor
        """

        #Initialze scale to false
        if scale is None:
            scale = [False] * len(features)

        # Check for valid synth
        if isinstance(synth, SynthBase):
            self.synth = synth
        else:
            raise TypeError('synth must inherit from SynthBase')

        # Check for valid features
        allValid = True
        for feature in features:
            if not isinstance(feature, FeaturesBase):
                allValid = False
                raise TypeError(f'Feature {feature} must inherit from FeaturesBase')
        if allValid:
            self.features = features


        # Check for valid output folder
        self.output_folder = os.path.abspath(output_folder)
        if not (os.path.exists(self.output_folder) and os.path.isdir(self.output_folder)):
            os.mkdir(self.output_folder)

        self.save_audio = save_audio

        # Default folder for audio output
        self.audio_folder_name = "audio"

        # Default folder for patch output
        self.patch_folder_name = "patch"
        self.create_patch_folder()

        # Default filenames for output files
        self.features_filename = "features.npy"
        self.patches_filename = "patches.npy"

        # Should the feature set data be scaled?
        self.should_scale = scale

        #Create all feature folders
        self.feature_folders = []
        for i in range(len(self.features)):
            featureName = str(self.features[i].__class__.__name__)
            self.feature_folders.append(featureName)
            self.create_feature_folder(featureName)




    def generate(self, size, technique='uniform', file_prefix="", fit_scaler_only=None, samples = None):
        """
        Generate dataset with a set of random patches. Saves the extracted features
        and parameter settings in separate .npy files. Files are stored in the output
        folder set during construction (defaults to current working directory) and
        saves the features as "features.npy" and patches as "patches.npy". These file
        names can be prefixed with a string set by the file_prefix argument. If audio
        files are being saved (configured during construction), then the audio files
        are saved in a separate audio folder and all audio files are also prefixed
        by the file_prefix.
        Args:
            size (int): Number of different synthesizer patches to render.
            technique (str, optional): Defines the sampling technique used for data generation
                of the parameter space.
            samples (nparray, optional): If not None, technique must be set to normal, since sampling now is done by
            drawing random samples from a normal distribution for each parameter
            file_prefix (str, optional): filename prefix for all output data.
            fit_scaler_only (list : bool, optional): If this is set to True, then
                no data will be saved and only scaler will be set or reset
                for the ith feature object.
        """
        if fit_scaler_only == None:
            fit_scaler_only = [False] * len(self.features)
        else:
            #Assert fit_scaler_only is of length n and all are booleans
            assert len(fit_scaler_only) == len(self.features)
            assert all(isinstance(el, bool) for el in fit_scaler_only)

        #if samples are not none
        if samples is not None:
            assert technique == "normal"

        # Get a single example to determine required array size required
        audio = self.synth.get_random_example(technique, samples)

        #Initialize patch set
        patch = self.synth.get_patch()
        patch_set = np.zeros((size, len(patch)), dtype=np.float32)

        #Create list to contain all feature data
        allFeatures = []
        shouldFeatureScale = []
        for i, feature in enumerate(self.features):
            currFeature = feature(audio)
            shape = list(currFeature.shape)
            shape.insert(0, size)
            feature_set = np.empty(shape, dtype=currFeature.dtype)
            # Should the features be normalized with the feature scaler?
            should_scale = self.should_scale[i] and feature.has_scaler()

            #Add feature to array
            allFeatures.append(feature_set)
            shouldFeatureScale.append(should_scale)

        #Generate all samples
        for i in trange(size, desc="Generating samples"):
            audio = self.synth.get_random_example(technique, samples)
            patch_set[i] = [p[1] for p in self.synth.get_patch()]

            #Save rendered audio if required
            if self.save_audio:
                self.create_audio_folder()
                audio.save(os.path.join(self.audio_folder_path, "%soutput_%s.wav" % (file_prefix, i)))
            #For every feature extract feature data
            for j in range(len(allFeatures)):
                allFeatures[j][i] = self.features[j](audio, scale=shouldFeatureScale[j])

        # For every feature:
        for j in range(len(allFeatures)):
            # If only fitting scaler, do that and return. Don't save any data
            if fit_scaler_only[j]:
                print("Fitting scaler only", flush=True)
                self.features[j].fit_scaler(allFeatures[j], transform=False)
                return

            if self.should_scale[j] and not self.features[j].has_scaler():
                print("Fitting scaler and scaling data", flush=True)
                allFeatures[j] = self.features[j].fit_scaler(allFeatures[j])

            # Save feature dataset
            np.save(os.path.join(self.create_feature_folder(self.feature_folders[j]), "%s%s" % (file_prefix, self.features_filename)), allFeatures[j])

        #Save patch dataset
        np.save(os.path.join(self.patch_folder_path, "%s%s" % (file_prefix, self.patches_filename)), patch_set)


    def save_scaler(self, feature_index, file_name):
        """
        Save feature scaler as a pickle file.
        Args:
            feature_index (int): feature index of features list that obj that must inherit from FeatureBase
            file_name (str): file name for scaler pickle file
        """
        try:
            feature_folder = self.create_feature_folder(self.feature_folders[feature_index])
            self.features[feature_index].save_scaler(os.path.join(feature_folder, file_name))
        except Exception as ex:
            print(f"The following exception occured: {ex}")
            raise ex


    def create_audio_folder(self):
        """
        Check for and create the audio output folder if necessary
        """
        self.audio_folder_path = os.path.abspath(os.path.join(self.output_folder, self.audio_folder_name))
        if not (os.path.exists(self.audio_folder_path) and os.path.isdir(self.audio_folder_path)):
            os.mkdir(self.audio_folder_path)




    def create_feature_folder(self, feature):
        """
        Check for and create the feature output folder if necessary
        Args:
            feature (str): Name of feature
        """
        feature_folder_path = os.path.abspath(os.path.join(self.output_folder, feature))
        if not (os.path.exists(feature_folder_path) and os.path.isdir(feature_folder_path)):
            os.mkdir(feature_folder_path)
        return feature_folder_path


    def create_patch_folder(self):
        """
        Check for and create the patch output folder if necessary
        """
        self.patch_folder_path = os.path.abspath(os.path.join(self.output_folder, self.patch_folder_name))
        if not (os.path.exists(self.patch_folder_path) and os.path.isdir(self.patch_folder_path)):
            os.mkdir(self.patch_folder_path)


    def patch_to_onehot(self, parameterModel):
        """
        Converts patches with parameters ranging from 0 - 1 to one hot encoded parameters
        """
        filenames = os.listdir(self.patch_folder_path)
        #Get all keys of params that are not overridden
        automatableParams = self.synth.get_automatable_keys()
        bins = 64
        for file in filenames:
            allPatches = []
            currPatches = np.load(os.path.join(self.patch_folder_path, file))
            for patch in currPatches:
                assert len(patch) == len(automatableParams)
                patch_onehot = np.array([])
                for i, param in enumerate(patch):
                    maxParam = parameterModel[automatableParams[i]]['max']
                    isDiscrete = parameterModel[automatableParams[i]]['isDiscrete']
                    #If continuous put in 64 bins and apply gaussian smoothing
                    if not isDiscrete:
                        value = round(param * (bins - 1))
                        onehot = tf.reshape(tf.one_hot([value], bins), bins).numpy()
                        onehot = gaussian_filter1d(onehot, 1)
                    else:
                        value = round(param * maxParam)
                        onehot = tf.reshape(tf.one_hot([value], maxParam + 1), maxParam + 1).numpy()
                    patch_onehot = np.concatenate((patch_onehot, onehot))
                allPatches.append(patch_onehot)
            allPatches = np.array(allPatches)
            np.save(os.path.join(self.patch_folder_path, "onehot_" + file), allPatches)
        return

