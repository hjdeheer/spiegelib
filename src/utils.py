import numpy as np
import spiegelib as spgl

def extendParamModel(parameterModel, synthDescription):
    #Add min-max for missing variables
    for i, (pModel, pSynth) in enumerate(zip(parameterModel, synthDescription)):
        if 0 <= i <= 3:
            pModel['min'] = 0
            pModel['max'] = 99
            pModel['name'] = pSynth['name']
            pModel['isDiscrete'] = False
        elif 'min' not in pModel:
            #These are switches
            pModel['min'] = 0
            pModel['max'] = 1
            pModel['name'] = pSynth['name']
            pModel['isDiscrete'] = True
    for pModel, pSynth in zip(parameterModel, synthDescription):
        pModel['name'] = pSynth['name']
        #If parameter ranges from 0-99 it is not discrete
        if pModel['max'] == 99:
            pModel['isDiscrete'] = False
        else:
            pModel['isDiscrete'] = True


import pandas as pd
synth = spgl.synth.SynthDawDreamer("../vsts/Dexed.dll",
                            note_length_secs=1.0,
                            render_length_secs=1.0)
#load 9 param config
synth.load_state("../vsts/op2_dexed.json")


def getParameterModel():
    parameterModel = np.load("../data/presets/allParams.npy", allow_pickle=True)
    extendParamModel(parameterModel, synth.parametersDesc)
    return parameterModel