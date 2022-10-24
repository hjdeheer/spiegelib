import subprocess
import glob
import os
import json

from tqdm import tqdm

import spiegelib as spgl
import numpy as np
import shutil
from dataclasses import dataclass


@dataclass
class Parameter:
    link: int
    min: int
    max: int


def remove_patches():
    dir = "C:/Users/hugod/Desktop/DX7toXFM2"
    files = [os.remove(x) for x in glob.glob(f"{dir}/*.syx")]

def setup_scripts():
    #Remove all curr patches
    dirMove = "C:/Users/hugod/Desktop/DX7toXFM2"
    dir = "C:/Users/hugod/Desktop/DX7toXFM2/dexedpresets"
    for filename in glob.glob(f"{dir}/**/*.SYX", recursive=True):
        shutil.copy(filename, f"{dirMove}/{os.path.basename(filename).lower()}")



def execute_scripts():
    dir = "C:/Users/hugod/Desktop/DX7toXFM2"
    files = [os.path.basename(x).lower() for x in glob.glob(f"{dir}/*.syx")]
    os.chdir(dir)
    for syxFile in tqdm(files):
        command = f"java -classpath .;.\json_simple-1.1.jar ReadSysEx {syxFile}"
        os.system(command)

def load_json():
    dir = "C:/Users/hugod/Desktop/DX7toXFM2/out/DX7"
    files = glob.glob(f"{dir}/*.json")
    allPatches = []
    for jsonFile in tqdm(files):
        with open(jsonFile) as json_file:
            try:
                data = json.load(json_file, strict=False)
                allPatches.append(data['parameters'])
            except Exception:
                continue

    print(len(allPatches))
    allPatches = np.array(allPatches[:60000])
    np.save("../data/presets/jsonPresets.npy", allPatches)
    return allPatches

def get_correct_param_linking():
    #Here we want an array with the same order of parameters in DawDreamer
    #patches[0] should be a dict with a min / max value and values
    #Maps params to dawdreamer params

    linkingDict = {}
    opLength = 21
    ops = 6
    offset = 23
    #All OP done
    for i in range(ops):
        for j in range(opLength):
            k = (j) + (i * opLength)
            if j <= 7:
                linkingDict[k] = Parameter(offset + k + i, 0, 99)
            elif 8 <= j <= 10:
                linkingDict[k] = Parameter(offset + k + 5 + i, 0, 99)
            elif j == 11 or j == 12:
                linkingDict[k] = Parameter(offset + k + 5 + i, 0, 3)
            elif j == 13 or j == 15:
                linkingDict[k] = Parameter(offset + k + 5 + i, 0, 7)
            elif j == 14:
                linkingDict[k] = Parameter(offset + k + 5 + i, 0, 3)
            elif j == 16:
                linkingDict[k] = Parameter(offset + k - 8 + i, 0, 99)
            elif j == 17:
                linkingDict[k] = Parameter(offset + k - 8 + i, 0, 1)
            elif j == 18:
                linkingDict[k] = Parameter(offset + k - 8 + i, 0, 31)
            elif j == 19:
                linkingDict[k] = Parameter(offset + k - 8 + i, 0, 99)
            elif j == 20:
                linkingDict[k] = Parameter(offset + k - 8 + i, 0, 14)
    linkingDict[126] = Parameter(15, 0, 99)
    linkingDict[127] = Parameter(16, 0, 99)
    linkingDict[128] = Parameter(17, 0, 99)
    linkingDict[129] = Parameter(18, 0, 99)
    linkingDict[130] = Parameter(19, 0, 99)
    linkingDict[131] = Parameter(20, 0, 99)
    linkingDict[132] = Parameter(21, 0, 99)
    linkingDict[133] = Parameter(22, 0, 99)
    linkingDict[134] = Parameter(4, 0, 31)
    linkingDict[135] = Parameter(5, 0, 7)
    linkingDict[136] = Parameter(6, 0, 1)
    linkingDict[137] = Parameter(7, 0, 99)
    linkingDict[138] = Parameter(8, 0, 99)
    linkingDict[139] = Parameter(9, 0, 99)
    linkingDict[140] = Parameter(10, 0, 99)
    linkingDict[141] = Parameter(11, 0, 1)
    linkingDict[142] = Parameter(12, 0, 5)
    linkingDict[143] = Parameter(14, 0, 7)
    linkingDict[144] = Parameter(13, 0, 48)

    #Missing all switches and 0 - 4
    return linkingDict

#Here we want an array with the same order of parameters in DawDreamer
#patches[0] should be a dict with a min / max value and values
def model_params(linking):
    patches = np.load("../data/presets/jsonPresets.npy", allow_pickle=True)
    np.random.shuffle(patches)
    allParams = []
    for i in range(155):
        allParams.append({})
    for patch in patches:
        for paramDict in patch[:145]:
            key = paramDict['Par#']
            value = paramDict['Value']
            paramObj = linking[key]
            #If dictionary is empty
            if not bool(allParams[paramObj.link]):
                allParams[paramObj.link]['min'] = paramObj.min
                allParams[paramObj.link]['max'] = paramObj.max
                allParams[paramObj.link]['value'] = []
                allParams[paramObj.link]['value'].append(value)
            #If already values in dict:
            else:
                allParams[paramObj.link]['value'].append(value)

    allParams = np.array(allParams)

    #Normalize params and calulate std and mean
    for paramDict in allParams:
        if not bool(paramDict):
            continue
        values = np.array(paramDict['value'])
        normalizedValues = (values - paramDict['min']) / (paramDict['max'] - paramDict['min'])
        paramDict['value'] = normalizedValues
        paramDict['std'] = np.std(normalizedValues)
        paramDict['mean'] = np.mean(normalizedValues)
    allParams = np.array(allParams)
    np.save("../data/presets/allParams.npy", allParams)
    return allParams

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


def updateParameterModel(parameterModel):
    extendParamModel(parameterModel, synth.parametersDesc)
    np.save("../data/presets/allParamsUpdated.npy", parameterModel)
    np.save("../data/presets/automatableKeys.npy", np.array(synth.get_automatable_keys()))
    return parameterModel

import pandas as pd
synth = spgl.synth.SynthDawDreamer("../vsts/Dexed.dll",
                            note_length_secs=1.0,
                            render_length_secs=1.0)

#load 9 param config
synth.load_state("../vsts/op2_dexed.json")

if __name__ == "__main__":
    #remove_patches()
    #setup_scripts()
    #execute_scripts()
    #jsonPatches = load_json()
    #Load the patches
    # jsonPatches = np.load("../data/presets/jsonPresets.npy", allow_pickle=True)
    linking = get_correct_param_linking()
    allParams = model_params(linking)
    updateParameterModel(allParams)



