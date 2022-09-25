#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Init for synth components
"""

from .synth_base import SynthBase

try:
    import librenderman
    from .synth_vst import SynthVST
except:
    print("librenderman package not installed, SynthVST class is unavailable. To use VSTs please install librenderman.")
    print("https://spiegelib.github.io/spiegelib/getting_started/installation.html")


try:
    import dawdreamer as daw
    from .synth_dawdreamer import SynthDawDreamer
except:
    print("Dawdreamer package not installed, SynthDawDreamer class is unavailable. To use VSTs please install dawdreamer using pip install dawdreamer.")



