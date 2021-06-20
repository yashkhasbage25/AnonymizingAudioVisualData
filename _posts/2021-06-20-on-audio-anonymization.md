---
layout: post
title: How to anonymize audio?
---

The proposal stated 6 effects for anonymization: pitch, bass and treble, distortion, echo, reverb, and wah-wah effect

These are implemented for audacty. Although there exists a scripting API for audacity, the script requires audacity to be running in background. This condition is not favourable to us. 

Can these be done using some other python library? No. 
I have checked a lot of libraries on https://github.com/faroit/awesome-python-scientific-audio however, none of them supported something apart from pitch and noise. The python-audio-effects (https://github.com/carlthome/python-audio-effects)[pysndfx] supported all of them, but it does not work at all.

pysndfx is build on SoX, which has a wrapper https://pysox.readthedocs.io/en/latest/example.html . 


Pitch: pitch(n_semitones: float, quick: bool = False)

Treble: treble(gain_db: float, frequency: float = 3000.0, slope: float = 0.5)

Bass: bass(gain_db: float, frequency: float = 100.0, slope: float = 0.5)

Distortion: overdrive(gain_db: float = 20.0, colour: float = 20.0)

Echo: echo(gain_in: float = 0.8, gain_out: float = 0.9, n_echos: int = 1, delays: List[float] = [60], decays: List[float] = [0.4])	

echos(gain_in: float = 0.8, gain_out: float = 0.9, n_echos: int = 1, delays: List[float] = [60], decays: List[float] = [0.4])

reverb(reverberance: float = 50, high_freq_damping: float = 50, room_scale: float = 100, stereo_depth: float = 100, pre_delay: float = 0, wet_gain: float = 0, wet_only: bool = False)


Important pages: 
* https://pyra-handheld.com/boards/threads/help-with-using-sox-as-a-voice-changer.98878/
* https://askubuntu.com/questions/421947/is-there-a-way-to-modulate-my-voice-on-the-fly
* https://security.stackexchange.com/questions/227146/is-changing-pitch-enough-for-anonymizing-a-persons-voice