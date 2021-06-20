
import os
import sox
import argparse
import os.path as osp
import numpy as np
import scipy.io.wavfile as wavefile
from playsound import playsound

def parse_args():

    parser = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )

    default_input_path = 'data/hello.wav'
    default_output_path = 'data/out.wav'
    choices_tr = ['pitch', 'treble', 'bass', 'distortion', 'echo', 'reverb', 'chorus', 'flanger', 'tremolo']

    parser.add_argument('-i', '--input_path', default=default_input_path, type=str, help='input path')
    parser.add_argument('-o', '--output_path', default=default_output_path, type=str, help='output path')
    parser.add_argument('--tr', default=[], choices=choices_tr, type=str, nargs='+', help='transoforms')
    parser.add_argument('-pdb', action='store_true', help='run with pdb')

    return parser.parse_args()


if __name__ == '__main__':

    args = parse_args()

    if args.pdb:
        import pdb
        pdb.set_trace()
    tfm = sox.Transformer()

    # pitch
    if 'pitch' in args.tr:
        tfm.pitch(-10)
    if 'treble' in args.tr:
        # treble(gain_db: float, frequency: float = 3000.0, slope: float = 0.5)
        tfm.treble(10)
    if 'bass' in args.tr:
        # bass(gain_db: float, frequency: float = 100.0, slope: float = 0.5)
        tfm.bass(10)
    if 'distortion' in args.tr:
        # overdrive(gain_db: float = 20.0, colour: float = 20.0)
        tfm.overdrive()
    if 'echo' in args.tr:
        # echo(gain_in: float = 0.8, gain_out: float = 0.9, n_echos: int = 1, delays: List[float] = [60], decays: List[float] = [0.4])
        tfm.echo()
    if 'reverb' in args.tr:
        # reverb(reverberance: float = 50, high_freq_damping: float = 50, room_scale: float = 100, stereo_depth: float = 100, pre_delay: float = 0, wet_gain: float = 0, wet_only: bool = False)
        tfm.reverb()
    if 'chorus' in args.tr:
        # chorus(gain_in: float = 0.5, gain_out: float = 0.9, n_voices: int = 3, delays: Optional[List[float]] = None, decays: Optional[List[float]] = None, speeds: Optional[List[float]] = None, depths: Optional[List[float]] = None, shapes: Optional[List[typing_extensions.Literal['s', 't'][s, t]]] = None)
        tfm.chorus()
    if 'flanger' in args.tr:
        # flanger(delay: float = 0, depth: float = 2, regen: float = 0, width: float = 71, speed: float = 0.5, shape: typing_extensions.Literal['sine', 'triangle'][sine, triangle] = 'sine', phase: float = 25, interp: typing_extensions.Literal['linear', 'quadratic'][linear, quadratic] = 'linear')
        tfm.flanger()
    if 'tremolo' in args.tr:
        # tremolo(speed: float = 6.0, depth: float = 40.0)
        tfm.tremolo()


    if osp.exists(args.output_path):
        os.remove(args.output_path)
    tfm.build_file(args.input_path, args.output_path)
    # playsound(args.output_path)
    # os.system('start ' + args.output_path)