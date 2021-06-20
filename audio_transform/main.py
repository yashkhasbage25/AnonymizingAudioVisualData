
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

    parser.add_argument('-i', '--input_path', default=default_input_path, type=str, help='input path')
    parser.add_argument('-o', '--output_path', default=default_output_path, type=str, help='output path')
    parser.add_argument('--tr', type=str, nargs='+', help='transoforms')
    parser.add_argument('-pdb', action='store_true', help='run with pdb')

    return parser.parse_args()


if __name__ == '__main__':

    args = parse_args()

    if args.pdb:
        import pdb
        pdb.set_trace()
    tfm = sox.Transformer()


    # tfm.pitch(-10)
    tfm.echo()

    if osp.exists(args.output_path):
        os.remove(args.output_path)
    tfm.build_file(args.input_path, args.output_path)
    # playsound(args.output_path)
    # os.system('start ' + args.output_path)