
import os
import sox
import argparse
import os.path as osp
import numpy as np

def parse_args():

    parser = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )

    choices_tr = ['pitch', 'treble', 'bass', 'distortion', 'echo', 'reverb', 'chorus', 'flanger', 'tremolo']

    parser.add_argument('-i', '--inpath', required=True, type=str, help='input path')
    parser.add_argument('-o', '--outpath', required=True, type=str, help='output path')
    parser.add_argument('--tr', default=[], choices=choices_tr, type=str, nargs='+', help='transoforms')


    default_pitch_n_semitones = 3
    default_treble_gain_db = 3
    default_treble_frequency = 3000
    default_treble_slope = 0.5
    default_bass_gain_db = 3
    default_bass_frequency = 100
    default_bass_slope = 0.5
    default_distortion_gain_db = 20
    default_distortion_colour = 20
    default_echo_gain_in = 0.8
    default_echo_gain_out = 0.9
    default_echo_n_echos = 1
    default_echo_delays = [60]
    default_echo_decays = [0.4]
    default_reverb_reverberance = 50
    default_reverb_high_freq_damping = 50
    default_reverb_room_scale = 100
    default_reverb_stereo_depth = 100
    default_reverb_pre_delay = 0
    default_reverb_wet_gain = 0
    default_reverb_wet_only = False
    default_chorus_gain_in = 0.5
    default_chorus_gain_out = 0.9
    default_chorus_n_voices = 3
    default_chorus_delays = None
    default_chorus_decays = None
    default_chorus_speeds = None
    default_chorus_depths = None
    default_chorus_shapes = None
    default_flanger_delay = 0
    default_flanger_depth = 2
    default_flanger_regen = 0
    default_flanger_width = 71
    default_flanger_speed = 0.5
    default_flanger_shape = 'sine'
    default_flanger_phase = 25
    default_flanger_interp = 'linear'
    default_tremolo_speed = 6
    default_tremolo_depth = 40

    parser.add_argument('--pitch_n_semitones', type=float, default=default_pitch_n_semitones, help='pitch n semitones')
    parser.add_argument('--treble_gain_db', type=float, default=default_treble_gain_db, help='treble gain_db')
    parser.add_argument('--treble_frequency', type=float, default=default_treble_frequency, help='treble frequency')
    parser.add_argument('--treble_slope', type=float, default=default_treble_slope, help='treble slope')
    parser.add_argument('--bass_gain_db', type=float, default=default_bass_gain_db, help='bass gain_db')
    parser.add_argument('--bass_frequency', type=float, default=default_bass_frequency, help='bass frequency')
    parser.add_argument('--bass_slope', type=float, default=default_bass_slope, help='bass slope')
    parser.add_argument('--distortion_gain_db', type=float, default=default_distortion_gain_db, help='distortion gain db')
    parser.add_argument('--distortion_colour', type=float, default=default_distortion_colour, help='distortion colour')
    parser.add_argument('--echo_gain_in', type=float, default=default_echo_gain_in, help='echo gain_in')
    parser.add_argument('--echo_gain_out', type=float, default=default_echo_gain_out, help='echo gain_out')
    parser.add_argument('--echo_n_echos', type=int, default=default_echo_n_echos, help='echo n_echos')
    parser.add_argument('--echo_delays', type=float, nargs='+', default=default_echo_delays, help='echo delays')
    parser.add_argument('--echo_decays', type=float, nargs='+', default=default_echo_decays, help='echo decays')
    parser.add_argument('--reverb_reverberance', type=float, default=default_reverb_reverberance, help='reverb reverberance')
    parser.add_argument('--reverb_high_freq_damping', type=float, default=default_reverb_high_freq_damping, help='reverb high_freq_damping')
    parser.add_argument('--reverb_room_scale', type=float, default=default_reverb_room_scale, help='reverb room scale')
    parser.add_argument('--reverb_stereo_depth', type=float, default=default_reverb_stereo_depth, help='reverb stereo depth')
    parser.add_argument('--reverb_pre_delay', type=float, default=default_reverb_pre_delay, help='reverb pre_delay')
    parser.add_argument('--reverb_wet_gain', type=float, default=default_reverb_wet_gain, help='reverb wet_gain')
    parser.add_argument('--reverb_wet_only', type=bool, default=default_reverb_wet_only, help='reverb wet_only')
    parser.add_argument('--chorus_gain_in', type=float, default=default_chorus_gain_in, help='chorus gain_in')
    parser.add_argument('--chorus_gain_out', type=float, default=default_chorus_gain_out, help='chorus gain_out')
    parser.add_argument('--chorus_n_voices', type=int, default=default_chorus_n_voices, help='chorus n_voices')
    parser.add_argument('--chorus_delays', type=float, nargs='+', default=default_chorus_delays, help='chorus delays')
    parser.add_argument('--chorus_decays', type=float, nargs='+', default=default_chorus_decays, help='chorus decays')
    parser.add_argument('--chorus_speeds', type=float, nargs='+', default=default_chorus_speeds, help='chorus speeds')
    parser.add_argument('--chorus_depths', type=float, nargs='+', default=default_chorus_depths, help='chorus depths')
    parser.add_argument('--chorus_shapes', type=str, nargs='+', default=default_chorus_shapes, help='chorus shapes')
    parser.add_argument('--flanger_delay', type=float, default=default_flanger_delay, help='flanger delay')
    parser.add_argument('--flanger_depth', type=float, default=default_flanger_depth, help='flanger depth')
    parser.add_argument('--flanger_regen', type=float, default=default_flanger_regen, help='flanger regen')
    parser.add_argument('--flanger_width', type=float, default=default_flanger_width, help='flanger width')
    parser.add_argument('--flanger_speed', type=float, default=default_flanger_speed, help='flanger speed')
    parser.add_argument('--flanger_shape', type=str, nargs='+', default=default_flanger_shape, help='flanger shape')
    parser.add_argument('--flanger_phase', type=float, default=default_flanger_phase, help='flanger phase')
    parser.add_argument('--flanger_interp', type=str, nargs='+', default=default_flanger_interp, help='flanger interp')
    parser.add_argument('--tremolo_speed', type=float, default=default_tremolo_speed, help='tremolo speed')
    parser.add_argument('--tremolo_depth', type=float, default=default_tremolo_depth, help='tremolo depth')

    parser.add_argument('-pdb', action='store_true', help='run with pdb')

    return parser.parse_args()


if __name__ == '__main__':

    args = parse_args()

    if args.pdb:
        import pdb
        pdb.set_trace()

    assert osp.exists(args.inpath)

    tfm = sox.Transformer()

    # pitch
    if 'pitch' in args.tr:
        # pitch(n_semitones: float, quick: bool = False)
        tfm.pitch(args.pitch_n_semitones)
    if 'treble' in args.tr:
        # treble(gain_db: float, frequency: float = 3000.0, slope: float = 0.5)
        tfm.treble(args.treble_gain_db,
            frequency=args.treble_frequency,
            slope=args.treble_slope
        )
    if 'bass' in args.tr:
        # bass(gain_db: float, frequency: float = 100.0, slope: float = 0.5)
        tfm.bass(args.bass_gain_db,
            frequency=args.bass_frequency,
            slope=args.bass_slope
        )
    if 'distortion' in args.tr:
        # overdrive(gain_db: float = 20.0, colour: float = 20.0)
        tfm.overdrive(args.distortion_gain_db, 
            colour=args.distortion_colour
        )
    if 'echo' in args.tr:
        # echo(gain_in: float = 0.8, gain_out: float = 0.9, n_echos: int = 1, delays: List[float] = [60], decays: List[float] = [0.4])
        tfm.echo(
            gain_in=args.echo_gain_in,
            gain_out=args.echo_gain_out,
            n_echos=args.echo_n_echos,
            delays=args.echo_delays,
            decays=args.echo_decays
        )
    if 'reverb' in args.tr:
        # reverb(reverberance: float = 50, high_freq_damping: float = 50, room_scale: float = 100, stereo_depth: float = 100, pre_delay: float = 0, wet_gain: float = 0, wet_only: bool = False)
        tfm.reverb(
            reverberance=args.reverb_reverberance,
            high_freq_damping=args.reverb_high_freq_damping,
            room_scale=args.reverb_room_scale,
            stereo_depth=args.reverb_stereo_depth,
            pre_delay=args.reverb_pre_delay,
            wet_gain=args.reverb_wet_gain,
            wet_only=args.reverb_wet_only
        )
    if 'chorus' in args.tr:
        # chorus(gain_in: float = 0.5, gain_out: float = 0.9, n_voices: int = 3, delays: Optional[List[float]] = None, decays: Optional[List[float]] = None, speeds: Optional[List[float]] = None, depths: Optional[List[float]] = None, shapes: Optional[List[typing_extensions.Literal['s', 't'][s, t]]] = None)
        tfm.chorus(
            gain_in=args.chorus_gain_in,
            gain_out=args.chorus_gain_out,
            n_voices=args.chorus_n_voices,
            delays=args.chorus_delays,
            decays=args.chorus_decays,
            speeds=args.chorus_speeds,
            depths=args.chorus_depths,
            shapes=args.chorus_shapes
        )
    if 'flanger' in args.tr:
        # flanger(delay: float = 0, depth: float = 2, regen: float = 0, width: float = 71, speed: float = 0.5, shape: typing_extensions.Literal['sine', 'triangle'][sine, triangle] = 'sine', phase: float = 25, interp: typing_extensions.Literal['linear', 'quadratic'][linear, quadratic] = 'linear')
        tfm.flanger(
            delay=args.flanger_delay,
            depth=args.flanger_depth,
            regen=args.flanger_regen,
            width=args.flanger_width,
            speed=args.flanger_speed,
            shape=args.flanger_shape,
            phase=args.flanger_phase,
            interp=args.flanger_interp
        )
    if 'tremolo' in args.tr:
        # tremolo(speed: float = 6.0, depth: float = 40.0)
        tfm.tremolo(
            speed=args.tremolo_speed,
            depth=args.tremolo_depth
        )

    tfm.build_file(args.inpath, args.outpath)
