import os
import ffmpeg
import argparse
import shutil
import os.path as osp
import subprocess

def parse_args():

    file_description = '''
    RHA - RedHenAnonymizer
    Red Hen Lab

    Given a video or audio, this tool anonymizes the face of person and his/her voice. 

    You can either hide face or swap face with some other face. Audio is anonymized by changing the pitch.

    Hider:
    Replaces the face with a white rectangle
    Usage: python rha.py --inpath <input_video_path> --outpath <output_video_path>

    Swapper:
    Swaps the face in video with a specified face. This is provided using the --facepath argument
    Usage: python rha.py --inpath <input_video_path> --facepath facebank/white_male/1.jpg --outpath <output_video_path> 

    Audio:
    The pitch of audio is changed by --pitch argument. It has to be an integer. Normally, 3 or -3 will work good.
    Increasing the pitch makes voice more female-like. While decreasing it makes male-like. 
    Usage: python rha.py --inpath <input_video_path> --outpath <output_video_path> --pitch 5
    '''    
    parser = argparse.ArgumentParser(
        description=file_description, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    
    parser.add_argument('--inpath', type=str, required=True, help='path to input video')
    parser.add_argument('--facepath', type=str, default="", help='path to anonymous (target) face, can be a static video or a image. If no facepath is provided, the face will be hided. Hence, this argument is optional.')
    parser.add_argument('--outpath', type=str, required=True, help='path to output video, should be a mp4 video')
    parser.add_argument('--pitch', type=float, default=3, help='pitch change amount, can be +/-')
    parser.add_argument('--cpu_only', action='store_true', help='Run on cpu only. However this flag is only for swapper. Hider will use/not use gpu depending on the tensorflow type you have installed. For tensorflow you can have a gpu or a cpu version.')
    parser.add_argument('-pdb', action='store_true', help='run with pdb debugger')
    
    return parser.parse_args()

if __name__ == '__main__':
    
    args = parse_args()
    if args.pdb: 
        import pdb
        pdb.set_trace()
        
    assert osp.exists(args.inpath), 'path not found: ' + args.inpath

    
    temp_dir = osp.abspath('anon_tmp_{}_{}'.format(osp.basename(args.inpath), osp.basename(args.facepath)))
    os.makedirs(temp_dir, exist_ok=True)
    inpath = osp.abspath(args.inpath)
    outpath = osp.abspath(args.outpath)
    
    metadata = ffmpeg.probe(inpath)
    assert len(metadata) == 2, '2 streams are not found in input'
    
    video_meta = None
    audio_meta = None
    for m in metadata['streams']:
        if m['codec_type'] == 'audio':
            audio_meta = m
        elif m['codec_type'] == 'video':
            video_meta = m
        else:
            raise Exception("unknown codec_type: {}".format(m['codec_type']))
    
    video_start = None
    audio_start = None
    assert (video_meta or audio_meta), 'neither video nor audio is present in input file'
    if video_meta is not None:
        video_start = float(video_meta['start_time'])
    if audio_meta is not None:
        audio_start = float(audio_meta['start_time'])
    
    video_start_arg = ""
    audio_start_arg = ""
    if video_start:
        video_start_arg = f" -itsoffset {video_start} "
    if audio_start:
        audio_start_arg = f" -itsoffset {audio_start} "
        

    if args.facepath:
        swappy_path = osp.join(osp.dirname(__file__), 'fsgan', 'inference', 'swap.py')
        assert osp.exists(swappy_path), f'path not found: {swappy_path}'
        facepath = osp.abspath(args.facepath)
        assert osp.exists(facepath), f'facepath does not exist {facepath}'
        print("Swapping faces, with the face:", facepath)
        print('Input video:', inpath)
        device_flag = ""
        if args.cpu_only:
            device_flag = " --cpu_only "
        # swap faces
        fsgan_outpath = osp.join(temp_dir, 'fsgan_out.mp4')
        command = f'python {swappy_path} {facepath} -t {inpath} -o {fsgan_outpath} --seg_remove_mouth --encoder_codec mp4v {device_flag}'
        error = os.system(command)
        if error:
            raise Exception(f'unable to swap faces. Check fsgan. error code: {error}')
        
        visually_anon_output = fsgan_outpath
    else:    
        print("Hiding the face, as the facepath argument was empty")
        visually_anon_output = osp.join(temp_dir, "hidden_face.mp4")
        hide_face_py_path = osp.join(osp.dirname(__file__), 'hide_face_robust.py')
        assert osp.exists(hide_face_py_path), f"file not found: {hide_face_py_path}"
        command = f"python {hide_face_py_path} --inpath {inpath} --outpath {visually_anon_output}"
        error = os.system(command)
        if error:
            raise Exception(f"unable to run face hider. Check hide_face_robust.py. error code: {error}")
    
    audcodec = 'wav'
    tmpaudiopath1 = osp.join(temp_dir, f'aud1.{audcodec}')  
    tmpaudiopath2 = osp.join(temp_dir, f'aud2.{audcodec}')
    
    subprocess.call(
        f"ffmpeg -y -i {inpath} -vn {tmpaudiopath2}",
        shell=True
    )
    
    audio_py_path = osp.join(osp.dirname(__file__), 'audio.py')
    assert osp.exists(audio_py_path), f"file not found: {audio_py_path}"
    cmd = f"python {audio_py_path} --inpath {tmpaudiopath2} --outpath {tmpaudiopath1} --tr pitch --pitch_n_semitones {args.pitch}"
    error = os.system(
        cmd
    )
    if error:
        raise Exception(f"unable to change audio. Check audio.py. error code: {error}")
    

    cmd = f"ffmpeg -y {video_start_arg} -i {visually_anon_output} {audio_start_arg} -i {tmpaudiopath1} -vcodec copy -acodec aac -map 0:v:0 -map 1:a:0 {outpath}"
    print('combining audio and video with command:', cmd)
    subprocess.call(
        cmd,
        shell=True
    )
    shutil.rmtree(temp_dir)
    print('video saved at:', outpath)
    
