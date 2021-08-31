import os
import sys
import ffmpeg
import shutil
import argparse
import os.path as osp
import subprocess

assert sys.version_info[0] >= 3, 'Python version below 3 are not allowed'

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
    
    anonymize_choices = ['audio', 'video', 'audiovideo']
    
    parser.add_argument('-i', '--inpath', type=str, required=True, help='path to input video')
    parser.add_argument('-f', '--facepath', type=str, default="", help='path to anonymous (target) face, can be a static video or a image. If no facepath is provided, the face will be hided. Hence, this argument is optional.')
    parser.add_argument('-o', '--outpath', type=str, required=True, help='path to output video, should be a mp4 video')
    parser.add_argument('-a', '--anonymize', type=str, required=True, choices=anonymize_choices, help='anonymize which data? audio, video or audio+video')
    parser.add_argument('-p', '--pitch', type=float, default=3, help='pitch change amount, can be +/-')
    parser.add_argument('--cpu_only', action='store_true', help='Run on cpu only. However this flag is only for swapper. Hider will use/not use gpu depending on the tensorflow type you have installed. For tensorflow you can have a gpu or a cpu version.')
    parser.add_argument('-pdb', action='store_true', help='run with pdb debugger')
    
    return parser.parse_args()

def get_mediatype(metadata):
    
    codecs = [m['codec_type'] for m in metadata['streams']]    
    codecs = list(set(codecs))
    if len(codecs) == 1:
        if codecs[0] == 'video':
            return 'video'
        elif codecs[0] == 'audio':
            return 'audio'
        else:
            raise Exception(f'unknown codec_type: {codecs[0]}')
    elif len(codecs) == 2:
        if codecs[0] == 'audio' and codecs[1] == 'audio':
            return 'audio'
        elif codecs[0] == 'audio' and codecs[1] == 'video':
            return 'audiovideo'
        elif codecs[0] == 'video' and codecs[1] == 'audio':
            return 'audiovideo'
        else:
            raise Exception(f'unknown codec_types: {codecs[0]}, {codecs[1]}')
    else:
        raise Exception(f'Too many data streams found: {len(codecs)}')

def get_offset_args(metadata):
    
    video_meta = None
    audio_meta = None
    for m in metadata['streams']:
        if m['codec_type'] == 'audio':
            audio_meta = m
        elif m['codec_type'] == 'video':
            video_meta = m
        else:
            raise Exception("unknown codec_type: {}".format(m['codec_type']))
    print(metadata)
    # get start times of audio and video, if exists
    video_start = None
    audio_start = None
    assert (video_meta or audio_meta), 'neither video nor audio is present in input file'
    if video_meta is not None and 'start_time' in video_meta:
        video_start = float(video_meta['start_time'])
    if audio_meta is not None and 'start_time' in audio_meta:
        audio_start = float(audio_meta['start_time'])
    
    video_start_arg = ""
    audio_start_arg = ""
    if video_start:
        video_start_arg = f" -itsoffset {video_start} "
    if audio_start:
        audio_start_arg = f" -itsoffset {audio_start} "
        
    return audio_start_arg, video_start_arg
        

if __name__ == '__main__':
    
    args = parse_args()
    if args.pdb: 
        import pdb
        pdb.set_trace()
        
    assert osp.exists(args.inpath), 'path not found: ' + args.inpath
    try:
        # create a temporary file system to store the intermediate outputs
        temp_dir = osp.abspath('anon_tmp_{}_{}'.format(osp.basename(args.inpath), osp.basename(args.facepath)))
        os.makedirs(temp_dir, exist_ok=True)
        inpath = osp.abspath(args.inpath)
        outpath = osp.abspath(args.outpath)
        
        metadata = ffmpeg.probe(inpath)    
        
        mediatype = get_mediatype(metadata)
        anonymize = args.anonymize
        if anonymize not in mediatype:
            raise Exception(f'Cannot anonymize {anonymize} when input file is {mediatype}')
        
        audio_start_arg, video_start_arg = get_offset_args(metadata)

        audcodec = 'wav'
        vidcodec = 'mp4'

        videoonly_path = osp.join(temp_dir, f'videoonly.{vidcodec}')
        audioonly_path = osp.join(temp_dir, f'audioonly.{audcodec}')
        
        outvideo_path = ""
        outaudio_path = ""
        
        # separate out video
        if 'video' in mediatype:
            print('Separating video from input')
            video_cmd = f'ffmpeg -y -i {inpath} -an {videoonly_path}' 
            error = subprocess.call(video_cmd, shell=True)
            if error:
                raise Exception('Error in separating video out of input file')
            outvideo_path = videoonly_path
        # separate out audio
        if 'audio' in mediatype:
            print('Separating audio from input')
            audio_cmd = f'ffmpeg -y -i {inpath} -vn {audioonly_path}'
            error = subprocess.call(audio_cmd, shell=True)
            if error:
                raise Exception('Error in separating audio out of input file')
            outaudio_path = audioonly_path
            
        if 'video' in mediatype and 'video' in anonymize:
            # use face swapper if --facepath is provided
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
                fsgan_cmd = f'python3 {swappy_path} {facepath} -t {videoonly_path} -o {fsgan_outpath} --seg_remove_mouth --encoder_codec mp4v {device_flag}'
                error = os.system(fsgan_cmd)
                if error:
                    raise Exception(f'unable to swap faces. Check fsgan. error code: {error}')
                
                outvideo_path = fsgan_outpath
            else:    
                # use face hider if --facepath is not provided
                print("Hiding the face, as the facepath argument was empty")
                mtcnn_outpath = osp.join(temp_dir, "hidden_face.mp4")
                hide_face_py_path = osp.join(osp.dirname(__file__), 'hide_face_robust.py')
                assert osp.exists(hide_face_py_path), f"file not found: {hide_face_py_path}"
                mtcnn_cmd = f"python3 {hide_face_py_path} --inpath {videoonly_path} --outpath {mtcnn_outpath}"
                error = os.system(mtcnn_cmd)
                if error:
                    raise Exception(f"unable to run face hider. Check hide_face_robust.py. error code: {error}")
                
                outvideo_path = mtcnn_outpath

        if 'audio' in mediatype and 'audio' in anonymize:
            # save the intermediate audio files as .wav
            tmpaudiopath1 = osp.join(temp_dir, f'aud1.{audcodec}')  
            
            # transform the audio
            audio_py_path = osp.join(osp.dirname(__file__), 'audio.py')
            assert osp.exists(audio_py_path), f"file not found: {audio_py_path}"
            audio_cmd = f"python3 {audio_py_path} --inpath {audioonly_path} --outpath {tmpaudiopath1} --tr pitch --pitch_n_semitones {args.pitch}"
            print('Anonymizing audio...')
            error = os.system(audio_cmd)
            if error:
                raise Exception(f"unable to change audio. Check audio.py. error code: {error}")  
            outaudio_path = tmpaudiopath1
        
        if mediatype == 'audio':
            print('Creating final audio file')
            cmd = f'ffmpeg -y -i {outaudio_path} {outpath}'
            error = subprocess.call(cmd, shell=True)
            if error:
                raise Exception('Error in converting intermediate audio to final output audio')
            
        elif mediatype == 'video':
            print('Creating final video file')
            cmd = f'ffmpeg -y -i {outvideo_path} {outpath}'
            error = subprocess.call(cmd, shell=True)
            if error:
                raise Exception('Error in converting intermediate video to final output video')
            
        elif mediatype == 'audiovideo':
            print('Combining audio and video to form a single output file')
            cmd = f'ffmpeg -y {video_start_arg} -i {outvideo_path} {audio_start_arg} -i {outaudio_path} -vcodec copy -acodec aac -map 0:v:0 -map 1:a:0 {outpath}'
            error = subprocess.call(cmd, shell=True)
            if error:
                raise Exception('Error in combining intermediate audio and video files to form a final output video')
        else:
            raise Exception(f'unknown mediatype: {mediatype}')
    finally:
            
        # remove the temporary files
        print('removing temporary files/folders')
        shutil.rmtree(temp_dir)
        print('video saved at:', outpath)