import os
from pydub import AudioSegment
from pydub.silence import split_on_silence
import logging

def preprocess_audio(audio_file_path, mic: str):

    # 加载音频文件
    audio_file = AudioSegment.from_wav(audio_file_path)

    # 设置参数
    min_silence_len = 500 # 静默长度（毫秒）
    if mic == 'mic-1':
        silence_thresh = -38 # 静默阈值（分贝）
    elif mic == 'mic-2':
        silence_thresh = 20
    elif mic == 'mic-3':
        silence_thresh = 20
    else:
        silence_thresh = -38

    keep_silence = 100 # 切割后静默保留长度（毫秒）

    parts = []

    # 若切割后音频文件为空，则降低静默阈值，再次切割
    while len(parts) == 0:
        # 切割音频文件
        parts = split_on_silence(audio_file, min_silence_len=min_silence_len, silence_thresh=silence_thresh, keep_silence=keep_silence)
        silence_thresh -= 1    

    # 只保留第0段
    part = parts[0]

    # 若音頻長度大於250毫秒，則只保留最後五百毫秒
    if len(part) > 250:
        part = part[-250:]
    elif len(part) < 250:
        part = part + AudioSegment.silent(duration=250-len(part))

    # 保存新音频文件至當前目錄
    file_name = 'cleaned_' + os.path.basename(audio_file_path)
    part.export(os.path.join(os.path.dirname(audio_file_path), file_name), format='wav')


if __name__ == '__main__':

    logging.basicConfig(level=logging.INFO, format='%(asctime)s - [%(levelname)s] %(message)s', datefmt='%Y-%m-%d %H:%M:%S')
    datas_root_path = os.path.join(os.getcwd(), 'Pineapple_New_Test_Data')

    # preprocess_audio('D:\\vscodeProject\\pineapple-classifier\\training\\0001\\cam-2\\pine-side\\mic-3\\01.wav', 'mic-3')

    # for each folder in the training folder
    for folder in os.listdir(datas_root_path):
        
        # check if the folder is a directory
        if not os.path.isdir(os.path.join(datas_root_path, folder)):
            continue

        folder_path = os.path.join(datas_root_path, folder)

        audio_folders = [
            os.path.join(folder_path, 'cam-1/pine-bottom/mic-1'),
            #os.path.join(folder_path, 'cam-1/pine-bottom/mic-2'),
            os.path.join(folder_path, 'cam-1/pine-side/mic-1'),
            #os.path.join(folder_path, 'cam-1/pine-side/mic-2'),
            os.path.join(folder_path, 'cam-2/pine-bottom/mic-1'),
            #os.path.join(folder_path, 'cam-2/pine-bottom/mic-2'),
            #os.path.join(folder_path, 'cam-2/pine-bottom/mic-3'),
            os.path.join(folder_path, 'cam-2/pine-side/mic-1'),
            #os.path.join(folder_path, 'cam-2/pine-side/mic-2'),
            #os.path.join(folder_path, 'cam-2/pine-side/mic-3'),
        ]

        
        for audio_folder in audio_folders:
            for audio_file in os.listdir(audio_folder):
                if not audio_folder.endswith('mic-1') and audio_file.startswith('cleaned_'):
                    os.remove(os.path.join(audio_folder, audio_file))
                    continue
                if audio_file.startswith('cleaned_'):
                    continue
                if f'cleaned_{audio_file}' in os.listdir(audio_folder):
                    logging.info(f'Audio file: {audio_file} has been preprocessed, skipping...')
                    continue
                
                audio_file_path = os.path.join(audio_folder, audio_file)
                mic = os.path.basename(os.path.dirname(audio_folder))
                logging.info(f'Preprocessing audio file: {audio_file_path}...')
                preprocess_audio(audio_file_path, mic)
                logging.info('Done')
                