
# from video import VideoSeparator
import moviepy.editor as mp
from moviepy.editor import *
import cv2
from third_party.pyannote_whisper.utils import diarize_text
import numpy as np 
from pyannote.audio import Pipeline
from sklearn.metrics.pairwise import cosine_similarity
from face_detect import FaceRecognizer
import networkx as nx
import pickle
from tool import *
from speechbrain.pretrained import WaveformEnhancement
import torch
from faster_whisper import WhisperModel
from speechbrain.pretrained import SpectralMaskEnhancement
import gc

import matplotlib.pyplot as plt
import matplotlib.patches as patches
import numpy as np
import time 
import shutil
from VideoSpeakerDialization import VSDialization


device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
run_opts={"device":device}

def get_data_attributes( instance):
    attributes = {}
    for attr, value in instance.__dict__.items():
        if not callable(value) and not attr.startswith('__'):
            attributes[attr] = value
    return attributes





def draw_faces_with_boxes(face, face_inframes, boxex, numpy_frames):
    show_length = min(len(numpy_frames), 10) + 1
    fig, ax = plt.subplots(1, show_length, figsize=(15, 5))
    
    for i in range(show_length):
        left, top, right, bottom  = boxex[i]
        left, top, right, bottom = int(left), int(top), int(right), int(bottom)
        numpy_frames[i] = cv2.rectangle(numpy_frames[i], (left, top), (right, bottom), (0, 255, 0), 2)
        ax[i].imshow(np.asarray(numpy_frames[i]))
        ax[i].set_title(f"{face} - F {face_inframes[i]}")
        ax[i].axis("off")
    plt.show()
    time.sleep(0.5)
    plt.close()

        
class VideoSeparator:
    def __init__(self, path = None, diarize_with_audiio = True, tolerance =  [0.65, 0.70, 0.7],max_encoding_length = 10,  main_person_threshold = 10, load_data_path =None, start_multi_processing=True):
        self.path = path
        self.size = 512    
        self.num_processes = 8
        self.max_encoding_length = max_encoding_length
        self.main_person_threshold = main_person_threshold 
        self.start_multi_processing = start_multi_processing
        self.tolerance = tolerance
        self.audio_text_result = None
        self.speaker_appear_time = None
        self.speaker_text = None
        self.main_persons = None

        self.diarize_with_audiio = diarize_with_audiio 
        
        
        if load_data_path is not None:
            self.load_data(load_data_path)
        else:
            self.FR = FaceRecognizer(tolerance = self.tolerance, scale = 1, max_encoding_length =max_encoding_length)

 

    def save_data(self, file_path):
        # data = get_data_attributes(self)
   

        with open(file_path, 'wb') as f:
            pickle.dump(self.pack_data, f)

    def load_data(self, file_path):
        with open(file_path, 'rb') as f:
            self.pack_data = pickle.load(f)


    def read_video(self, path = None ):
        if path is not None:
            self.path = path
        self.video = VideoFileClip(self.path)

        self.frames = []
        for index, frame in enumerate(self.video.iter_frames()):
            if index % self.video.fps == 0:
                self.frames.append(frame)  

        return self.frames

    def read_audio(self, path = None):
        if path is not None:
            self.path = path
        self.audio = VideoFileClip(self.path).audio
        self.audio.write_audiofile("audio.wav", fps = 16000)
        return self.audio


    def __resize_frame__(self, frame):
        return cv2.resize(frame, (self.size, self.size))

    def __create_histogram__(self,time_list, bins=10, max_time=1000):
        histogram, _ = np.histogram(time_list, bins=bins, range=(0, max_time))
        return histogram / np.sum(histogram)

    def whisper_diarize(self, audio_file = None):
     
     
        Speaker_Dialization = VSDialization()
        assert len(self.frames) > 0, "Please read video first"
        segments, face2speaker = Speaker_Dialization.analyze(self.pack_data,self.frames )
        del Speaker_Dialization
        self.pack_data['audio_text_result'] = segments
        self.pack_data['face2speaker'] = face2speaker
        return segments
        # self.pack_data['speaker_appear_time'] = self.speaker_appear_time if self.speaker_appear_time is not None else None           
        # self.pack_data['speaker_text'] = self.speaker_text if self.speaker_text is not None else None
        
        
        # self.pipeline = Pipeline.from_pretrained("pyannote/speaker-diarization@2.1",
        #                                     use_auth_token="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")
        # model = WhisperModel("medium.en", device='cuda', compute_type="float32")

        # self.asr_result, _ = model.transcribe("audio.wav",beam_size = 1)
        # diarization_result = self.pipeline("audio.wav" )
        # # asr_result = list(asr_result)
        # self.audio_text_result = diarize_text(self.asr_result, diarization_result)
        # self.speaker_number = np.unique([ spk for seg, spk, sent in self.audio_text_result if spk != None ])

        # self.speaker_appear_time = {}
        # for spk in self.speaker_number:
        #     self.speaker_appear_time[spk] = []

        # for seg, spk, sent in self.audio_text_result:
        #     if spk != None:
        #         self.speaker_appear_time[spk].extend([i for i in range(int(seg.start), int(seg.end)+1)])

        # self.speaker_text = {}
        # for spk in self.speaker_number:
        #     self.speaker_text[spk] = []

        # for seg, spk, sent in self.audio_text_result:
        #     if spk != None:
        #         self.speaker_text[spk].append(sent)
        # del model 
        # torch.cuda.empty_cache()
        # gc.collect()
        
        # return self.audio_text_result,self.speaker_appear_time, self.speaker_text



    
    def process_video(self, path = None):
        if path is not None:
            self.path = path
        clip = self.read_video(self.path)
        for item in clip:
            # print(item.shape)
             self.FR.process_unknown_image(item)
        self.FR.merge_lists_with_same_person()
        # self.FR.merge_lists_with_same_person(tolerance=self.tolerance[2], length= 5)
        # self.FR.merge_lists_with_same_person(tolerance=self.tolerance[2])

        self.pack_data = self.FR.pack_data()
        del self.FR
        gc.collect()

        self.main_persons = []

        for i in  self.pack_data['face_records'].keys():
            if len( self.pack_data['face_records'][i])>self.main_person_threshold:
                # print("Person "+ str(i) +" appears "+ str( len(video_dict[i])) + ' times')
                self.main_persons.append(i)

        
        return  self.pack_data['face_records']

    def process_audio(self, path = None):
        self.read_audio(path)
        return self.whisper_diarize()

    def process_video_and_audio(self, video_path = None, audio_path = None):
        self.process_video(video_path)

        self.process_audio(audio_path)

    def face2folder(self, path = None, imshow = False):
        if path == None:
            path = 'face_images'
        count = 0
        shutil.rmtree(path, ignore_errors=True)
        for face in self.pack_data['face_records'].keys():
            face_inframes = self.pack_data['face_records'][face]
            boxex = [i[1] for i in self.pack_data['face_locations'][face] if i[0] in face_inframes]
            
            # Use list comprehension to get the frames
            numpy_frames = [self.frames[index] for index in face_inframes]
            
            # Resize the image in numpy_frames with a 1/2 scale

            numpy_frames = [cv2.cvtColor(i, cv2.COLOR_BGR2RGB) for i in numpy_frames]
            if imshow:
                draw_faces_with_boxes(face, face_inframes, boxex, numpy_frames)
            # Save the boxex rectangle numpy_frames to the folder
            for i in range(len(boxex)):
                top, bottom = max(int(boxex[i][1]) - 50, 0 ), max(int(boxex[i][3]) + 50, 0)
                left, right = max(int(boxex[i][0]) - 50, 0), max(int(boxex[i][2]) +50, 0)
                roi = numpy_frames[i][top:bottom, left:right]
                os.makedirs(f"{path}/face_{face}", exist_ok=True)
                cv2.imwrite(f"{path}/face_{face}/{count}.jpg", roi)
                count += 1



