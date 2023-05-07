import requests
from transformers import Blip2Processor, Blip2ForConditionalGeneration
import torch
from video import VideoSeparator
import cv2
import numpy as np
from PIL import Image, ImageDraw
import matplotlib.pyplot as plt
from tqdm import tqdm 


from tool import *

device = "cuda" if torch.cuda.is_available() else "cpu"




vs = VideoSeparator( load_data_path = 'data.pkl', path = "/home/huimingsun/Desktop/RESEARCH_PROJECT/NGP/data/video.mp4")
vs.read_video()
frame_list = vs.frames
audio_text_result = vs.pack_data['audio_text_result']


transformed_audio_result = []
for item in audio_text_result:
    transformed_audio_result.append([[int(item['start']), int(item['end'])], item['speaker'], item['text']])


processor = Blip2Processor.from_pretrained("Salesforce/blip2-opt-6.7b")
model = Blip2ForConditionalGeneration.from_pretrained(
    "Salesforce/blip2-opt-6.7b", torch_dtype=torch.float16
).to(device)


resized_video = [cv2.resize(frame, (256, 256)) for frame in vs.frames]
resized_video = [Image.fromarray(frame) for frame in resized_video]

time_length = 8


frame_interval = time_length // 4
# set the desired size of the new image
new_size = (512, 512)

# create a new list to store the modified images
modified_video = []

# loop through the resized video frames and combine every 4 frames into a new image
for i in range(0, len(resized_video), time_length):
    # extract 4 frames
    # frames = resized_video[i:i+4]
    frames = [resized_video[i], resized_video[i+frame_interval], resized_video[i+2*frame_interval], resized_video[i+3*frame_interval]]

    # create a new canvas image
    canvas = Image.new('RGB', (new_size[0]*2, new_size[1]*2), (255, 255, 255))
    # paste the frames onto the canvas
    for j in range(4):
        x = (j % 2) * new_size[0]
        y = (j // 2) * new_size[1]
        # resize the frame to fit the canvas
        resized_frame = frames[j].resize(new_size)
        canvas.paste(resized_frame, (x, y))
    # add the modified image to the list
    draw = ImageDraw.Draw(canvas)
    draw.line((new_size[0], 0, new_size[0], new_size[1]*2), fill=(255, 255, 255), width=4)
    draw.line((0, new_size[1], new_size[0]*2, new_size[1]), fill=(255, 255, 255), width=4)
    canvas = cv2.resize(np.array(canvas), (512, 512))
    modified_video.append(canvas)





def multi_frame_Video_caption(time, modified_video,time_length):

    summary_text = f'The vision summary for frame {time} to {time+time_length} is: '
    time = time // time_length

    questions = [None,"Describe what happend in the image", "Describe the first scene.", "Describe the second scene.", "Describe the third scene.", "Describe the forth scene."]
    scene_list = ['overall scenes', 'overall scenes', 'The first scene', 'The second scene ', 'The third scene ', 'The fourth scene ']
    for index,question in enumerate(questions):

        prompt = f"Question:{question} Answer:" if question else None
        # prompt = None

        inputs = processor(images=modified_video[time], text=prompt, return_tensors="pt").to(device, torch.float16)

        generated_ids = model.generate(**inputs)
        generated_text = processor.batch_decode(generated_ids, skip_special_tokens=True)[0].strip()
        generated_text = generated_text.replace('scene', " " + str(scene_list[index])+' ')

        # for index_i, i in enumerate(['The first scene is', 'The second scene is', 'The third scene is', 'The fourth scene is']):
        #     index_i += 1
        summary_text += generated_text 
    return summary_text

def find_near_speech_text(speech_data, T, time_range, speaker_time = None):
    near_text_data = ' '

    for data in speech_data:
        start_time, end_time = data[0]
        speaker = data[1]
        text = data[2]

        if (
            start_time <= T <= end_time or
            T - time_range <= start_time <= T and T - time_range <= end_time <= T
        ):
            if speaker is None: speaker = 'UNKNOW'
            if speaker_time != None: 
                string = f' At frame {start_time} to {end_time}, {speaker} says: {text} '
            else:
                string = f' {speaker} says: {text} '
            near_text_data += string

    return str(near_text_data)




def combine_speech_video(time,modified_video,transformed_audio_result, time_length = 4):
    visual_text =multi_frame_Video_caption(time = time, modified_video = modified_video, time_length = time_length)
    speech_text = find_near_speech_text(transformed_audio_result, T = time, time_range= 20, speaker_time = None)
    
    all_text =   f' Frame {time} to {time+time_length}: visual_text: ' + visual_text + ' speech_text: '+ speech_text + f" Frame {time} to {time+time_length} text end. "
    visual_text =   f' Frame {time} to {time+time_length}: visual_text: ' + visual_text + f" Frame {time} to {time+time_length} text end. "
    speech_text =   f' Frame {time} to {time+time_length}:  speech_text: '  + speech_text + f" Frame {time} to {time+time_length} text end. "

    
    
    return all_text, visual_text, speech_text



all_speech_text = find_near_speech_text(transformed_audio_result, T = len(resized_video), time_range= len(resized_video), speaker_time = True)

all_summary_text = []
all_visual_text = []
# with get_openai_callback() as cb:
for index,time in enumerate(tqdm(range(0, len(resized_video), time_length))):
    all_text, visual_text, speech_text = combine_speech_video(time = time, modified_video = modified_video, transformed_audio_result = transformed_audio_result, time_length = time_length)
    all_summary_text.append(all_text)
    all_visual_text.append(visual_text)

        
#list to string 
continus_summary_text = ' '.join(all_summary_text)
continus_visual_text = ' '.join(all_visual_text)



numbers = num_tokens_from_string(all_speech_text,"cl100k_base" )
split_speech_file('data/speech_data', all_speech_text,( numbers//800 + 1))




numbers = num_tokens_from_string(continus_visual_text,"cl100k_base" )
split_video_file('data/vision_text', continus_visual_text,( numbers//1100 + 1))



numbers = num_tokens_from_string(continus_summary_text,"cl100k_base" )
            
split_video_file('data/summary_text', continus_summary_text,( numbers//1100 + 1))




