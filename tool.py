import itertools
import os 
from PIL import Image
import io
import numpy as np
import cv2
import base64
import tiktoken
from llm_planner import llm_check_relevent_text
#os.remove dir text_dir

encoding = tiktoken.encoding_for_model("gpt-3.5-turbo")
def num_tokens_from_string(string: str, encoding_name: str) -> int:
    """Returns the number of tokens in a text string."""
    encoding = tiktoken.get_encoding(encoding_name)
    num_tokens = len(encoding.encode(string))
    return num_tokens

def convert_image(np_image):
    #bgr to rgb
    np_image = cv2.cvtColor(np_image, cv2.COLOR_BGR2RGB)
    retval, buffer = cv2.imencode(".jpg", np_image)

    image_data = base64.b64encode(buffer).decode("utf-8")

    # Decode byte string as UTF-8 and return
    return image_data

def unite_matched_times( video_dict, audio_dict, matched):
    united_times = {}

    for audio_key, video_key in matched.items():
        united_times[video_key] = video_dict[video_key] + audio_dict[audio_key]

    return united_times



def pad_video_dict( video_dict, padding =5):
    padded_video_dict = {}

    for key, times in video_dict.items():
        padded_times = []

        for idx, t in enumerate(times):
            # Check if the current frame is within the threshold distance of the previous or next frame
            if idx > 0 and t - times[idx - 1] <= padding or idx < len(times) - 1 and times[idx + 1] - t <= padding:
                for i in range(padding):
                    if t - i >= 0 and t - i not in padded_times:
                        padded_times.append(t - i)

        padded_video_dict[key] = sorted(padded_times)
    # delet the empty value in the dict
    for key in list(padded_video_dict.keys()):
        if len(padded_video_dict[key]) == 0:
            del padded_video_dict[key]
        
    return padded_video_dict

def get_continuing_intervals(dictionary):
    intervals = {}
    for key, times in dictionary.items():
        key_intervals = []
        for k, g in itertools.groupby(enumerate(times), lambda x: x[0] - x[1]):
            interval = list(map(lambda x: x[1], g))
            if len(interval) > 1:
                key_intervals.append((interval[0], interval[-1]))
        if len(key_intervals) > 0:
            intervals[key] = key_intervals

    return intervals


import re
import os
import shutil

def split_video_file(path, content, num_files):
    try:
        shutil.rmtree(f"{path}")
    except:
        pass

    def split_text(text):
        pattern = r"(Frame \d+ to \d+:)"
        segments = re.split(pattern, text)
        cleaned_segments = []

        for i in range(1, len(segments), 2):
            cleaned_segments.append(segments[i] + segments[i+1])

        return cleaned_segments

    os.makedirs(f"{path}")

    segments = split_text(content)
    print( len(segments))

    # Calculate the number of segments to put in each chunk
    segments_per_chunk = (4097//int(np.mean([len(i) for i in segments])))
    # Split the segments into chunks
    chunks = [segments[i:i+segments_per_chunk] for i in range(0, len(segments), segments_per_chunk)]

    # Write each chunk to a separate file
    for i, chunk in enumerate(chunks):
        with open(f'{path}/{i+1}.txt', 'w') as f:
            f.write('\n'.join(chunk))


def split_speech_file(path, content, num_files):
    def split_text(text):
        segments = text.split("At frame")
        cleaned_segments = []

        for segment in segments:
            if segment.strip() != "":
                cleaned_segments.append("At frame " + segment.strip())

        return cleaned_segments

    shutil.rmtree(f"{path}", ignore_errors=True)
    os.makedirs(f"{path}")
    # Read the input file

    # Split the content into segments based on "At frame"
    segments = split_text(content)
    print( len(segments))

    # Calculate the number of segments to put in each chunk
    segments_per_chunk = (4097//int(np.mean([len(i) for i in segments])))


    # Split the segments into chunks
    chunks = [segments[i:i+segments_per_chunk] for i in range(0, len(segments), segments_per_chunk)]

    # Write each chunk to a separate file
    for i, chunk in enumerate(chunks):
        with open(f'{path}/{i+1}.txt', 'w') as f:
            f.write('\n'.join(chunk))

def write_text(path, text):
    with open(path, 'w') as file:
        for sentence in re.split('(?<=\.) (?=\d)', text):
            lines = re.split(r'(Frame \d+ to \d+  text end)', sentence)
            for i, line in enumerate(lines):
                if re.match(r'Frame \d+ to \d+ text end.', line):
                    file.write(line.strip() + ' ')
                    if i < len(lines) - 1:
                        file.write('\n \n')
                else:
                    file.write(line.strip() )
            if not re.search(r'Frame \d+ to \d+ text end.', sentence):
                file.write('\n')
        file.close()





# def frame_interval_filter(relavent_text,frame_distance = 12 ):
#     # Extract frame intervals
#     frame_intervals = re.findall(r'Frame (\d+) to (\d+):', relavent_text)

#     # Convert strings to integers
#     frame_intervals = [(int(start), int(end)) for start, end in frame_intervals]

#     # Sort frame intervals
#     frame_intervals.sort(key=lambda x: x[0])

#     updated_frame_intervals = []
#     to_be_decided = []
    
#     for i in range(len(frame_intervals) - 1):
#         interval1 = frame_intervals[i]
#         interval2 = frame_intervals[i + 1]

#         frame_diff = interval2[0] - interval1[1]

                
#         if frame_diff <= frame_distance:
        
#             padding_start = interval1[1] + 1
#             padding_end = interval2[0] - 1
#             interval1 = (interval1[0], padding_end)
#             interval2 = (padding_start, interval2[1])
#             updated_frame_intervals.append(interval1)

#             # frame_intervals[i] = interval1
#             # frame_intervals[i + 1] = interval2
#         else :
#             to_be_decided.append((interval1[1], interval2[0]))

#     # print("Updated Frame Intervals:", updated_frame_intervals)
#     # print("To Be Decided:", to_be_decided)
#     return frame_intervals, to_be_decided




def make_continuous_intervals(frame_intervals, padding=5):
    # Convert frame_intervals into a continuous list of frame numbers
    continuous_frames = []
    for interval in frame_intervals:
        continuous_frames.extend(range(interval[0], interval[1] + 1))

    # Sort the continuous frame numbers
    continuous_frames.sort()

    # Perform the padding
    padded_intervals = []
    current_interval = (continuous_frames[0], continuous_frames[0])
    for i in range(1, len(continuous_frames)):
        current_frame = continuous_frames[i]
        frame_diff = current_frame - current_interval[1]

        if frame_diff <= padding:
            current_interval = (current_interval[0], current_frame)
        else:
            padded_intervals.append(current_interval)
            current_interval = (current_frame, current_frame)

    padded_intervals.append(current_interval)

    return padded_intervals



def extract_number_ranges(relavent_text_list, end_index= None, length = 0):
    if end_index is None:
        end_index = len(relavent_text_list)
    joined_text = ' '.join(relavent_text_list[:end_index - length]) 
    pattern = r"Frame (\d+) to (\d+):"
    matches = re.findall(pattern, joined_text)
    related_frames_interval = []
    
    for i in matches:
        related_frames_interval.append((int(i[0]),int(i[1])))
        
    related_frames_interval = sorted(related_frames_interval, key=lambda x: x[0])
    return related_frames_interval         


def recursive_check(question, relavent_text, length, start, end):


    part_relavent_text = relavent_text[start:end]
    checked_result = llm_check_relevent_text(question, ' '.join(part_relavent_text))

    if len(checked_result) <= 1:
        return start
    
    elif len(checked_result) >= length // 2:
        return recursive_check(question, relavent_text, length // 2, end, end + length // 2 + 1)
    else:
        return recursive_check(question, relavent_text, length // 2, start - length // 2, start)




def frame_interval_filter(frame_intervals, frame_distance=12):
    # Sort frame intervals
    frame_intervals.sort(key=lambda x: x[0])

    merged_intervals = []
    to_be_decided = []

    if 0<len(frame_intervals) <=3:
        return frame_intervals, [] 
    elif len(frame_intervals) == 0:
        return [], []
    
    i = 0
    while i < len(frame_intervals) - 1:
        interval1 = frame_intervals[i]
        interval2 = frame_intervals[i + 1]

        frame_diff = interval2[0] - interval1[1]

        if frame_diff < frame_distance:
            # Merge intervals
            merged_interval = (interval1[0], interval2[1])
            merged_intervals.append(merged_interval)
            i += 2  # Skip the next interval as it is already merged
        else:
            to_be_decided.append(interval1)
            i += 1
    middle_intervals = merged_intervals
    middle_intervals.sort(key=lambda x: x[0])

    merged_intervals = []
    for interval in middle_intervals:
        if not merged_intervals or merged_intervals[-1][1] < interval[0]:
            # if the current interval doesn't overlap with the previous interval, add it to the merged list
            merged_intervals.append(interval)
        else:
            # if the current interval overlaps with the previous interval, merge them
            merged_intervals[-1] = (merged_intervals[-1][0], max(merged_intervals[-1][1], interval[1]))
    return merged_intervals, to_be_decided

def find_to_be_decided_indices(to_be_decided, dataset_text_list):
    indices = []
    # Iterate over each item in to_be_decided
    for item in to_be_decided:
        # Construct the search string using the item
        search_string = f'Frame {item[0]} to {item[1]}:'
        # Iterate over the elements in dataset_text_list
        for i, text in enumerate(dataset_text_list):
            # Check if the search string is present in the text
            if search_string in text:
                # If found, add the index to the list and break out of the inner loop
                indices.append(i)
                break
    return indices





def extend_small_interval(questions, dataset_text_list, frame_interval_list,length = 3):
    string_frame_interval_list = [f'{i[0]} to {i[1]}' for i in frame_interval_list]
    small_indexs = find_index(dataset_text_list, string_frame_interval_list)
    length += 1
    small_indexs_extend_status = []
    print(string_frame_interval_list)
    for check_index in small_indexs:
        small_indexs_extend_status.append(llm_doule_add_relevent_text(' '.join(questions), ' '.join(dataset_text_list[check_index:check_index+length])))
    # small_indexs_extend_status =[True, False]
    for index, status in enumerate(small_indexs_extend_status):
        
        if status:
            
            new_element = (frame_interval_list[index][0] ,frame_interval_list[index][1] +length)
            frame_interval_list.remove(frame_interval_list[index])
            # print(new_element)
            frame_interval_list.append(new_element)
    return frame_interval_list


def find_index(data_text_list, frame_interval_list):
    indexs = []
    for index,text in enumerate(data_text_list):
        for interval in frame_interval_list:
            if interval in text:
                indexs.append(index)
            if len(indexs) == len(frame_interval_list):
                return indexs      
            


def extend_small_interval(questions, dataset_text_list, frame_interval_list,length = 3):
    string_frame_interval_list = [f'{i[0]} to {i[1]}' for i in frame_interval_list]
    small_indexs = find_index(dataset_text_list, string_frame_interval_list)
    length += 1
    small_indexs_extend_status = []

    for check_index in small_indexs:
        small_indexs_extend_status.append(llm_doule_add_relevent_text(' '.join(questions), ' '.join(dataset_text_list[check_index:check_index+length])))
    # small_indexs_extend_status =[True, False]
    for index, status in enumerate(small_indexs_extend_status):
        
        if status:
            
            new_element = (frame_interval_list[index][0] ,frame_interval_list[index][1] +length)
            frame_interval_list.remove(frame_interval_list[index])
            # print(new_element)
            frame_interval_list.append(new_element)
    return frame_interval_list


format_intervals = lambda intervals :[{"start": start, "end": end} for start, end in intervals]
