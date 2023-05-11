import whisperx
import whisper
from pyannote.audio.pipelines.speaker_verification import PretrainedSpeakerEmbedding
from pyannote.audio import Audio
from pyannote.core import Segment
from sklearn.cluster import AgglomerativeClustering
from sklearn.metrics import silhouette_score
from sklearn.preprocessing import StandardScaler
import torch
import numpy as np
import cv2
from third_party.facenet_pytorch import MTCNN, InceptionResnetV1
import copy
import gc

class VSDialization:
    def __init__(self, whisper_model = 'medium.en',  face_embedding_weight = 0.08):
        self.device = "cuda" if torch.cuda.is_available() else "cpu"

        self.whisper_model = whisper_model
        self.face_embedding = InceptionResnetV1(pretrained='casia-webface').eval().to(self.device)
        self.face_embedding_weight = face_embedding_weight
        self.speaker_embedding = PretrainedSpeakerEmbedding( 
            "speechbrain/spkrec-ecapa-voxceleb",
            device=torch.device("cuda" if torch.cuda.is_available() else "cpu"))
        self.segments = []


    # Add all the helper functions here as methods

    def _process_segments(self):
        # Code for processing segments with whisperX
        model = whisper.load_model(self.whisper_model)
        result = model.transcribe(self.audio_file,beam_size = 2)
        model_a, metadata = whisperx.load_align_model(language_code=result["language"], model_name = 'WAV2VEC2_ASR_LARGE_LV60K_960H', device=self.device )
        asr_result = [i for i in result["segments"] if '♪' not in i['text']]
        result_aligned = whisperx.align(asr_result, model_a, metadata, self.audio_file, self.device)

        # Filter out the repeated text
        repeat_text = None
        list_needs_to_be_removed = []
        filter_result = copy.deepcopy(result_aligned['segments'])
        for index,item in enumerate(result_aligned['segments']):
            if item['text'] == repeat_text:
                list_needs_to_be_removed.append(index)
            else:
                repeat_text = item['text']
        filter_result = [i for j, i in enumerate(filter_result) if j not in list_needs_to_be_removed]


        # Convert back to original openai format
        for segment_chunk in filter_result:
            chunk = {}
            chunk["start"] = segment_chunk["start"]
            chunk["end"] = segment_chunk["end"]
            chunk["text"] = segment_chunk["text"]
                
            self.segments.append(chunk)

        del model


        for i in range(len(self.segments)):
            value_list = [i for i in  range(int(self.segments[i]['start']), int(self.segments[i]['end'])+1)]
            result = self._get_keys_with_appear_times(self.packed_data['face_records'], value_list )

            person_app_appframe = self._people_sorted_by_appearances(result)
            self.segments[i]['likely_face'] = [i[0] for i in person_app_appframe]
            self.segments[i]['likely_face_appear_times'] = [i[1] for i in person_app_appframe]
            self.segments[i]['likely_face_appear_last'] = [i[2] for i in person_app_appframe]

        
    def _get_keys_from_value_list(self, data, value_list):
        matched_keys = []
        for key, value in data.items():
            if any(v in value_list for v in value):
                matched_keys.append(key)
                
        return matched_keys
    def _get_keys_with_appear_times(self, data, value_list):
        matched_keys = self._get_keys_from_value_list(data, value_list)
        appear_times = [(key, sum(v in value for v in value_list), list(set(value_list).intersection(set(value))) ) for key, value in data.items() if key in matched_keys]
        return appear_times
    
    
    
    def _people_sorted_by_appearances(self, input_list):
        """
        Get a list of people sorted by most appearances and last appearance frame in descending order.
        Conditions:
            - Same appearances, sort by the bigger frame.
            - Max appearances, sort by the bigger frame among the most appearances person.
        """

        sorted_list = []

        # Create a tuple for each person containing their name, last appearance frame, and number of appearances
        for person, appearances, value in input_list:
            last_appearance_frame = value[-1]
            sorted_list.append((person, appearances, last_appearance_frame ))

        # Sort the list based on appearances (descending) and last appearance frame (descending)
        sorted_list.sort(key=lambda x: (x[2], x[1]), reverse=True)

        return sorted_list

    def _find_human_face(self, desired_key, lst):
        for key, arr in lst:
            if key == desired_key:
                return arr
        assert False, "key not found"

    def _get_face_embedding(self, box, frame):
        image = cv2.resize(frame, (frame.shape[1]//2, frame.shape[0]//2))
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        x1, y1, x2, y2 = map(int, box)
        #min value is 0
        x1, y1, x2, y2 = max(0,x1), max(0,y1), max(0,x2), max(0,y2)
        roi = image[y1:y2, x1:x2]
        # print(roi.shape)
        roi = cv2.resize(roi, (128, 128))
        # plt.imshow(roi)
        # plt.show()
        return  self.face_embedding(torch.from_numpy(roi).permute(2,0,1).float().unsqueeze(0).to(self.device)).detach().cpu().numpy()[0]



    def _find_best_speaker_to_face_mapping(self, face_data):
        face_speaker_mapping = {}
        face_data_copy = {k: v[:] for k, v in face_data.items()}

        while face_data_copy:
            # Find the face with the biggest count time for each speaker
            max_face_count = -1
            max_face = None
            max_speaker = None

            for face, speaker_list in face_data_copy.items():
                if speaker_list:
                    speaker, count = speaker_list[0]
                    if count > max_face_count:
                        max_face_count = count
                        max_face = face
                        max_speaker = speaker

            # If no face with the biggest count time is found, break the loop
            if max_face is None:
                break

            # Link the face to the speaker and remove the speaker from all other face items
            face_speaker_mapping[max_face] = max_speaker

            for face, speaker_list in face_data_copy.items():
                face_data_copy[face] = [(speaker, count) for (speaker, count) in speaker_list if speaker != max_speaker]

            # Remove the face from the face_data_copy
            del face_data_copy[max_face]

        return face_speaker_mapping



    def _process_embeddings(self):
        # Create face embedding                
        face_embedding_list = []
        for seg in self.segments:
            people_most_like = seg['likely_face']
            people_last_appearance = seg['likely_face_appear_last']

            
            if len(people_most_like)!=0:
                person_most_like = people_most_like[0]
                person_last_appearance = people_last_appearance[0]
                try:
                    box = self._find_human_face(person_last_appearance,self.packed_data['face_locations'][person_most_like])
                    # if len(box)==0:
                    #     embedding = [0]*512
                    #     continue
                    box = [int(i) for i in box]
                    seg['most_like_box'] = box 
                    embedding = self._get_face_embedding(box, self.frames[person_last_appearance])
                # if embedding is None:
                except:
                    embedding = [0]*512
           
            else:
                embedding = [0]*512
            face_embedding_list.append(embedding)
            
        F_embeddings = np.array(face_embedding_list)
        
        
        # Create speaker embedding
        def segment_embedding(segment):
            audio = Audio()
            start = segment["start"]
            end = segment["end"]
            clip = Segment(start, end)
            second_waveform, sample_rate  = audio.crop(self.audio_file, clip)

            return self.speaker_embedding(second_waveform[None])


        S_embeddings = np.zeros(shape=(len(self.segments), 192))
        for i, segment in enumerate(self.segments):
            S_embeddings[i] = segment_embedding(segment)
        S_embeddings = np.nan_to_num(S_embeddings)
        
      
                
                
        # Normalize the embeddings
                        
        # Initialize MinMaxScaler
        scaler = StandardScaler()
        zero_rows = np.where(~F_embeddings.any(axis=1))[0]
        # Normalize both embeddings
        speech_embeddings_normalized = scaler.fit_transform(S_embeddings)
        face_embeddings_normalized = scaler.fit_transform(F_embeddings)

        non_zero_rows = np.where(F_embeddings.any(axis=1))[0]
        mean_non_zero_face_embedding = np.mean(F_embeddings[non_zero_rows], axis=0)

        #zero rows in face embeddings are replaced with 1 
        face_embeddings_normalized[zero_rows] = 1
        face_embeddings_normalized = face_embeddings_normalized * self.face_embedding_weight

        # Concatenate the embeddings
        self.combined_embeddings = np.concatenate((speech_embeddings_normalized, face_embeddings_normalized), axis=1)

    def _combine_embedding_cluster(self, num_speakers=0):

        if num_speakers == 0:
        # Find the best number of speakers
            score_num_speakers = {}
            print(f"Finding the best number of speakers from 2 to {len(self.packed_data['face_records'].keys())*2}")
            for num_speakers in range(2,  len(self.packed_data['face_records'].keys())*2):
                clustering = AgglomerativeClustering(num_speakers).fit(self.combined_embeddings)
                score = silhouette_score(self.combined_embeddings, clustering.labels_, metric='euclidean')
                score_num_speakers[num_speakers] = score
            best_num_speaker = max(score_num_speakers, key=lambda x:score_num_speakers[x])
            print(f"The best number of speakers: {best_num_speaker} with {score_num_speakers[best_num_speaker]} score", score)
        else:
            best_num_speaker = num_speakers
            
        # Assign speaker label   
        clustering = AgglomerativeClustering(n_clusters = int(best_num_speaker)).fit(self.combined_embeddings)
        labels = clustering.labels_
        for i in range(len(self.segments)):
            self.segments[i]["speaker"] = 'SPEAKER ' + str(labels[i] + 1)




    def analyze(self, packed_data, frames, audio_file ='audio.wav', num_speakers = 0):
        self.packed_data = packed_data
        self.frames = frames
        self.audio_file = audio_file
        self._process_segments()
        self._process_embeddings()
        self._combine_embedding_cluster(num_speakers)
        face_speaker_mapping = self._process_data_to_face_map(self.segments)
        return self.segments, face_speaker_mapping

    def re_analyze(self,face_embedding_weight, num_speakers = 0):
        self.face_embedding_weight = face_embedding_weight
        self._combine_embedding_cluster(num_speakers)
        face_speaker_mapping = self._process_data_to_face_map(self.segments)
        return self.segments, face_speaker_mapping
    

    def __del__(self):
        del self.face_embedding
        del self.speaker_embedding
        torch.cuda.empty_cache()
        gc.collect()    
        
        
    def _process_data_to_face_map(self, input_data, interval=20, min_count=2):
        summary_data = {}

        # Aggregate the data by speaker and face within the specified time interval
        for entry in input_data:
            start_time = entry['start']
            speaker = entry['speaker']
            likely_faces = entry['likely_face']
            likely_face_appear_times = entry['likely_face_appear_times']

            if speaker not in summary_data:
                summary_data[speaker] = {}

            for face, face_appear_times in zip(likely_faces, likely_face_appear_times):
                if face not in summary_data[speaker]:
                    summary_data[speaker][face] = {'count': face_appear_times, 'timestamps': [start_time]}
                else:
                    summary_data[speaker][face]['count'] += face_appear_times
                    summary_data[speaker][face]['timestamps'].append(start_time)

        self.face_votes = {}

        # Filter out isolated faces that appear only once within the specified time interval
        # and count votes for each face per speaker
        isolated_faces = []
        for speaker, face_counts in summary_data.items():
            face_counts_copy = face_counts.copy()
            for face, face_data in face_counts_copy.items():
                timestamps = sorted(face_data['timestamps'])
                isolated_timestamps = []
                for i in range(1, len(timestamps)):
                    left_interval_count = sum(1 for t in timestamps[:i] if timestamps[i] - t <= interval)
                    right_interval_count = sum(1 for t in timestamps[i+1:] if t - timestamps[i] <= interval)
                    if left_interval_count + right_interval_count <=  min_count:
                        isolated_timestamps.append(timestamps[i])

                if len(isolated_timestamps) >= 1:
                    isolated_faces.append(face)

        # Call the update_input_data function with input_data and isolated_faces as arguments
        input_data = self._update_input_data(input_data, isolated_faces)

        summary_data = {}

        # Aggregate the data by speaker and face within the specified time interval
        for entry in input_data:
            start_time = entry['start']
            speaker = entry['speaker']
            likely_faces = entry['likely_face']
            likely_face_appear_times = entry['likely_face_appear_times']

            if speaker not in summary_data:
                summary_data[speaker] = {}

            for face, face_appear_times in zip(likely_faces, likely_face_appear_times):
                if face not in summary_data[speaker]:
                    summary_data[speaker][face] = {'count': face_appear_times, 'timestamps': [start_time]}
                else:
                    summary_data[speaker][face]['count'] += face_appear_times
                    summary_data[speaker][face]['timestamps'].append(start_time)


        self.face_votes = {}

        # Count votes for each face per speaker
        for speaker, face_counts in summary_data.items():
            for face, face_data in face_counts.items():
                if face not in self.face_votes:
                    self.face_votes[face] = {}
                if speaker not in self.face_votes[face]:
                    self.face_votes[face][speaker] = 0
                self.face_votes[face][speaker] += face_data['count']
     

        face_speaker_mapping = {}

        for face, speaker_votes in self.face_votes.items():
            best_speaker, max_count, second_best_speaker, second_max_count = self._find_best_speaker(speaker_votes)
            face_speaker_mapping[face] = (best_speaker, max_count, second_best_speaker, second_max_count)

        for face, (best_speaker, max_count, second_best_speaker, second_max_count) in face_speaker_mapping.items():
            for other_face, (other_best_speaker, other_max_count, other_second_best_speaker, other_second_max_count) in face_speaker_mapping.items():
                if face != other_face and best_speaker == other_best_speaker:
                    if max_count > other_max_count:
                        face_speaker_mapping[other_face] = (other_second_best_speaker, other_second_max_count, other_best_speaker, other_max_count)
                    else:
                        face_speaker_mapping[face] = (second_best_speaker, second_max_count, best_speaker, max_count)
                    break

        result = {face: (best_speaker, max_count) for face, (best_speaker, max_count, _, _) in face_speaker_mapping.items()}
        # result = {face: best_speaker for face, (best_speaker, max_count, _, _) in face_speaker_mapping.items()}

        result = {k: v[0] for k, v in result.items() if  v[1] >= 3}
        
        return result


    def _find_best_speaker(self, speaker_votes):
        max_count = 0
        second_max_count = 0
        best_speaker = None
        second_best_speaker = None
        for speaker, count in speaker_votes.items():
            if count > max_count:
                second_max_count = max_count
                second_best_speaker = best_speaker
                max_count = count
                best_speaker = speaker
            elif count > second_max_count:
                second_max_count = count
                second_best_speaker = speaker
        return best_speaker, max_count, second_best_speaker, second_max_count
    
        
    def _update_input_data(self, input_data, isolated_faces):
        new_input_data = []

        for entry in input_data:
            likely_faces = entry['likely_face']
            new_likely_faces = [face for face in likely_faces if face not in isolated_faces]
            if new_likely_faces:
                new_entry = entry.copy()
                new_entry['likely_face'] = new_likely_faces
                new_input_data.append(new_entry)
            else:
                new_input_data.append(entry)

        return new_input_data