import cv2
import numpy as np
import uuid
import copy
from third_party.facenet_pytorch import MTCNN, InceptionResnetV1
import torch
from PIL import Image, ImageEnhance
import matplotlib.pyplot as plt
import gc
import random
class FaceRecognizer:
    def __init__(self, tolerance=[0.6,0.8], scale =  1, box_margin = 20, min_appearances = 3, max_encoding_length=15):
        self.tolerance = tolerance[0]
        self.second_tolerance = tolerance[1]
        self.max_encoding_length = max_encoding_length
        self.face_images = {}
        self.face_locations = {}
        self.face_encodings = {}
        self.face_names = []
        self.face_records = {}
        self.box_margin = box_margin
        self.current_image_name = None
        self.image_counter = 0
        self.min_appearances = min_appearances
        self.failed_encode_image = []
        self.scale = scale
        self.device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
        self.face_detector = MTCNN(
            image_size=160, margin=5, min_face_size=160,
            thresholds=[0.6, 0.7, 0.7], factor=0.709, post_process=True, keep_all=True,
            device='cuda')
        # self.embedding = InceptionResnetV1(pretrained='vggface2').eval().to(self.device)
        
        self.embedding = InceptionResnetV1(pretrained='casia-webface').eval().to(self.device)

    def load_image(self, image_path):
        image = cv2.imread(image_path)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        return image



    def get_face_locations(self,image, size = 512, scale=1/24):
        box, prob = self.face_detector.detect(Image.fromarray(image))
        box = box.astype(int)
        #box expand 1/24 of the image size 
        for i in range(len(box)):
            box[i][0] = box[i][0] - int(image.shape[0]*scale)
            box[i][1] = box[i][1] - int(image.shape[1]*scale)
            box[i][2] = box[i][2] + int(image.shape[0]*scale)
            box[i][3] = box[i][3] + int(image.shape[1]*scale)

        return box 


    def get_face_images(self, image, face_images_locations):
        face_images = []
        for face_location in face_images_locations:
            top, right, bottom, left = face_location
            bottom, left,top, right = min(bottom, top), min(left, right), max(bottom, top), max(left, right) 

            face_images.append(image[left:right, bottom:top])
        return face_images


    def get_face_encodings(self, face_images, batch_boxes):
        face_encodings = []
        for index, face_image in enumerate(copy.copy(face_images)):
            try:
                face_encodings.append( self.get_face_feature(face_image))
            except:
                print('failed to encode image')
                self.failed_encode_image.append(face_image)
                face_images.remove(face_image)
                batch_boxes.remove(batch_boxes[index])
        return face_encodings,face_images,batch_boxes


    def get_faces(self, image):
        pillow_image = Image.fromarray(image)
        enhancer = ImageEnhance.Brightness(pillow_image)

        pillow_image = enhancer.enhance(1.5)
        # pillow_image.save('test.jpg')
    
        x_aligned, prob, batch_boxes = self.face_detector(pillow_image, return_prob=True)
        if len(x_aligned) == 0 :
            return [],[]
        x_aligned = (x_aligned.permute(0,2,3,1).cpu())

        # x_aligned = (x_aligned.permute(0,2,3,1).cpu()+1)/2
        #convert to list on first dimension
        x_aligned = torch.chunk(x_aligned, x_aligned.shape[0], dim=0)
      
        #get index of prob  value > 0.9 in prob list
        index = [i for i, x in enumerate(prob) if x > 0.97]
        x_aligned = [x_aligned[i] for i in index]
        batch_boxes = [batch_boxes[i] for i in index]
        # add margin to the boxes
        batch_boxes = [[max(0,box[0]-self.box_margin),max(0,box[1]-self.box_margin),min(image.shape[1],box[2]+self.box_margin),min(image.shape[0],box[3]+self.box_margin)] for box in batch_boxes]
        
        # print('index',index)
        #keep the x_aligned with the same index 
        x_aligned = [x.squeeze(0).numpy() for x in x_aligned]
        # print('x_aligned',x_aligned[0].shape)
        return x_aligned, batch_boxes

    def get_face_feature(self,face_image):
        # face_image  = face_image
        #add brightness to the image
        # face_image 
        return self.embedding(torch.from_numpy(face_image).permute(2,0,1).unsqueeze(0).to(self.device).float()).squeeze(0).detach().cpu().numpy()

    def process_unknown_image(self, image):
        if type(image) == np.ndarray:
            unknown_image = np.array(image)
        elif type(image) == str:
            unknown_image = self.load_image(image)
        if self.scale != 1:
            unknown_image = cv2.resize(unknown_image, (unknown_image.shape[1]//self.scale, unknown_image.shape[0]//self.scale))
        unknown_face_images,batch_boxes = self.get_faces(unknown_image)

        #no face detected in the image
        if len(unknown_face_images) == 0 :

            self.image_counter += 1
            return None
        unknown_face_encodings, unknown_face_images, batch_boxes = self.get_face_encodings(unknown_face_images,batch_boxes)
        #asser that the number of face images and face encodings are the same
        assert len(unknown_face_images) == len(unknown_face_encodings), 'number of face images and face encodings are not the same'
        new_faces = False
        for iii, encoding in enumerate(unknown_face_encodings):
            mean_distances = []
            for index in self.face_encodings:
                similarities = [self.cosine_similarity(enc, encoding) for enc in self.face_encodings[index]]
                mean_similarity = np.mean(similarities)
                mean_distances.append(1 - mean_similarity)
            
            if mean_distances:
                min_distance = min(mean_distances)
                index = mean_distances.index(min_distance)
            else:
                min_distance = float("inf")
                index = len(self.face_encodings)

            if min_distance <= self.tolerance:
                if len(self.face_encodings[index]) <= self.max_encoding_length:
                    self.face_encodings[index].append(encoding)
                self.face_images[index].append(unknown_face_images[iii])

            else:

                new_faces = True
                new_index = len(self.face_encodings)
                self.face_encodings[new_index] = [encoding]
                self.face_images[new_index] = []

                self.face_images[new_index].append(unknown_face_images[iii])

                index = new_index


            if index in self.face_records:
                # print('old',index)
                self.face_records[index].append(self.image_counter)
                self.face_locations[index].append((self.image_counter,batch_boxes[iii]))

            else:

                self.face_records[index] = [self.image_counter]
                self.face_locations[index] = []
                self.face_locations[index].append((self.image_counter,batch_boxes[iii]))

        self.image_counter += 1
        return new_faces
        
    def cosine_similarity(self,a, b):
        return np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b))

    def process_image(self, image):
        _ = self.process_unknown_image(image)

    def is_new_person(self, image):
        return self.process_unknown_image(image)
    
    def merge_lists_with_same_person(self, tolerance=None, length=None):
        
        print('Number of people before remix: ', len(self.face_encodings))
        if length is not None:
            #sample the face_encodings
            self.face_encodings = {k: random.sample(v,min(length,len(v))) for k, v in self.face_encodings.items()}
            

        if tolerance is not None:
            self.second_tolerance = tolerance

        merged = True
        while merged:
            merged = False

            # Sort keys based on the length of face_encodings values
            keys = sorted(self.face_encodings, key=lambda x: len(self.face_encodings[x]), reverse=True)

            merged_keys = set()

            for i in keys:
                if i in merged_keys:
                    continue

                # Sort remaining keys based on the length of face_encodings values
                remaining_keys = sorted([k for k in keys if k != i and k not in merged_keys],
                                        key=lambda x: len(self.face_encodings[x]))
                mean_distances_value = []
                list_j  = []
                for j in remaining_keys:
                    # Calculate mean distances between embeddings
                    mean_distances = [1 - np.mean([self.cosine_similarity(enc_i, enc_j)
                                                for enc_j in self.face_encodings[j]])
                                    for enc_i in self.face_encodings[i]]

                    mean_distance = np.mean(mean_distances)
                    mean_distances_value.append(mean_distance)
                    list_j.append(j)
                if list_j:  # Check if list_j is not empty   
                    # print(np.argmin(mean_distances),len(list_j)) 
                    min_j = list_j[np.argmin(mean_distances_value)]
                    min_value = np.min(mean_distances_value)
                    if min_value <= self.second_tolerance:
                        merged = True
                        merged_keys.add(i)

                        # Merge face data
                        self.face_encodings[i].extend(self.face_encodings[min_j])
                        self.face_images[i].extend(self.face_images[min_j])
                        self.face_records[i].extend(self.face_records[min_j])
                        self.face_locations[i].extend(self.face_locations[min_j])

                        del self.face_encodings[min_j]
                        del self.face_images[min_j]
                        del self.face_records[min_j]
                        del self.face_locations[min_j]
                        
                        keys = sorted(self.face_encodings, key=lambda x: len(self.face_encodings[x]), reverse=True)


                        break

        # Re-index the data after merging
        # self.face_encodings = {i: v for i, v in enumerate(self.face_encodings.values())}
        # self.face_images = {i: v for i, v in enumerate(self.face_images.values())}
        # self.face_records = {i: v for i, v in enumerate(self.face_records.values())}
        # self.face_locations = {i: v for i, v in enumerate(self.face_locations.values())}
        # Filter face data based on min_appearances
  
        # Re-index the data
        self.face_encodings = {i: v for i, v in enumerate(self.face_encodings.values())}
        self.face_images = {i: v for i, v in enumerate(self.face_images.values())}
        self.face_records = {i: v for i, v in enumerate(self.face_records.values())}
        self.face_locations = {i: v for i, v in enumerate(self.face_locations.values())}

        self.face_encodings = {k: v for k, v in self.face_encodings.items() if len(v) > self.min_appearances}
        self.face_records = {k: v for k, v in self.face_records.items() if len(v) > self.min_appearances}
        self.face_locations = {k: v for k, v in self.face_locations.items() if len(v) > self.min_appearances}
        self.face_images = {k: v for k, v in self.face_images.items() if len(v) > self.min_appearances}
        
        print('Number of people after remix: ', len(self.face_encodings))
        


    def draw_face(self, show_number = 10):

        for i in self.face_images.keys():
            for image_list in self.face_images[i]:
                for image in image_list[:show_number]:
                    plt.imshow((image+1)/2)
                    plt.title("Person "+ str(i) +" appears "+ str( len(self.face_records[i])) + ' times')
                    plt.show()

    def face_info(self):
        for i in self.face_images.keys():
            print("Person "+ str(i) +" appears "+ str( len(self.face_records[i])) + ' times')

    def pack_data(self):
        data = {}
        # data['face_encodings'] = self.face_encodings
        # data['face_images'] = self.face_images
        data['face_records'] = self.face_records
        data['face_locations'] = self.face_locations
        return data

    def clean_data(self):
        self.face_encodings = {}
        self.face_images = {}
        self.face_records = {}
        self.face_locations = {}
    def __del__(self):
        del self.face_detector
        del self.embedding
        torch.cuda.empty_cache()
        gc.collect()    

# from video import VideoSeparator

# clip = VideoSeparator('/home/huimingsun/Desktop/RESEARCH_PROJECT/NGP/data/video.mp4').read()
# # from face_detect import FaceRecognizer
# FR = FaceRecognizer(tolerance = [0.6,0.65],max_encoding_length =15)
# for item in clip:
#     # print(item.shape)
#     FR.process_unknown_image(item)
# FR.merge_lists_with_same_person()

