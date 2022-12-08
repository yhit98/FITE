
import os
import json
from transformers import AutoProcessor, AutoModel

processor = AutoProcessor.from_pretrained("openai/clip-vit-base-patch16")

model = AutoModel.from_pretrained("openai/clip-vit-base-patch16")

from PIL import Image
import requests

from transformers import CLIPProcessor, CLIPModel

import os
import json

twitter15_image_path = "../../../data/IJCAI2019_data/twitter2015_images/"
twitter17_image_path = "../../../data/IJCAI2019_data/twitter2017_images/"

twitter15_face_path = "./face_attributes/tw15_"
twitter17_face_path = "./face_attributes/tw17_"

twitter15_save_path = "../face_descriptions/twitter15_face_discription_clip16.json"
twitter17_save_path = "../face_descriptions/twitter17_face_discription_clip16.json"

tw15_path = "../../../data/IJCAI2019_data/twitter2015/"
tw17_path = "../../../data/IJCAI2019_data/twitter2017/"

twitter15_img = os.listdir(twitter15_image_path)
twitter17_img = os.listdir(twitter17_image_path)

def generate_face_descriptions(path, mode, image_path, save_path, face_path):
    descriptions = {}
    # get face attributes
    face_path = face_path+mode+"_face_attributes.json"
    with open(face_path, "r") as f:
        faces = json.load(f)

    fin = open(path+mode+".txt", 'r', encoding='utf-8', newline='\n', errors='ignore')
    lines = fin.readlines()
    fin.close()
    image_names = []
    aspects = []
    face_nums = []
    for i in range(0, len(lines), 4): 
        text = lines[i].strip()
        aspect = lines[i+1].strip() 
        polarity = lines[i+2].strip()
        image_name = lines[i+3].strip()
        aspects.append(aspect)
        image_names.append(image_name)
        face = faces[image_name]
        face_nums.append(len(face)) 

    for image_name in image_names:
        tmp_image_path = image_path + image_name
        image = Image.open(tmp_image_path)
        face = faces[image_name]
        image_aspects = []
        for j in range(len(image_names)):
            if image_names[j] == image_name:
                image_aspects.append(aspects[j]) 
        head_sentence = "There are " + str(len(face)) + " people in the picture. "
        if len(face) == 0:
            descriptions[image_name] = head_sentence
            continue 

        tmp_fds = []
        for one_face in face:
            age = one_face["age"]
            gender = one_face['gender']
            race = one_face['dominant_race']
            emotion = one_face['dominant_emotion']
            
            if one_face['race'][race] >= 40:
                tmp_sentence = "a " + str(age) + "-year-old " + race + " " + gender 
            else:
                tmp_sentence = "a " + str(age) + "-year-old " + gender 
            if one_face['emotion'][emotion] >= 40:
                tmp_sentence = tmp_sentence + " with a " +  emotion + " expression. "
            else:
                tmp_sentence = tmp_sentence + ". "
            tmp_fds.append(tmp_sentence)

        tmp_aspect_clip = [-1]*len(image_aspects)
        for t in range(len(image_aspects)):
            clip_senntences = []
            for tmp_df in tmp_fds:
                tmp_s = image_aspects[t] + ", " + tmp_df
                clip_senntences.append(tmp_s)
            try:
                inputs = processor(text=clip_senntences, images=image, return_tensors="pt", padding=True)
                outputs = model(**inputs)
                logits_per_image = outputs.logits_per_image # this is the image-text similarity score
                probs = logits_per_image.softmax(dim=1) # we can take the softmax to get the label probabilities
                print(probs[0])
                for index_i in range(len(probs[0])):
                    if probs[0][index_i] == max(probs[0]):
                        tmp_aspect_clip[t] = index_i
                        break
            except:
                continue
        for _df in range(len(tmp_fds)):
            aspects_for_df = []
            for ac in range(len(tmp_aspect_clip)):
                if tmp_aspect_clip[ac] == _df:
                    aspects_for_df.append(image_aspects[ac])
            print(len(aspects_for_df))
            print("tmp_aspect_clip:", tmp_aspect_clip)
            print("_df:",_df)
            
            if len(aspects_for_df) == 0:
                head_sentence = head_sentence + tmp_fds[_df]
            elif len(aspects_for_df) == 1:
                try:
                    head_sentence = head_sentence + aspects_for_df[0] + " wi" + tmp_fds[_df].split("wi")[1]
                except:
                    head_sentence = head_sentence + aspects_for_df[0] + ", " + tmp_fds[_df]
            else:
                df_w = []
                for aspect_w in aspects_for_df:
                    try:
                        t_w = aspect_w + " wi" + tmp_fds[_df].split("wi")[1]
                    except:
                        t_w = aspect_w + ", " + tmp_fds[_df]
                    df_w.append(t_w)
                try:
                    inputs = processor(text=df_w, images=image, return_tensors="pt", padding=True)
                    outputs = model(**inputs)
                    logits_per_image = outputs.logits_per_image # this is the image-text similarity score
                    probs = logits_per_image.softmax(dim=1) # we can take the softmax to get the label probabilities
                    print(probs[0])
                    for index_i in range(len(probs[0])):
                        if probs[0][index_i] == max(probs[0]):
                            head_sentence = head_sentence + df_w[index_i]
                            break
                except:
                    continue
        descriptions[image_name] = head_sentence

    return descriptions




def get_twitter_face_descriptions(path, image_path, save_path, face_path):
    modes = ["test", "dev", "train"]
    # modes = ["test"]
    face_descriptions = {} 
    for mode in modes:
        descriptions = generate_face_descriptions(path, mode, image_path, save_path, face_path)
        face_descriptions.update(descriptions)
    with open(save_path, "w+") as f:
        json.dump(face_descriptions, f)


get_twitter_face_descriptions(tw15_path, twitter15_image_path, twitter15_save_path, twitter15_face_path)
get_twitter_face_descriptions(tw17_path, twitter17_image_path, twitter17_save_path, twitter17_face_path)
