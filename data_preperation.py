## here we are generating data breed wise
## and assighning colour breedwise and class wise
import time
import glob
import json
import pandas as pd
import numpy as np
import cv2
import pickle
import logging

config=json.load(open("config.json",'r'))

def read_txt_and_prepare_data(config):
    with open(config['annotaion_directory']+"\\trainval.txt",'r') as f:
        train_txt=f.readlines()
    temp=[]     
    for i in train_txt: 
        flag=True
        c=""
        split_val=i.split(" ")[0].split("_")[0:-1]
        for j in split_val:
            if flag:
                c+=j
                flag=False
            else:
                c+="_"+j
        if c not in temp:
            temp.append(c)

    breed={"cat":[],'dog':[]}

    for i in temp:
        if i[0].islower():
            breed['dog'].append(i)
        else:
            breed['cat'].append(i)

    cat_breed_color={i:[] for i in breed['cat']}
    dog_breed_color={i:[] for i in breed['dog']}

    ## red channel for cat
    ## green channel for dog
    temp=0
    for i in cat_breed_color.keys():
        cat_breed_color[i].append(130+temp)
        cat_breed_color[i].append(0)
        cat_breed_color[i].append(0)
        temp+=10
    ## manually adding color for dog_breed

    # json.dump(dog_breed_color,open("dog_breed_color.json",'w'))
    # json.dump(cat_breed_color,open("cat_breed_color.json",'w'))
    



def display_img(img,img1,time):
    cv2.imshow("img",img)
    cv2.imshow("img1",img1)
    cv2.waitKey(time)
    cv2.destroyAllWindows()

def color_trymap(img,cl):

    img=np.where(img==2,0,img)
    c=np.where(img==1)
    c=np.array(list(set(zip(c[0],c[1]))))
    for i in c:
        img[i[0],i[1]]=cl
    return img

def trimap_to_img(config):

    trimap=config['annotaion_directory']+"\\trimaps"
    image=config['images_directory']
    cat_breed_color=json.load(open("cat_breed_color.json",'r'))
    dog_breed_color=json.load(open("dog_breed_color.json",'r'))

    x,y=[],[]
    
    for i in glob.glob(trimap+"\\*.png"):
        try:
            temp=i.split("\\")[-1].split(".")[0].split("_")[0:-1]
            if len(temp)>1:
                animal=""
                flag=1
                for j  in temp:
                    if flag:
                        animal+=j
                        flag=0
                    else:
                        animal+="_"+j

            else:
                animal=i.split("\\")[-1].split(".")[0].split("_")[0]
            tempy=cv2.imread(i,1)
            tempy=cv2.resize(tempy,(128,128))
            tempx=cv2.imread(image+"\\"+i.split("\\")[-1].split(".")[0]+".jpg",1)
            tempx=cv2.resize(tempx,(128,128))
            if animal in cat_breed_color.keys():
                c=cat_breed_color[animal]
                tempy=color_trymap(tempy,c)
            elif animal in dog_breed_color.keys():
                c=dog_breed_color[animal]
                tempy=color_trymap(tempy,c)
            x.append(tempx)
            y.append(tempy)
        except Exception as error:
            logging.exception(error)
    return np.array(x),np.array(y)


def save_data(x,y):
    pickle.dump(x,open("x_128.pkl","wb"))
    pickle.dump(y,open("y_128_mask_rgb.pkl","wb"))

# x,y=pickle.load(open("x_128.pkl",'rb')),pickle.load(open("y_128_mask_rgb.pkl",'rb'))

x,y=trimap_to_img(config)
save_data(x,y)
