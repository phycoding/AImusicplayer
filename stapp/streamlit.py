import streamlit as st
import os
import cv2
import matplotlib.pyplot as plt
import numpy as np
import PIL
import keras
from keras.preprocessing.image import img_to_array
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
eye_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_eye.xml')
smile_cascade = cv2.CascadeClassifier(cv2.data.haarcascades +'haarcascade_smile.xml')
my_model = keras.models.load_model(r'C:\Users\ASUS\stapp\my_model\my_model.h5')
lookup = ('anger', 'disgust', 'fear', 'happiness', 'sadness', 'surprise', 'neutral')
anger_count = 0
disgust_count = 0
fear_count = 0
happiness_count = 0
sadness_count = 0
surprise_count = 0
nuetral_count = 0


# File Selecting for Songs Database

anger = ['Bangirimata', 'churake','Mankada']
disgust = ['He Ishwara','Husn']
fear = ['Ek Zindegi','Tu Chale']
happiness = ['Buttabamma','Choosa Choosa','Nashe si chadh gayi']
sadness = ['Crezy','Jashne-Bahara','Yeaina']
surprise = ['Aabaad Barbaad','BurjKhalifa','Dil Hai Deewana','Hurdum Humdam','Telusa']
Angerpath= 'C:\\Users\\ASUS\\stapp\\songs\\Anger'
Disgustpath= 'C:\\Users\\ASUS\\stapp\\songs\\Disgust'
Fearpath= 'C:\\Users\\ASUS\\stapp\\songs\\Fear'
Happinesspath= 'C:\\Users\\ASUS\\stapp\\songs\\Happiness'
Sadnesspath= 'C:\\Users\\ASUS\\stapp\\songs\\Sadness'
Surprisepath= "C:\\Users\\ASUS\\stapp\\songs\\Surprise"
anger_dict = {name:os.path.join(Angerpath,str(filee)+'.mp3') for name,filee in zip(anger,anger)}
disgust_dict = {name:os.path.join(Disgustpath,str(filee)+'.mp3') for name,filee in zip(disgust,disgust)}
fear_dict = {name:os.path.join(Fearpath,str(filee)+'.mp3') for name,filee in zip(fear,fear)}
happiness_dict = {name:os.path.join(Happinesspath,str(filee)+'.mp3') for name,filee in zip(happiness,happiness)}
sadness_dict = {name:os.path.join(Sadnesspath,str(filee)+'.mp3') for name,filee in zip(sadness,sadness)}
surprise_dict = {name:os.path.join(Surprisepath,str(filee)+'.mp3') for name,filee in zip(surprise,surprise)}
my_dict = {**anger_dict,**disgust_dict,**fear_dict,**happiness_dict,**sadness_dict,**surprise_dict}
my_list = anger+disgust+fear+happiness+sadness+surprise

def image_input():
    content_file=None
    if st.sidebar.checkbox('Upload'):
        content_file = st.sidebar.file_uploader("Choose a Content Image", type=["png", "jpg", "jpeg"])
    else:
      st.markdown('Please select **Upload button**.')
        

    if content_file is not None:
        content = detect_face(content_file)
        #orig = cv2.cvtColor(content,cv2.COLOR_BGR2GRAY)
        x = detect_emotion(content)
    else:
        st.warning("Upload an Image OR Untick the Upload Button)")
        st.stop()
    return st.image(content,caption = "You are looking like to have in: %s" %x)
def detect_face(images):
  try:
    images = PIL.Image.open(images)
    new_img = np.array(images.convert('RGB'))
    faces = face_cascade.detectMultiScale(new_img, 1.1, 4)
    for (x,y,w,h) in faces:
      #gray = cv2.cvtColor(new_img,cv2.COLOR_BGR2GRAY)
      cv2.rectangle(new_img,(x,y),(x+w,y+h),(255,0,0),7)
    return new_img
  except:
    #new_img = np.array(images.convert('RGB'))
    images = np.array(images)
    images.astype(np.float32)
    faces = face_cascade.detectMultiScale(images, 1.1, 4)
    for (x,y,w,h) in faces:
      #gray = cv2.cvtColor(new_img,cv2.COLOR_BGR2GRAY)
      cv2.rectangle(images,(x,y),(x+w,y+h),(255,0,0),7)
    return images
def webcam_input():
    st.header("Webcam Live Feed")
    #run = st.checkbox("Capture Your emotion")
    #st.button('My button', key='my_button',on_click=run)
    FRAME_WINDOW = st.image([], channels='BGR')
    SIDE_WINDOW = st.sidebar.image([], width=100, channels='BGR')
    status_text = st.empty()
    
    camera = cv2.VideoCapture(0)
    
    i = 0
    def run():
      for i in range(1,50):
        _, frame = camera.read()
        frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
        orig = frame.copy()
        orig = detect_face(orig)
        orig1 = cv2.cvtColor(orig,cv2.COLOR_BGR2GRAY)
        x = detect_emotion(orig)
       #orig1 = cv2.resize(orig1,(1,48,48,1),interpolation=cv2.INTER_AREA)
       #pred = my_model.predict(orig1)
        #orig = imutils.resize(frame, width=300)
        #frame = imutils.resize(frame, width=WIDTH)
        #target = detect_face_video(orig)
       
        FRAME_WINDOW.image(orig)
        SIDE_WINDOW.image(frame.copy())
        status_text.text('You are in: %s'%x)
        print(x)
       ##audio = st.sidebar.selectbox("Choose the style model: ",('Bangirimata','hello'),key='mylove'+str(i)
       #i= i+1
        if x == 'anger':
          global anger_count
          anger_count = anger_count+1
        if x == 'disgust':
          global disgust_count
          disgust_count = disgust_count+1
        if x == 'fear':
          global fear_count
          fear_count = fear_count+1
        if x == 'happiness':
          global happiness_count
          happiness_count = happiness_count+1
        if x == 'nuetral':
          global nuetral_count
          nuetral_count = nuetral_count+1
        if x == 'sadness':
          global sadness_count
          sadness_count = sadness_count+1
        if x == 'surprise':
          global surprise_count
          surprise_count = surprise_count+1
      y = [anger_count,disgust_count,fear_count,happiness_count,sadness_count,surprise_count,nuetral_count]
      y = np.array(y)
      y = np.argmax(y)
      st.header('Looks like you are in: %s ' %lookup[y])
      global my_dict
      global my_list
      if lookup[y] == 'anger':
        my_dict = anger_dict
        my_list = anger
      if lookup[y] == 'sadness':
        my_list = sadness
        my_dict = sadness_dict
      if lookup[y] == 'happiness':
        my_list = happiness
        my_dict = happiness_dict
      if lookup[y] == 'disgust':
        my_list = disgust
        my_dict = disgust_dict
      if lookup[y] == 'surprise':
        my_list = surprise
        my_dict = surprise_dict
      if lookup[y] == 'fear':
        my_list = fear
        my_dict = fear_dict
      if lookup[y] == 'nuetral':
        st.header('Choose any song and boost')
        n = random.randint(1,12)
        audio = st.selectbox("Recommended songs for your Mood",[my_list[n]])
        st.title('Hope it will boost your Mood')
        st.audio(my_dict[audio])
        
      import random
      try:
        n = random.randint(1,4)
        audio = st.selectbox("Recommended songs for your Mood",[my_list[n]])
        st.title('Hope it will boost your Mood')
        st.audio(my_dict[audio])
      except:
        n = random.randint(1,2)
        audio = st.selectbox("Recommended songs for your Mood",[my_list[n]])
        st.title('Hope it will boost your Mood')
        st.audio(my_dict[audio])
      

    
      
      

      #audio = st.sidebar.selectbox("Recommended songs for your Mood",Surprise_Name)
      #st.title('Hope it will boost your Mood')
      #st.audio(Surprise_dict[audio])
      #st.audio(Anger_dict['Bangiri Mata'])
   
    #else:
        #st.warning("NOTE: Streamlit currently doesn't support webcam. So to use this, clone this repo and run it on local server.")
        #st.warning('Stopped')
    st.button('Capture', key='my_button',on_click=run)
      #st.title([fear_count,anger_count,happiness_count,nuetral_count,sadness_count,surprise_count,disgust_count])
    #st.audio(Anger_dict['Bangiri Mata'])
    #return print(surprise_count,sadness_count,happiness_count,anger_count,fear_count,disgust_count,anger_count)
def detect_emotion(images):
  try:
    images = PIL.Image.open(images)
    new_img = np.array(images.convert('RGB'))
    faces = face_cascade.detectMultiScale(new_img, 1.1, 4)
    for (x,y,w,h) in faces:
      gray = cv2.cvtColor(new_img,cv2.COLOR_BGR2GRAY)
      roi_gray=gray[y:y+h,x:x+w]
      roi_gray=cv2.resize(roi_gray,(48,48),interpolation=cv2.INTER_AREA)
      if np.sum([roi_gray])!=0:
        roi=roi_gray.astype('float')/255.0
        roi=img_to_array(roi)
        roi=np.expand_dims(roi,axis=0)
        preds=my_model.predict(roi)[0]
        
  except:
    #new_img = np.array(images.convert('RGB'))
    images = np.array(images)
    images.astype(np.float32)
    faces = face_cascade.detectMultiScale(images, 1.1, 4)
    #status_text = st.text("loading..")
    status_text = st.empty()
    for (x,y,w,h) in faces:
      gray = cv2.cvtColor(images,cv2.COLOR_BGR2GRAY)
      roi_gray=gray[y:y+h,x:x+w]
      status_text.empty()
      roi_gray=cv2.resize(roi_gray,(48,48),interpolation=cv2.INTER_AREA)
      if np.sum([roi_gray])!=0:
        roi=roi_gray.astype('float')/255.0
        roi=img_to_array(roi)
        roi=np.expand_dims(roi,axis=0)
        preds=my_model.predict(roi)[0]
        label = lookup[np.argmax(preds)]
        return label
def main():
  st.title("AI Music player")
  st.header('Let allow us to boost your mood')
  st.sidebar.title('Navigation')
  
  method = st.sidebar.radio('Go To ->', options=['Webcam'],key = 'amaifkajlfj')
  st.sidebar.header('Choose your Music')
  if method == 'Image':
    image_input()
  else:
    webcam_input()
  audio = st.sidebar.selectbox("Play a song of your choice",my_list,key = 'you')
  st.header('Hope you are enjoying the song')
  print(my_dict)
  st.audio(my_dict[audio])

  #audio = st.sidebar.selectbox("Choose the style model: ",('Bangirimata','hello'))

  #if method == 'Image':
    #image_input()
  #else:
    #webcam_input(my_list,my_dict)
if __name__ == "__main__": 
    main()
