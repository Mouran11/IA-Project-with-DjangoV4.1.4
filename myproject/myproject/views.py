from django.shortcuts import render
import numpy as np
import tensorflow as tf
import os
from django.conf import settings
from django.core.files.storage import default_storage
from tensorflow.keras.utils import img_to_array, load_img
from tensorflow.keras import layers, models
from .Model import MyModel
import librosa
import librosa.display
import matplotlib.pyplot as plt
from keras.models import load_model
from python_speech_features import mfcc
import numpy as np



def home(request):
    return render(request, 'index.html')


# def DeleteTemporaryFiles():
#     files = os.scandir('./static/model_data')
#     for file in files:
#         if file.is_file():
#             os.remove(file)

def predict(request):
    if request.method == 'POST':
        file = request.FILES['image']
        file_name = os.path.join(settings.MEDIA_ROOT, file.name)
        fileName= default_storage.save(file_name, file)
        file_url = default_storage.path(fileName)
        print(fileName)
        (name, prob) = MyModel.predict(img_path=file_url)
        print(name, prob)
        if name == 'NORMAL':
            prob = f'La probabilité de ne pas avoir de pneumonie est  {str((1-prob*100)*100)[:5]}%'
        elif name == 'PNEUMONIA':
            prob = f"La probabilité d'étre infecté par la pneumonie est {str(prob*100)[:5]}%"
        return render(request, 'results.html', {
            'predicted_name': name,
            'probability': prob,
            'url':'img/'+fileName,
        })
    return render(request, 'index.html')
           

def heart(request):
    return render(request, 'classify_heart.html')

def load_file_data (file_names, duration=12, sr=16000):
    input_length=sr*duration
    data = []
    sound_file=file_names
    print ("load file ",sound_file)
            # utiliser la technique kaiser_fast pour une extraction plus rapide
    X, sr = librosa.load( sound_file, sr=sr, duration=duration,res_type='kaiser_fast') 
    dur = librosa.get_duration(y=X, sr=sr)
            # pad audio file same duration
    if (round(dur) < duration):
        print ("fixing audio lenght :", file_names)
        y = librosa.util.fix_length(X, input_length)                         
            # extraire la fonction mfcc pour normalisée des données
        mfccs = np.mean(librosa.feature.mfcc(y=X, sr=sr, n_mfcc=40).T,axis=0)
        feature = np.array(mfccs).reshape([-1,1])
        data.append(feature)             
        
    return data
    
def generate_waveform(file_path):
    y, sr = librosa.load(file_path , duration=5)
    plt.figure()
    librosa.display.waveshow(y, sr=sr)
    plt.axis('off')
    plt.savefig(settings.MEDIA_ROOT+'/waveform.png', bbox_inches='tight', pad_inches=-0.1)

def predictHeart(request):
    if request.method == 'POST':
        audio_file = request.FILES['file']
        file_name = os.path.join(settings.MEDIA_ROOT, audio_file.name)
        fileName= default_storage.save(file_name, audio_file)
        file_url = default_storage.path(fileName)
        
        MAX_SOUND_CLIP_DURATION=12
        CLASSES = ['artifact','murmur','normal']

        # Map integer value to text labels
        label_to_int = {k:v for v,k in enumerate(CLASSES)}
        print (label_to_int)
        # map integer to label text
        int_to_label = {v:k for k,v in label_to_int.items()}
        print(int_to_label)        

        # Load the pre-trained model
        model = load_model('./'+settings.STATIC_URL+'model1_heartsound.h5')

        # Make a prediction using the model
        test_sounds = load_file_data(file_names=file_url, duration=MAX_SOUND_CLIP_DURATION)
        print('Testing record files: ', len(test_sounds))
        test = np.array(test_sounds).reshape((len(test_sounds),-1,1))
        prediction = model.predict(test)
        pred = np.argmax(prediction, axis=1)
        for i in range(len(pred)):
            if pred[i] == 0 :
                res = f'vos données sont mal enregistrées - {int_to_label[pred[i]]} - , utilisez à nouveau un autre enregistrement '
            elif pred[i] == 1 : 
                res = f"Après analyse de votre enregistrement de classe {int_to_label[pred[i]]} - , nous avons constaté que vous avez un problème de battemmnet cardiaque"
            else :
                res = f"vous etes pas malade avec une prédiction d'un personne,- {int_to_label[pred[i]]} -"

        
        plot = generate_waveform(file_url)
        # Render the prediction in a template
        return render(request, 'classification_results.html', {'res': res, 'url':'img/'+settings.MEDIA_ROOT+'/waveform.png'})
    return render(request, 'classify_heart.html')
