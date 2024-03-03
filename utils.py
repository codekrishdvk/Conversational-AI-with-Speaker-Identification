#  --------------------------------------------- si -------------------------------------------------

import os
import wave
import time
import pickle
import pyaudio
import warnings
import numpy as np
from sklearn import preprocessing
from scipy.io.wavfile import read
import python_speech_features as mfcc
from sklearn.mixture import GaussianMixture 

warnings.filterwarnings("ignore")

def calculate_delta(array):
    rows, cols = array.shape
    deltas = np.zeros((rows, 20))
    N = 2
    for i in range(rows):
        index = []
        j = 1
        while j <= N:
            first = max(0, i - j)
            second = min(rows - 1, i + j)
            index.append((second, first))
            j += 1
        deltas[i] = (array[index[0][0]] - array[index[0][1]] + (2 * (array[index[1][0]] - array[index[1][1]]))) / 10
    return deltas


def extract_features(audio, rate):
    mfcc_feature = mfcc.mfcc(audio, rate, 0.025, 0.01, 20, nfft=1200, appendEnergy=True)    
    mfcc_feature = preprocessing.scale(mfcc_feature)
    delta = calculate_delta(mfcc_feature)
    combined = np.hstack((mfcc_feature, delta)) 
    return combined


def record_audio_train():
    Name = input("Please Enter Your Name:")
    for count in range(5):
        FORMAT = pyaudio.paInt16
        CHANNELS = 1
        RATE = 44100
        CHUNK = 512
        RECORD_SECONDS = 10
        device_index = 2
        audio = pyaudio.PyAudio()
        print("----------------------record device list---------------------")
        info = audio.get_host_api_info_by_index(0)
        numdevices = info.get('deviceCount')
        for i in range(0, numdevices):
            if (audio.get_device_info_by_host_api_device_index(0, i).get('maxInputChannels')) > 0:
                print("Input Device id ", i, " - ", audio.get_device_info_by_host_api_device_index(0, i).get('name'))
        print("-------------------------------------------------------------")
        index = int(input())       
        print("recording via index "+str(index))
        stream = audio.open(format=FORMAT, channels=CHANNELS,
                        rate=RATE, input=True,input_device_index=index,
                        frames_per_buffer=CHUNK)
        print("recording started")
        Recordframes = []
        for i in range(0, int(RATE / CHUNK * RECORD_SECONDS)):
            data = stream.read(CHUNK)
            Recordframes.append(data)
        print("recording stopped")
        stream.stop_stream()
        stream.close()
        audio.terminate()
        OUTPUT_FILENAME = f"{Name}-sample{count}.wav"
        WAVE_OUTPUT_FILENAME = os.path.join("C:\\Users\\rkris\\Documents\\code\\duplicate\\speakerIdentification\\training_set", OUTPUT_FILENAME)
        trainedfilelist = open("C:\\Users\\rkris\\Documents\\code\\duplicate\\speakerIdentification\\training_set_addition.txt", 'a')
        trainedfilelist.write(OUTPUT_FILENAME+"\n")
        waveFile = wave.open(WAVE_OUTPUT_FILENAME, 'wb')
        waveFile.setnchannels(CHANNELS)
        waveFile.setsampwidth(audio.get_sample_size(FORMAT))
        waveFile.setframerate(RATE)
        waveFile.writeframes(b''.join(Recordframes))
        waveFile.close()

def record_audio_test():
    FORMAT = pyaudio.paInt16
    CHANNELS = 1
    RATE = 44100
    CHUNK = 512
    RECORD_SECONDS = 10
    device_index = 2
    audio = pyaudio.PyAudio()
    print("----------------------record device list---------------------")
    info = audio.get_host_api_info_by_index(0)
    numdevices = info.get('deviceCount')
    for i in range(0, numdevices):
        if (audio.get_device_info_by_host_api_device_index(0, i).get('maxInputChannels')) > 0:
            print("Input Device id ", i, " - ", audio.get_device_info_by_host_api_device_index(0, i).get('name'))
    print("-------------------------------------------------------------")
    index = int(input())       
    print("recording via index "+str(index))
    stream = audio.open(format=FORMAT, channels=CHANNELS,
                    rate=RATE, input=True,input_device_index=index,
                    frames_per_buffer=CHUNK)
    print("recording started")
    Recordframes = []
    for i in range(0, int(RATE / CHUNK * RECORD_SECONDS)):
        data = stream.read(CHUNK)
        Recordframes.append(data)
    print("recording stopped")
    stream.stop_stream()
    stream.close()
    audio.terminate()
    OUTPUT_FILENAME = "sample.wav"
    WAVE_OUTPUT_FILENAME = os.path.join("C:\\Users\\rkris\\Documents\\code\\duplicate\\speakerIdentification\\testing_set", OUTPUT_FILENAME)
    trainedfilelist = open("C:\\Users\\rkris\\Documents\\code\\duplicate\\speakerIdentification\\testing_set_addition.txt", 'a')
    trainedfilelist.write(OUTPUT_FILENAME+"\n")
    waveFile = wave.open(WAVE_OUTPUT_FILENAME, 'wb')
    waveFile.setnchannels(CHANNELS)
    waveFile.setsampwidth(audio.get_sample_size(FORMAT))
    waveFile.setframerate(RATE)
    waveFile.writeframes(b''.join(Recordframes))
    waveFile.close()

def train_model():
    source = "C:\\Users\\rkris\\Documents\\code\\duplicate\\speakerIdentification\\training_set\\"   
    dest = "C:\\Users\\rkris\\Documents\\code\\duplicate\\speakerIdentification\\trained_models\\"
    train_file = "C:\\Users\\rkris\\Documents\\code\\duplicate\\speakerIdentification\\training_set_addition.txt"        
    file_paths = open(train_file,'r')
    count = 1
    features = np.asarray(())
    for path in file_paths:    
        path = path.strip()   
        sr, audio = read(os.path.join(source, path))
        vector = extract_features(audio, sr)
        
        if features.size == 0:
            features = vector
        else:
            features = np.vstack((features, vector))

        if count == 5:    
            gmm = GaussianMixture(n_components=6, max_iter=200, covariance_type='diag', n_init=3)
            gmm.fit(features)
            
            # dumping the trained gaussian model
            picklefile = path.split("-")[0] + ".gmm"
            pickle.dump(gmm, open(os.path.join(dest, picklefile), 'wb'))
            print(f"Modeling completed for speaker: {picklefile} with data point = {features.shape}")   
            features = np.asarray(())
            count = 0
        count += 1

def test_model():
    source = "C:\\Users\\rkris\\Documents\\code\\duplicate\\speakerIdentification\\testing_set\\"  
    modelpath = "C:\\Users\\rkris\\Documents\\code\\duplicate\\speakerIdentification\\trained_models\\"
    test_file = "C:\\Users\\rkris\\Documents\\code\\duplicate\\speakerIdentification\\testing_set_addition.txt"       
    file_paths = open(test_file,'r')
     
    gmm_files = [os.path.join(modelpath, fname) for fname in os.listdir(modelpath) if fname.endswith('.gmm')]
     
    # Load the Gaussian gender Models
    models = [pickle.load(open(fname, 'rb')) for fname in gmm_files]
    speakers = [fname.split("\\")[-1].split(".gmm")[0] for fname in gmm_files]
     
    # Read the test directory and get the list of test audio files 
    for path in file_paths:   
        path = path.strip()   
        sr, audio = read(os.path.join(source, path))
        vector = extract_features(audio, sr)
         
        log_likelihood = np.zeros(len(models)) 
        
        for i in range(len(models)):
            gmm = models[i]  # checking with each model one by one
            scores = np.array(gmm.score(vector))
            log_likelihood[i] = scores.sum()
         
        winner = np.argmax(log_likelihood)
        print("\tdetected as - ", speakers[winner])
        time.sleep(1.0)
        global identifiedName
        identifiedName = speakers[winner]
      

while True:
    choice = int(input("\n-----------------------------------------\n\tSpeaker Identification\n-----------------------------------------\n\n 1.Record audio for training \n 2.Train Model \n 3.Record audio for testing \n 4.Test Model\n-----------------------------------------\nEnter Your Choice : "))
    if choice == 1:
        record_audio_train()
    elif choice == 2:
        train_model()
    elif choice == 3:
        record_audio_test()
    elif choice == 4:
        test_model()
    elif choice > 4:
        exit
        break
        
# -------------------------------------------    utils    -------------------------------------------------------------------

from openai import OpenAI
import os
import base64
import streamlit as st

api_key = "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"

user_name = identifiedName

client = OpenAI(api_key=api_key) 

def get_answer(messages):
    system_message = [{"role": "system", "content": f'''As an English language tutor AI -- be like a normal indian friend, start the conversation with "Hello {user_name} [if the user name is krishnan , say hello admin] ,
                        I am Krish. I am your English language tutor.
                        After telling this start the conversation, test their english grammatical knowledge by asking one or two 
                        funny english question like realated to indian or tamilan jokes,
                        by identifying their knowledege go along with, you should not leave them or stop in any situation,
                        each ending of your answer you should ask question like engaging them,
                        ensuring each response prompts the user to engage in conversation,
                        correct any grammatical mistakes,
                        and ask questions to maintain dialogue flow.
                        If the user's response is grammatically incorrect,
                        provide the correct version and prompt them to repeat it.
                        Encourage the user to repeat after you for practice.
                       
                        If the user asks for a quiz, generate a quiz question related to English language learning to help improve their skills.
                        If the user asks for their preparing for interview and they ask to interview them, you should get their what job they looking for and then you go for interview like asking 4 to 5 questions.
                        '''}]
    messages = system_message + messages
    response = client.chat.completions.create(
        model="gpt-3.5-turbo-1106",
        messages=messages
    )
    return response.choices[0].message.content

def speech_to_text(audio_data):
    with open(audio_data, "rb") as audio_file:
        transcript = client.audio.transcriptions.create(
            model="whisper-1",
            response_format="text",
            file=audio_file,
            language="en"
        )
    return transcript

def text_to_speech(input_text):
    response = client.audio.speech.create(
        model="tts-1",
        voice="nova",
        input=input_text
    )
    webm_file_path = "temp_audio_play.mp3"
    with open(webm_file_path, "wb") as f:
        response.stream_to_file(webm_file_path)
    return webm_file_path

def autoplay_audio(file_path: str):
    with open(file_path, "rb") as f:
        data = f.read()
    b64 = base64.b64encode(data).decode("utf-8")
    md = f"""
    <audio autoplay>
    <source src="data:audio/mp3;base64,{b64}" type="audio/mp3">
    </audio>
    """
    st.markdown(md, unsafe_allow_html=True)




# ========================
        




