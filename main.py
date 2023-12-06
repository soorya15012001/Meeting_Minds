import tkinter as tk
from time import sleep
# pavucontrol => If microphone is not working

import warnings
warnings.filterwarnings('ignore')
from vosk import SetLogLevel
SetLogLevel(-1)

from vosk import Model, KaldiRecognizer
import os
import pyaudio
import torch
from transformers import AutoTokenizer, AutoModelWithLMHead
from rake_nltk import Rake
import datetime
import re
import tkinter as tk
from tkinter import *
import threading
import time
import numpy as np

tokenizer = AutoTokenizer.from_pretrained('t5-base')
model = AutoModelWithLMHead.from_pretrained('t5-base', return_dict=True)


def T5_word_embed(model, tokenizer, text):

    final_text = " ".join(text)

    i = tokenizer(final_text, return_tensors="pt", return_attention_mask=False, add_special_tokens=False)
    o = model.encoder.embed_tokens(i.input_ids)



    final_embed = {}
    words = final_text.split()

    for id, embed in zip(i.word_ids(),o[0]):
        final_embed[words[id]] = embed
    
    sentence_embed = {}
    for i in text:
        final = []
        for w in i.split():
            final.append(final_embed[w].detach().numpy())
        
        sentence_embed[i] = np.mean(final, axis=0)
    
    return sentence_embed
        

def T5_sim(q, docs):
    doc_sim = {}
    q = list(q.values())
    for d in docs.items():
        d_v = np.array(d[1])
        q_v = np.array(q)
        sim = np.dot(q_v, d_v)
        doc_sim[d[0]] = sim[0]
    return dict(sorted(doc_sim.items(), key=lambda item: item[1], reverse=True))



def summarizer(summary):
    output_label.config(text="", wraplength=1500, justify="center", font=("Arial", 15))    
    output_label.config(text="\n"+summary, wraplength=1500, justify="center", font=("Arial", 15))
    output_label.pack()



def raker(rake):
    output_label.config(text="", wraplength=1500, justify="center", font=("Arial", 15))    
    output_label.config(text="\n"+rake, wraplength=1500, justify="center", font=("Arial", 15))
    output_label.pack()



def read_and_store_text(entry, text, read_button):
    output_label.config(text="", wraplength=1500, justify="center", font=("Arial", 15))

    query = [entry.get()]

    query_embed = T5_word_embed(model, tokenizer, query) 
    data_embed = T5_word_embed(model, tokenizer, text)
    final_sim = ""
    for i in list(T5_sim(query_embed, data_embed).keys())[:5]:
        i = re.sub('<[^>]+>', '', i).strip()
        final_sim = final_sim.strip() + "\n\n* " + i

    output_label.config(text="\n"+final_sim, wraplength=1500, justify="center", font=("Arial", 15))
    output_label.pack()

    entry.pack_forget()
    read_button.pack_forget()




def search(text):
    output_label.config(text="", wraplength=1500, justify="center", font=("Arial", 15))    

    entry = tk.Entry(root, width=30)
    entry.pack(pady=10)

    read_button = tk.Button(root, text="Submit", command= lambda: read_and_store_text(entry, text, read_button))
    read_button.pack(pady=10)



def task():
    # The window will stay open until this function call ends.
    p = pyaudio.PyAudio()
    stream = p.open(format=pyaudio.paInt16, channels=1, rate=16000, input=True, frames_per_buffer=8000)
    stream.start_stream()

    sp_model = Model("model")
    rec = KaldiRecognizer(sp_model, 16000)

    def takeCommand():
        while True:
            final_string = ''

            data = stream.read(2000)
            if len(data) == 0:
                break
            if rec.AcceptWaveform(data):
                curr = eval(rec.Result())['text']
                final_string = final_string + curr
                return final_string.lower()

    text = []
    while True:
        label_listen = tk.Label(text="LISTENING...")
        label_listen.pack()

        msg = takeCommand()
        print(msg)
        text.append(msg)
        if "bye" in msg.strip():
            break
        # time.sleep(5)
        # break


    print("outside")
    label_listen.destroy()
    label_process = tk.Label(text="PROCESSING...")
    label_process.pack()

    # text = ['After receiving the unexpected diagnosis of "cardiovascular cancer," you immediately consult with a team of renowned medical experts who are puzzled by the rarity of your condition', ' Despite the gravity of the situation, they propose an unconventional treatment plan', ' In an effort to combat the supposed cancer, they recommend a combination of paracetamol, vitamin supplements, and a carefully curated diet', '\n\nYour treatment regimen begins with paracetamol, which the doctors believe will alleviate some of the symptoms associated with this peculiar form of "cardiovascular cancer', '" They prescribe a specific dosage: one tablet at 4:00 PM and another at 11:00 PM daily', ' The medical team emphasizes the importance of adhering strictly to this schedule for optimal effectiveness', '\n\nIn addition to medication, they suggest incorporating lifestyle changes to support your overall well-being', ' A personalized exercise routine, stress management techniques, and a diet rich in antioxidants become integral parts of your daily life', ' The medical community is closely monitoring your progress, and your case has become a subject of interest among researchers due to its uniqueness', '\n\nAs you embark on this unexpected journey, you find support from friends and family, who rally around you with encouragement and positivity', ' Local and international media catch wind of this extraordinary case, turning you into a symbol of hope for those facing challenging health issues', '\n\nThroughout the month-long treatment, you document your experiences in a blog, sharing your journey with the world', ' Your story resonates with many, inspiring others to approach adversity with resilience and optimism', '\n\nAs the days pass, your health gradually improves', ' The medical team, intrigued by the positive response to the paracetamol regimen, conducts further studies to understand the underlying mechanisms at play', ' The mysterious "cardiovascular cancer" begins to fade into medical history as a unique anomaly with a surprisingly manageable solution']

    final_text = ' '.join(text).strip()


    inputs = tokenizer.encode("summarize: " + final_text, return_tensors='pt')
    summary_ids = model.generate(inputs, max_length=512, min_length=80, length_penalty=5., num_beams=2)
    summary = tokenizer.decode(summary_ids[0])
    summary = re.sub('<[^>]+>', '', summary).strip()

    r = Rake()
    r.extract_keywords_from_text(final_text)
    rake = ""
    for i in r.get_ranked_phrases()[0:10]:
        rake = rake + "\n\n* " + i




    label.pack_forget()
    label_process.pack_forget()

    btn1 = Button(root, text = 'See Summary', command = lambda: summarizer(summary))
    btn2 = Button(root, text = 'See Top Sentences', command = lambda: raker(rake))
    btn3 = Button(root, text = 'Search something', command = lambda: search(text))

    btn1.pack()
    btn2.pack()
    btn3.pack()




    # label = tk.Label(text=summary, wraplength=400, justify="center", font=("Arial", 17))
    # label.pack()
    # btn = Button(root, text = 'See Summary', command = lambda: raker(rake, summary, label, btn)) 
    # btn.pack(side = 'bottom')
    

    # summarizer(summary, rake, label_process, label)




root = tk.Tk()
root.geometry('1500x1500') 
output_label = tk.Label(root, text="")


frameCnt = 23
frames = [PhotoImage(file='mygif.gif',format = 'gif -index %i' %(i)) for i in range(frameCnt)]

def update(ind):
    frame = frames[ind]
    ind += 1
    if ind == frameCnt:
        ind = 0
    label.configure(image=frame)
    root.after(100, update, ind)

label = Label(root)
label.pack()

# root.after(500, task)
task_thread = threading.Thread(target=task)
task_thread.start()
root.after(0, update, 0)

root.mainloop()
task_thread.join(1)