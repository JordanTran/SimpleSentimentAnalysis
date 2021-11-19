import emoji
import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfTransformer
from os import system, name
import torch
import torch.nn as nn
import pickle
from tkinter import *
import threading 
#Feedfoward network
class SimpleNetwork(nn.Module):
    def __init__(self, input_size, hidden_size, num_classes):
        super(SimpleNetwork,self).__init__()
        self.l1 = nn.Linear(input_size, hidden_size)
        self.reLu = nn.ReLU()
        self.l2 = nn.Linear(hidden_size , num_classes)
    def forward(self, x):
        out = self.l1(x)
        out = self.reLu(out)
        out = self.l2(out)
        return out
#returning emoji from sentence input
def emogen(sentence, vect , tf):
        comment = np.array([sentence])
        sentiment = {0:emoji.emojize(":pleading_face:"),
            1:emoji.emojize(":neutral_face:"),
            2:emoji.emojize(":grinning_face_with_smiling_eyes:")}
        prediction = sentimentModel(torch.tensor((tf.transform(vect.transform(comment)).astype(np.float32).toarray())))
        return sentiment[prediction.detach().numpy().argmax()]
def clear():
    output.delete(0.0,END)
    textentry.delete(0,END)
    output.config(state=DISABLED)
def click(e):
    entered = textentry.get()
    output.config(state=NORMAL)
    output.delete(0.0,END)
    #output.insert(END, entered+'\n')
    output.insert(END, emogen(entered,vect,tf))
    delay = threading.Timer(2,clear)
    delay.start()
def close_window():
    window.destroy()
    exit()
# Loading Models
with open('textprocessing_models.pkl', 'rb') as f:
     vect, tf = pickle.load(f)
sentimentModel = SimpleNetwork(131695, 100, 3)
sentimentModel.load_state_dict(torch.load('sentimentModel.pth'))
sentimentModel.eval()

#Creating gui
window = Tk()
window.title('Simple Sentiment Analysis')
window.geometry("1000x1000")
window.minsize(1000, 1000)
window.maxsize(1000, 1000)
window.configure(background = 'black')
l = Label(window, text= 'Enter your comment:',bg = 'black', fg = 'white', font = 'none 12 bold')
l.place(x=500, y=50, anchor="center")
textentry = Entry(window, width = 50,font = ("Courier", 18), bg = 'white')
textentry.place(x=500, y=75, anchor="center")
output = Text(window,width=2,height = 1, wrap = WORD , background='black',fg='white', font = ("Courier", 350),highlightthickness = 0, borderwidth=0,state=DISABLED)
output.place(x=520, y=500, anchor="center")
exitButton = Button(window, text ='Exit',width=14,command=close_window, justify='center')
exitButton.place(x=500, y=987, anchor="center")
window.bind('<Return>',click)

window.mainloop()
