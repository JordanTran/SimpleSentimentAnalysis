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
def computeWeights(entered,vect,tf):
    comment = np.array([entered])
    prediction = sentimentModel(torch.tensor((tf.transform(vect.transform(comment)).astype(np.float32).toarray()))).detach().numpy()
    return "Negative Score: " + str(prediction[0][0])+ "\nNeutral Score: "+str(prediction[0][1])+ "\nPositive Score: "+ str(prediction[0][2])
def clear():
    output.delete(0.0,END)
    textentry.delete(0,END)
    weights.delete(0.0,END)
    output.config(state=DISABLED)
    weights.config(state=DISABLED)
def click(e):
    entered = textentry.get()
    output.config(state=NORMAL)
    weights.config(state=NORMAL)
    #output.insert(END, entered+'\n')
    output.insert(END, emogen(entered,vect,tf))
    weights.insert(END, computeWeights(entered,vect,tf))
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
window.title('Pre-Alphas Emogen V0.01')
width= window.winfo_screenwidth()               
height= window.winfo_screenheight()               
window.geometry("%dx%d"%(width, height))
window.minsize(width, height)
window.maxsize(width, height)
window.configure(background = 'black')
l = Label(window, text= 'Enter your comment:',bg = 'black', fg = 'white', font = 'none 12 bold')
l.place(x=width/2, y=50, anchor="center")
textentry = Entry(window, width = 50,font = ("Courier", 18), bg = 'white')
textentry.place(x=width/2, y=75, anchor="center")
output = Text(window,width=2,height = 1, wrap = WORD , background='black',fg='white', font = ("Courier", 350),highlightthickness = 0, borderwidth=0,state=DISABLED)
output.place(x=width/2, y=height/2, anchor="center")
exitButton = Button(window, text ='Exit',font = 'none 12 bold',width=14,command=close_window, justify='center')
exitButton.place(x=width/2, y=height-100, anchor="center")
weights = Text(window,width=50,height = 10, wrap = WORD , background='black',fg='white', font = ("Courier", 12),highlightthickness = 0, borderwidth=0,state=DISABLED)
weights.place(x=400, y=height-100, anchor="center")
window.bind('<Return>',click)

window.mainloop()
