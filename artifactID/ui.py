import os
from tkinter import *
from tkinter import filedialog

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import pydicom as dicom
from matplotlib.backends.backend_tkagg import (FigureCanvasTkAgg, NavigationToolbar2Tk)

myWindow = Tk()
myWindow.geometry("1000x600")
myWindow.title("User Interface")

# Find Home Directory (My Home Directory Path = /Users/cameronashe)
myDirectory_Home = os.path.expanduser('~')
myFolder_Path = str()


# Create Class To Manage Plot
class myPlot_Manager(object):
    def __init__(self, myPlot, myImages):
        self.myPlot = myPlot
        # Create Figure Title
        myPlot.set_title('DICOM Image Slices')

        self.myImages = myImages
        rows, cols, self.mySlices = myImages.shape
        self.myIndex = self.mySlices // self.mySlices
        self.myLength = self.mySlices - 1
        self.myPics = myPlot.imshow(self.myImages[:, :, self.myIndex], cmap='gray')
        self.myUpdate()

    def myScroll(self, event):
        if type(event) == mpl.backend_bases.MouseEvent:
            # print("%s %s" % (event.button, event.step))
            if event.button == 'up':
                self.myIndex = (self.myIndex + 1) % self.mySlices
            else:
                self.myIndex = (self.myIndex - 1) % self.mySlices
            self.myUpdate()

    def myUpdate(self):
        self.myPics.set_data(self.myImages[:, :, self.myIndex])
        self.myPlot.set_xlabel('Slice Number: %s/%s' % (self.myIndex, self.myLength))
        self.myPics.axes.figure.canvas.draw()


# Function For Creating List of DICOM Slices
def myDICOM_Slices(myFolderPath):
    myFolder_List = os.listdir(myFolderPath)
    myDICOM_Files = [dicom.dcmread(myFolderPath + '/' + myFiles, force=True) for myFiles in myFolder_List]
    return myDICOM_Files


# Function For Creating A Stack of DICOM Images
def myDICOM_Stack(myDICOMFiles):
    myDICOM_Pile = np.dstack([myDICOMSlices.pixel_array for myDICOMSlices in myDICOMFiles])
    return myDICOM_Pile


# Function For Run Artifact ID
def myArtifactID():
    print(myFolder_Path)


# Function For myFile_Open Command
def myFile_Open():
    # Create Directory Window to Choose Folder
    global myFolder_Path
    myFolder_Path = filedialog.askdirectory(initialdir=myDirectory_Home)

    if myFolder_Path != '':
        # Create the Run Artifact ID Button
        myArtifactIDButton['state'] = 'normal'
        myArtifactIDButton['command'] = myArtifactID

        # Call Function That Creates List of DICOM Slices
        myFolder_Slices = myDICOM_Slices(myFolder_Path)

        # Call Function That Creates Stack of DICOM Images
        myFolder_Stack = myDICOM_Stack(myFolder_Slices)

        # Create Figure and Plot
        myFigure, myPlot = plt.subplots(1, 1)

        # Turn Off Tick Labels on Plot
        myPlot.axes.xaxis.set_ticks([])
        myPlot.axes.yaxis.set_ticks([])

        # Import The Stack of DICOM Images Into Plot
        myFigure_Manager = myPlot_Manager(myPlot, myFolder_Stack)

        # Embed Matlibplot into Tkinter Window
        myCanvas = FigureCanvasTkAgg(myFigure, master=myWindow)
        myCanvas.draw()

        # Connect Toolbar to Tkinter Window
        myToolbar = NavigationToolbar2Tk(myCanvas, myWindow)
        myToolbar.update()
        myCanvas.get_tk_widget().place(relx=0.5, rely=0, anchor='n')

        # Connect Scroll Event to Figure
        myWindow.bind_all("<MouseWheel>", myFigure_Manager.myScroll)
        myCanvas.mpl_connect('scroll_event', myFigure_Manager.myScroll)


# Function For myWindow_Quit Command
def myWindow_Quit():
    myWindow.quit()
    myWindow.destroy()


# Create Open File Button
myOpenButton = Button(myWindow, text="Open File", command=myFile_Open)
# Place Open File Button on Screen
myOpenButton.place(relx=0.45, rely=0.8, anchor='n')
# Create Run Artifact ID Button
myArtifactIDButton = Button(myWindow, text="Run Artifact ID", state=DISABLED)
# Place Run Artifact ID Button on Screen
myArtifactIDButton.place(relx=0.55, rely=0.8, anchor='n')
# Create Quit Button
myQuitButton = Button(myWindow, text="Quit", command=myWindow_Quit)
# Place Quit Button on Screen
myQuitButton.place(relx=0.5, rely=.85, anchor='n')
# Create Main Window Loop
myWindow.mainloop()
