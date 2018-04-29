import sys,io
import json
import numpy
from PIL import Image

#Function for saving image annotation data as a JSON file
def buildLabelJSON(labelList, image, name):
    #Initialise Dictionaries
    labelDict = {}
    imgDict = {}

    #Convert Label Array to Dictionary
    for i in range(0,20):
        labelDict[labelList[1][i].getRgb()[:3]] = labelList[0][i]

    #Convert Image Data to array and build dictionary between pixels and labels
    pixeldata = list(image.getdata())
    data = numpy.asarray(image)
    data = data[:,:,:3]
    for x, y,z in numpy.ndindex(data.shape):
        imgDict[str(x)+","+str(y)] = labelDict[tuple(data[x,y])]

    #Save as JSON
    with io.open((name+".json"),'w',encoding="utf-8") as outfile:
      outfile.write(unicode(json.dumps(imgDict, ensure_ascii=False)))
