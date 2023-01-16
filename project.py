import numpy as np      # library for mathematical matrix operations
import matplotlib.pyplot as plt     # library for visualization  
from matplotlib import image
import tensorflow as tf
import keras
import pathlib
from pathlib import Path
import PIL
from PIL import Image
import os
#print("TensorFlow version:", tf.__version__)

def plot_image(i, predictions_array, true_label, img):
    true_label, img = true_label[i], img[i]
    plt.grid(False)
    plt.xticks([])
    plt.yticks([])

    plt.imshow(img, cmap=plt.cm.binary)

    predicted_label = np.argmax(predictions_array)
    if predicted_label == true_label:
        color = 'blue'
    else:
        color = 'red'

    plt.xlabel("{} {:2.0f}% ({})".format(class_name[predicted_label],
                                    100*np.max(predictions_array),
                                    class_name[true_label]),
                                    color=color)

def plot_value_array(i, predictions_array, true_label):
    true_label = true_label[i]
    plt.grid(False)
    plt.xticks(range(10))
    plt.yticks([])
    thisplot = plt.bar(range(10), predictions_array, color="#777777")
    plt.ylim([0, 1])
    predicted_label = np.argmax(predictions_array)

    thisplot[predicted_label].set_color('red')
    thisplot[true_label].set_color('blue')

def createDataSet(dir):
    dataList = []
    labelList = []
    #labelList = np.array([])   #you can very well use this too.

    dataSetDir = Path(dir)
    #data_directory = tf.keras.utils.get_file(origin=data_path, untar=True)
    #data = tf.keras.utils.image_dataset_from_directory(data_directory)
    i = -1

    allFolders = os.listdir(dataSetDir)
    for curFolder in allFolders:
        i = i + 1
        curDataPath = dir + "/" + curFolder
        curDataDir = Path(curDataPath)
        allData = os.listdir(curDataDir)
        #print(curFolder)
    
        for curData in allData:
            #print(curData)
            #image = Image.open(curData, curDataPath)       #a differnt way of reading an image
            data = image.imread((curDataPath+"/"+curData), "PNG") 
            
            #print(data)
            #plt.imshow(data)
            #plt.show()
            #plt.close()
            dataList.append(data)
            labelList.append(i)
            #labelList = np.append(labelList, i)

    dataList = np.array(dataList)
    labelList = np.array(labelList)    
    #print(dataList)
    #print(labelList)

    return dataList, labelList


'''
trainingSet = []

testdatapath = "D:/Courses/CS340/Project/Testdata/English/Fnt/Sample001"
#data_path = Path(testdatapath)
img = image.imread("img001-00001.png", testdatapath)
#image = Image.open("img001-00001.png")
#image.show()
print(img)
#plt.imshow(image)
#plt.show()

trainingSet.append(img)

print( "train data : ")
print(trainingSet)
'''

class_name = ["0", "1", "2", "3", "4", "5", "6", "7", "8", "9"]
trainigDataPath = "D:/Courses/CS340/Project/TrainingData"
testingDataPath = "D:/Courses/CS340/Project/TestingData"

'''
dataFoldersDir = Path(testdatapath)
#data_directory = tf.keras.utils.get_file(origin=data_path, untar=True)
#data = tf.keras.utils.image_dataset_from_directory(data_directory)
trainingSet = []
trainigLabel = np.array([])
i = -1
allSampleFolder = os.listdir(dataFoldersDir)
for curSample in allSampleFolder:
    i = i + 1
    imgPath = testdatapath + "/" + curSample
    imgDataDir = Path(imgPath)
    allImage = os.listdir(imgDataDir)
    print(curSample)
   
    for curImage in allImage:
        #imgName = curImage
        #print(curImage)
        
        #image = Image.open(imgName, imgPath)
        img = image.imread((imgPath+ "/"+ curImage), "PNG") 
        #print(img)
        
        #print(image.imread(curImage, imgPath))
        #plt.imshow(img)
        #plt.show()
        trainingSet.append(img)
        trainigLabel = np.append(trainigLabel, i)
        #plt.close()
    
print(trainigLabel.size)
trainingSet = np.array(trainingSet)
print(trainingSet)
'''
trainingSet, trainigLabel = createDataSet(trainigDataPath)
testingSet, testingLabel = createDataSet(testingDataPath)

#print(trainingSet)
#print(trainigLabel)

model = tf.keras.Sequential([
    tf.keras.layers.Flatten(input_shape=(128, 128)),
    tf.keras.layers.Dense(128, activation='relu'),
    tf.keras.layers.Dense(10)
])

model.compile(optimizer='adam', loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True), metrics=['accuracy'])

model.fit(trainingSet, trainigLabel, epochs=10)

#test_loss, test_acc = model.evaluate(testingSet, testingLabel, verbose=2)
#print('Test accuracy :', test_acc)
#print('Test loss : ', test_loss)

predictionModel = tf.keras.Sequential([model, tf.keras.layers.Softmax()])
predictions = predictionModel.predict(testingSet)

#print(predictions[0])

#print(np.argmax(predictions[0]))       # predicted label
#print(testingLabel[0])                 # test label

#### Ploting the predictions
num_rows = 5
num_cols = 3
num_images = num_rows*num_cols
plt.figure(figsize=(2*2*num_cols, 2*num_rows))
for i in range(num_images):
    plt.subplot(num_rows, 2*num_cols, 2*i+1)

    #review the plot_image function above
    plot_image(i, predictions[i], testingLabel, testingSet)

    plt.subplot(num_rows, 2*num_cols, 2*i+2)

    #review the plot_value_array function above
    plot_value_array(i, predictions[i], testingLabel)
plt.tight_layout()
plt.show()