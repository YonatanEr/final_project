#import the Image prediction module
from imageai.Prediction import ImagePrediction

#import numpy, PIL , Numpy
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt

print(123123)

exit(0)
#create an instance of the class Image prediction

prediction = ImagePrediction()

# set mode as resnet
prediction.setModelTypeAsResNet()
#set model path

prediction.setModelPath("model/resnet50_weights_tf_dim_ordering_tf_kernels.h5")
#load model
prediction.loadModel()
#


# set image location
image ="img/bird.png"



# compute image probabilities and predict class
predictions, probabilities = prediction.predictImage( image, result_count=5)


print(predictions)
print(probabilities)


# visualize output
x = Image.open(image)
fig,ax = plt.subplots()
ax.imshow(x)
bb = ax.set_title("Prediction: "+ predictions[0] + ",        Probability: " + str(probabilities[0]) )