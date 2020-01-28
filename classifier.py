import tensorflow as tf
import cv2
import os
import numpy as np

os.chdir('C:\\Users\\jaide\\OneDrive\\Documents\\VSCODE\\openCV\\roi')

for roi in os.listdir():
    if(roi=='my_model.h5'):
        continue
    img = cv2.imread(roi)
    img = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
    cv2.imshow('roi',img)

    img = np.array([img],dtype=np.float32)
    img = img/255

    img = img.reshape(-1,28,28,1)
    print(img.shape)

    new_model = tf.keras.models.load_model('my_model.h5')
    pred = new_model.predict(img)
    print('number : ',np.argmax(pred,axis=1))


cv2.waitKey(0)
cv2.destroyAllWindows()

    

    
