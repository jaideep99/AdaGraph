import tensorflow as tf
import cv2
import numpy as np

def predict_digitnn(img):

    img = np.array([img],dtype=np.float32)
    img = img/255

    img = img.reshape(-1,28,28,1)
    
    new_model = tf.keras.models.load_model('my_model.h5')
    pred = new_model.predict(img)
    pred = np.argmax(pred,axis=1)[0]
    return pred
    

    
