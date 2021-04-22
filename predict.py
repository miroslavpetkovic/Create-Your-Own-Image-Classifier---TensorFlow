import tensorflow as tf
import tensorflow_hub as hub
import numpy as np
from PIL import Image
import json
import argparse


parser = argparse.ArgumentParser(description='to predict the probabilities of flowers')
parser.add_argument('image_path',help="image path")
parser.add_argument('model_path',help="model path")
parser.add_argument('--top_k',type= int ,help="top required probabilities", required = False, default = 3)
parser.add_argument('--json_file_path',
                    help="file path for a json file to map the names",
                    required = False,
                    default = None)
args = parser.parse_args()

def process_image(image) :
    image = tf.convert_to_tensor(image)
    image = tf.image.resize(image , (224,224))
    image /= 225
    image = image.numpy()
    return image

def predict(image_path, model, top_k) :
        image = Image.open(image_path)
        image_array = np.asarray(image)
        processed_image = process_image(image_array)
        processed_image_d = np.expand_dims(processed_image, axis=0)
        prediction = model.predict(processed_image_d)
        probabilities , classes = tf.math.top_k(prediction, top_k)
        return list(probabilities.numpy()[0]) , list(classes.numpy()[0])

def main (image_path , model_path , top_k, json_file_path) :
    model = tf.keras.models.load_model(model_path,custom_objects={'KerasLayer':hub.KerasLayer})
    prob , classes = predict(image_path , model , top_k)

    print(f'probabilites    = {prob}')
    print(f'classes numbers = {classes}')

    if json_file_path != None :
        with open('label_map.json', 'r') as f:
            class_names = json.load(f)
        classes_names_list = []
        for i in classes :
            classes_names_list.append(class_names[str(i +1)])
        print (f'classes names   = {classes_names_list}')
    else :
        classes_names_list = None

    return prob , classes , classes_names_list





if __name__ == '__main__' :
    main(args.image_path , args.model_path , args.top_k , args.json_file_path)