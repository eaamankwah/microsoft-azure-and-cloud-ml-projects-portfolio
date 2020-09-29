#!/usr/bin/env python
# coding: utf-8

# # Custom Training with TensorFlow in Sagemaker

# # Download Data

# In[1]:


get_ipython().run_line_magic('matplotlib', 'inline')

import os
import tarfile
import urllib
import shutil
import json
import random
import numpy as np
import tensorflow as tf
import sagemaker

from PIL import Image
from matplotlib import pyplot as plt

urls = ['http://www.robots.ox.ac.uk/~vgg/data/pets/data/images.tar.gz',
        'http://www.robots.ox.ac.uk/~vgg/data/pets/data/annotations.tar.gz']

print('Libraries imported')


# In[2]:


def download_and_extract(data_dir, download_dir):
    for url in urls:
        target_file = url.split('/')[-1]
        if target_file not in os.listdir(download_dir):
            print('Downloading', url)
            urllib.request.urlretrieve(url, os.path.join(download_dir, target_file))
            tf = tarfile.open(url.split('/')[-1])
            tf.extractall(data_dir)
        else:
            print('Already downloaded', url)

def get_annotations(file_path, annotations={}):
    
    with open(file_path, 'r') as f:
        rows = f.read().splitlines()

    for i, row in enumerate(rows):
        image_name, _, _, _ = row.split(' ')
        class_name = image_name.split('_')[:-1]
        class_name = '_'.join(class_name)
        image_name = image_name + '.jpg'
        
        annotations[image_name] = 'cat' if class_name[0] != class_name[0].lower() else 'dog'
    
    return annotations


# In[3]:


if not os.path.isdir('data'):
    os.mkdir('data')

download_and_extract('data', '.')


# # Dataset for Training

# In[4]:


annotations = get_annotations('data/annotations/trainval.txt')
annotations = get_annotations('data/annotations/test.txt', annotations)

total_count = len(annotations.keys())
print('Total examples', total_count)


# In[5]:


next(iter (annotations.items()))


# In[6]:


classes = ['cat', 'dog']
sets = ['train', 'validation']
root_dir = 'custom_data'

if not os.path.isdir(root_dir):
    os.mkdir(root_dir)
    
for set_name in sets:
    if not os.path.isdir(os.path.join(root_dir, set_name)):
        os.mkdir(os.path.join(root_dir, set_name))
    for class_name in classes:
        folder = os.path.join(root_dir, set_name, class_name)
        if not os.path.isdir(folder):
            os.mkdir(folder)


# Copy the files to correct set/ class folders

# In[7]:


for image, class_name in annotations.items():
    target_set = 'validation' if random.randint(0, 99) < 20 else 'train'
    target_path = os.path.join(root_dir, target_set, class_name, image)
    shutil.copy(os.path.join('data/images/', image), target_path)


# In[8]:


sets_counts = {
    'train': 0,
    'validation': 0
}

for set_name in sets:
    for class_name in classes:
        path = os.path.join(root_dir, set_name, class_name)
        count = len(os.listdir(path))
        print(path, 'has', count, 'images')
        sets_counts[set_name] += count

print(sets_counts)


# # Training Script - Create Model

# In[9]:


get_ipython().run_cell_magic('writefile', 'train.py', "\nimport tensorflow as tf\nimport argparse\nimport os\nimport json\n\ndef create_model():\n    model = tf.keras.models.Sequential([\n        tf.keras.applications.mobilenet_v2.MobileNetV2(include_top=False, weights='imagenet',\n                                                       pooling='avg', input_shape=(128, 128, 3)),\n        tf.keras.layers.Dropout(0.5),\n        tf.keras.layers.Dense(1, activation='sigmoid')\n    ])\n    \n    model.layers[0].trainable = False\n    model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])\n    return model")


# # Training Script - Data Generators

# In[10]:


get_ipython().run_cell_magic('writefile', '-a train.py', "\ndef create_data_generators(root_dir, batch_size):\n    train_data_generator = tf.keras.preprocessing.image.ImageDataGenerator(\n        preprocessing_function=tf.keras.applications.mobilenet_v2.preprocess_input,\n        horizontal_flip=True,\n        zoom_range=[0.8, 1.2],\n        rotation_range=20\n    ).flow_from_directory(\n        os.path.join(root_dir, 'train'),\n        target_size=(128, 128),\n        batch_size=batch_size,\n        class_mode='binary'\n    )\n    \n    val_data_generator = tf.keras.preprocessing.image.ImageDataGenerator(\n        preprocessing_function=tf.keras.applications.mobilenet_v2.preprocess_input\n    ).flow_from_directory(\n        os.path.join(root_dir, 'validation'),\n        target_size=(128, 128),\n        batch_size=batch_size,\n        class_mode='binary'\n    )\n    \n    return train_data_generator, val_data_generator")


# # Training Script - Putting it Together

# In[11]:


get_ipython().run_cell_magic('writefile', '-a train.py', "\nif __name__ =='__main__':\n\n    parser = argparse.ArgumentParser()\n\n    # hyperparameters sent by the client are passed as command-line arguments to the script.\n    parser.add_argument('--epochs', type=int, default=3)\n    parser.add_argument('--batch_size', type=int, default=16)\n    parser.add_argument('--steps', type=int, default=int(5873/16))\n    parser.add_argument('--val_steps', type=int, default=(1476/16))\n\n    # input data and model directories\n    parser.add_argument('--model_dir', type=str)\n    parser.add_argument('--sm-model-dir', type=str, default=os.environ.get('SM_MODEL_DIR'))\n    parser.add_argument('--train', type=str, default=os.environ.get('SM_CHANNEL_TRAINING'))\n\n    args, _ = parser.parse_known_args()\n\n    local_output_dir = args.sm_model_dir\n    local_root_dir = args.train\n    batch_size = args.batch_size\n    \n    model = create_model()\n    train_gen, val_gen = create_data_generators(local_root_dir, batch_size)\n    \n    _ = model.fit(\n        train_gen,\n        epochs=args.epochs,\n        steps_per_epoch=args.steps,\n        validation_data=val_gen,\n        validation_steps=args.val_steps\n    )\n    \n    model.save(os.path.join(local_output_dir, 'model', '1'))")


# # Upload Dataset to S3

# In[12]:


sagemaker_session = sagemaker.Session()
role = sagemaker.get_execution_role()
bucket_name = 'objectpetsdata'

print('Uploading to S3..')
s3_data_path = sagemaker_session.upload_data(path=root_dir, bucket=bucket_name, key_prefix='data')

print('Uploaded to', s3_data_path)


# # Train with TensorFlow Estimator

# In[13]:


from sagemaker.tensorflow import TensorFlow

pets_estimator = TensorFlow(
    entry_point='train.py',
    role=role,
    train_instance_count=1,
    #train_instance_type='ml.p2.xlarge',
    train_instance_type='ml.p3.2xlarge',
    framework_version='2.1.0',
    py_version='py3',
    output_path='s3://objectpetsdata/'
)


# In[14]:


pets_estimator.fit(s3_data_path)


# # Deploy TensorFlow Model

# In[15]:


pets_predictor = pets_estimator.deploy(initial_instance_count=1, instance_type='ml.m4.xlarge')
print('\nModel Deployed!')


# # Final Predictions

# In[16]:


cat_dir = 'custom_data/validation/cat/'
cat_images = [os.path.join(cat_dir, x) for x in os.listdir(cat_dir)]
print(cat_images[0])

dog_dir = 'custom_data/validation/dog/'
dog_images = [os.path.join(dog_dir, x) for x in os.listdir(dog_dir)]
print(dog_images[0])


# In[17]:


def get_pred(image_path):
    img = tf.keras.preprocessing.image.load_img(image_path, target_size=(128, 128))
    img = tf.keras.preprocessing.image.img_to_array(img)
    img = tf.keras.applications.mobilenet_v2.preprocess_input(img)
    img = np.expand_dims(img, axis=0)

    results = pets_predictor.predict(img)
    return results


# In[18]:


image_path = cat_images[0]
results = get_pred(image_path)

print(results)


# In[19]:


class_id = int(np.squeeze(results['predictions']) > 0.5)
print('Predicted class_id:', class_id, 'with class_name:', classes[class_id])


# # Delete Model Endpoint

# In[20]:


sagemaker_session.delete_endpoint(pets_predictor.endpoint)


# In[ ]:




