# Import packages
import argparse
import numpy as np
import pickle
import tensorflow as tf
import tensorflow.keras.backend as K

# Import helpers
import utils
import models

# Parse args
parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument('imfile', metavar='imfile', type=str,
        help='A csv file that maps the images with the following format: <distorted_image_name>,<reference_image_name>\\n')
parser.add_argument('-m', '--model', type=str, default="wadiqam",
        help='The name of the model that will produce the quality estimates: one of <wadiqam, ms-wadiqam, aggregated> (defaults to wadiqam)')
parser.add_argument('-o', '--output', type=str, default='scores.pkl', 
        help='The name of the file in which the output scores will be written as a pickle object of a numpy array (defaults to scores.pkl)')
parser.add_argument('-p', '--n-patches', type=int, default=128,
        help='The number of random patches to use when computing quality estimates (defaults to 128)')
parser.add_argument('-df', '--distorted-folder', type=str, default='',
        help='The name of the folder that contains the distorted images; will be concatenated before the distorted images\' names')
parser.add_argument('-rf', '--reference-folder', type=str, default='',
        help='The name of the folder that contains the reference images; will be concatenated before the reference images\' names')

args = parser.parse_args()

# Recover lines from imfile and compute scores
with open(args.imfile, "r") as im_file:
    pairings = im_file.readlines()
    
for line in enumerate(pairings):
    if line[1][len(line[1])-1] == '\n':
        pairings[line[0]] = pairings[line[0]][:-1]

if args.model == "wadiqam":
    model_list = [models.get_wadiqam_model]
    weights = ["saved_weights/wadiqam.hdf5"]
elif args.model == "ms-wadiqam":
    model_list = [models.get_wadiqam_model_ms]
    weights = ["saved_weights/ms-wadiqam.hdf5"]
elif args.model == "aggregated":
    model_list = [models.get_wadiqam_model, models.get_wadiqam_model, models.get_wadiqam_model, models.get_wadiqam_model_ms]
    weights = ["saved_weights/wadiqam.hdf5", "saved_weights/wadiqam40.hdf5", "saved_weights/wadiqam60.hdf5", "saved_weights/ms-wadiqam.hdf5"]
else:
    print("Invalid value for --model:", args.model)
    exit()
    
scores = np.zeros((len(pairings),)).flatten()
for i in range(len(model_list)):
    is_ms = model_list[i] == models.get_wadiqam_model_ms
    generator = utils.generic_batch_generator(pairings, args.distorted_folder, args.reference_folder, args.n_patches, 32, 1, is_ms)
    model = model_list[i]()
    model.load_weights(weights[i])
    scores += model.predict_generator(generator, len(pairings)).flatten()
scores /= len(model_list)

# Save results
with open(args.output, "wb") as out_file:
    pickle.dump(scores, out_file)
print(scores)
