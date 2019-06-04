# For python 2 support
from __future__ import absolute_import, print_function, division, unicode_literals

# import necessary packages
import tensorflow as tf
from tensorflow.python import keras


# Import helper packages
import numpy as np
import pandas as pd
import sys, traceback

# import in-house packages
import generator_utils.generator_utilities as generator_utilities
import model_utils.person_reid_model as person_reid_model
import os_utils.os_utilities as os_utilities

def new_train_model(config: dict,):
    """
    Function to new train a model from scratch
    
    Arguments:
        config {dict} -- The configuration to train consisting of 
        {
            "no_of_epochs" : no of epochs to train
            "learning_rate" : The learning rate
            "batch_size" : The batch size to use
            "threads" : The no. of threads to use
            "gpus" : The total number of gpus to use 
            "train_data_path": The meta data for training
            "val_data_path": The meta data for validation
            "test_data_path": The meta data for testing
            "model_name" : The model name, NOTE: This will be used to save the model.
        }
    """
    tf.compat.v1.disable_eager_execution()

    # Get the configuration to train
    no_of_epochs = config["no_of_epochs"]
    learning_rate = config["learning_rate"]
    batch_size = config["batch_size"]
    threads = config["threads"]
    model_name = config["model_name"]
    no_gpus = config["gpus"]
    train_data_path = config["train_data_path"]
    val_data_path =  config["val_data_path"]
    test_data_path = config["test_data_path"]

    print("[INFO]: Using config: \n", config)
    # Set up environment
    folders = [
        "./models/checkpoints/", 
        "./output/csv_log/",
        "./models/best_model",
        "./models/saved_model",
        "./output/graphs/"
    ]
    os_utilities.make_directory(folders)

    # instantiate the new model and start train
    if no_gpus > 1:
        # strategy = tf.distribute.MirroredStrategy()
        print('[INFO] Number of GPU devices in use for training: {}'.format(no_gpus))
        
        with tf.device("/cpu:0"):
            model = person_reid_model.person_recognition_model()
        
        model = tf.keras.utils.multi_gpu_model(model, no_gpus)
        # with strategy.scope():
        #     model = person_reid_model.person_recognition_model()

        model.compile(optimizer = keras.optimizers.Adam(lr = learning_rate), loss = "categorical_crossentropy", metrics = ['accuracy'])
    
    else:
        model = person_reid_model.person_recognition_model()
        model.compile(optimizer = keras.optimizers.Adam(lr = learning_rate), loss = "categorical_crossentropy", metrics = ['accuracy'])

    # Load the generators for the data
    train_gen = generator_utilities.DataGenerator(dataset_metadata_path = train_data_path, batch_size = batch_size, dim = (299, 299), n_channels = 3, n_classes = 2, shuffle = True)
    val_gen = generator_utilities.DataGenerator(dataset_metadata_path = val_data_path, batch_size = batch_size, dim = (299, 299), n_channels = 3, n_classes = 2, shuffle = False)
    test_gen = generator_utilities.DataGenerator(dataset_metadata_path = test_data_path, batch_size = batch_size, dim = (299, 299), n_channels = 3, n_classes = 2, shuffle = False)

    # set callbacks the model
    csv_callback = keras.callbacks.CSVLogger("./output/csv_log/{0}_log.csv".format(model_name))
    tensorboard_callback = keras.callbacks.TensorBoard(log_dir = "./{0}_logs".format(model_name)) 
    checkpoint_callback = keras.callbacks.ModelCheckpoint("./models/checkpoints/{0}_checkpoint.h5".format(model_name), period = 1, save_weights_only = True)
    best_model_checkpoint_callback = keras.callbacks.ModelCheckpoint("./models/best_model/best_{0}_checkpoint.h5".format(model_name), save_best_only = True, save_weights_only = True)
    model_save_path = "./models/saved_model/{0}".format(model_name)

    # Train model
    try:
        model.fit_generator(generator = train_gen, 
                                epochs = no_of_epochs,
                                max_queue_size = 1, 
                                callbacks = [csv_callback, checkpoint_callback, best_model_checkpoint_callback, tensorboard_callback],
                                use_multiprocessing = True,
                                workers = threads,
                                validation_data = val_gen, 
                                shuffle = True)
    
    except KeyboardInterrupt:
        print("\n[INFO] Train Interrupted")
        model.save(model_save_path + "_interrupted.h5")
        del model
        sys.exit(2)

    except Exception as err:
        print("\n{CRITICAL}: Error, UnHandled Exception: ", err, "\n", traceback.print_exc())
        print("{CRITICAL}: Trying to save the model")
        model.save(model_save_path + "_error.h5")
        del model
        sys.exit(2)
        
    # Model saving
    model.save(filepath = model_save_path + ".h5")

    # Testing Results
    print("[+] Testing the model")
    loss, accuracy = model.evaluate_generator(generator = test_gen, max_queue_size = 1, use_multiprocessing = True, workers = threads, verbose = 1)
    print("[+] Test Loss: ", loss)
    print("[+] Test Accuracy: ", accuracy)


def deterred_train_model(model_path:str, resume_epoch:int, config: dict):
    """
    Function to resume a paused training model
    
    Arguments:
        model_weight_path {str} -- the weight path of the model to load and resume train
        resume_epoch {int} -- The epoch to resume
        config {dict} -- The configuration to train consisting of 
        {
            "no_of_epochs" : no of epochs to train
            "learning_rate" : The learning rate
            "batch_size" : The batch size to use
            "threads" : The no. of threads to use
            "train_data_path": The meta data for training
            "val_data_path": The meta data for validation
            "test_data_path": The meta data for testing
            "model_name" : The model name, NOTE: This will be used to save the model.
        }
    """
    
    tf.compat.v1.disable_eager_execution()

    # Get the configuration to train
    no_of_epochs = config["no_of_epochs"]
    learning_rate = config["learning_rate"]
    batch_size = config["batch_size"]
    threads = config["threads"]
    model_name = config["model_name"]
    no_gpus = config["gpus"]
    train_data_path = config["train_data_path"]
    val_data_path =  config["val_data_path"]
    test_data_path = config["test_data_path"]

    print("[INFO]: Using config: \n", config)
    # Set up environment
    folders = [
        "./models/checkpoints/", 
        "./output/csv_log/",
        "./models/best_model",
        "./models/saved_model",
        "./output/graphs/"
    ]
    os_utilities.make_directory(folders)

    # instantiate the new model and start train
    if no_gpus > 1:
        # strategy = tf.distribute.MirroredStrategy()
        print('[INFO] Number of GPU devices in use for training: {}'.format(no_gpus))
        
        with tf.device("/cpu:0"):
            model = person_reid_model.person_recognition_model()
        
        model = tf.keras.utils.multi_gpu(model, no_gpus)
        # with strategy.scope():
        #     model = person_reid_model.person_recognition_model()

        model.compile(optimizer = keras.optimizers.Adam(lr = learning_rate), loss = "categorical_crossentropy", metrics = ['accuracy'])
    
    else:
        model = person_reid_model.person_recognition_model()
        model.compile(optimizer = keras.optimizers.Adam(lr = learning_rate), loss = "categorical_crossentropy", metrics = ['accuracy'])

    # Load the generators for the data
    train_gen = generator_utilities.DataGenerator(dataset_metadata_path = train_data_path, batch_size = batch_size, dim = (299, 299), n_channels = 3, n_classes = 2, shuffle = True)
    val_gen = generator_utilities.DataGenerator(dataset_metadata_path = val_data_path, batch_size = batch_size, dim = (299, 299), n_channels = 3, n_classes = 2, shuffle = False)
    test_gen = generator_utilities.DataGenerator(dataset_metadata_path = test_data_path, batch_size = batch_size, dim = (299, 299), n_channels = 3, n_classes = 2, shuffle = False)

    # set callbacks the model
    csv_callback = keras.callbacks.CSVLogger("./output/csv_log/{0}_log.csv".format(model_name))
    tensorboard_callback = keras.callbacks.TensorBoard(log_dir = "./{0}_logs".format(model_name)) 
    checkpoint_callback = keras.callbacks.ModelCheckpoint("./models/checkpoints/{0}_checkpoint.h5".format(model_name), period = 1, save_weights_only = True)
    best_model_checkpoint_callback = keras.callbacks.ModelCheckpoint("./models/best_model/best_{0}_checkpoint.h5".format(model_name), save_best_only = True, save_weights_only = True)
    model_save_path = "./models/saved_model/{0}".format(model_name)

    # Train model
    try:
        model.fit_generator(generator = train_gen, 
                                epochs = no_of_epochs,
                                max_queue_size = 1, 
                                callbacks = [csv_callback, checkpoint_callback, best_model_checkpoint_callback, tensorboard_callback],
                                use_multiprocessing = True,
                                workers = threads,
                                validation_data = val_gen, 
                                shuffle = True, 
                                initial_epoch = resume_epoch)
    
    except KeyboardInterrupt:
        print("\n[INFO] Train Interrupted")
        model.save(model_save_path + "_interrupted.h5")
        del model
        sys.exit(2)

    except Exception as err:
        print("\n{CRITICAL}: Error, UnHandled Exception: ", err, "\n", traceback.print_exc())
        print("{CRITICAL}: Trying to save the model")
        model.save(model_save_path + "_error.h5")
        del model
        sys.exit(2)
        
    # Model saving
    model.save(filepath = model_save_path + ".h5")

    # Testing Results
    print("[+] Testing the model")
    loss, accuracy = model.evaluate_generator(generator = test_gen, max_queue_size = 1, use_multiprocessing = True, workers = threads, verbose = 1)
    print("[+] Test Loss: ", loss)
    print("[+] Test Accuracy: ", accuracy)