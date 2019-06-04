# for python 2 support
from __future__ import absolute_import, print_function, division, unicode_literals

# import necessary packages
import numpy as np
import pandas as pd
import os, random, progressbar

# NOTE: Private API
def __data_dict_generator(dataset_path: str):
    """
    Function to return a dictionary of all the dataset 
    as a dictionary of the order as such:
    {
        person: {
            index_no: total image path
        }
    }
    eg: 0001: 0001: total image path
    
    Arguments:
        dataset_path {str} -- The dataset path that has 
        all the person's image data

    Returns:
        {dict} -- A dictionary as mentioned
    """
    data_dict = dict()

    for person in os.listdir(dataset_path):
        data_dict[person] = dict()
        for image_path in os.listdir(dataset_path + person + "/"):
            index_no = image_path[7: 11]
            total_path = dataset_path + person + "/" + image_path
            if index_no not in data_dict[person].keys():
                data_dict[person][index_no] = [total_path]
            else:
                data_dict[person][index_no].append(total_path)
    
    return data_dict

def __positive_pair(data_dict: dict, person: str):
    """
    Function to make a positive pair for the person 
    given
    
    Arguments:
        data_dict {dict} -- The data dictionary of the entire dataset
        person {str} -- The person to query and make a positive pair
    
    Returns:
        {tuple} -- The data tuple of (sample1 path, sample2 path, 1)
    """
    index_no = list(data_dict[person].keys())
    random.shuffle(index_no)
    img_list1 = data_dict[person][index_no[-1]]
    img_list2 = data_dict[person][index_no[-2]]
    random.shuffle(img_list1)
    random.shuffle(img_list2)
    p1 = img_list1[-1]
    p2 = img_list2[-1]
    return (p1, p2, 1)

def __negative_pair(data_dict: dict, person: str):
    """
    Function to do the negative pair for the given 
    person
    
    Arguments:
        data_dict {dict} -- The dictionary of the dataset
        person {str} -- The person to make the negative pair
        
    Returns:
        {tuple} -- The data tuple of (sample1 path, sample2 path, 0)
    """
    person_list = list(data_dict.keys())
    person_list.remove(person)
    random.shuffle(person_list)
    counter_person = person_list[-1]
    
    index_no1 = list(data_dict[person].keys())
    random.shuffle(index_no1)
    index_no2 = list(data_dict[counter_person].keys())
    random.shuffle(index_no2)
    
    img_list1 = data_dict[person][index_no1[-1]]
    img_list2 = data_dict[counter_person][index_no2[-1]]
    random.shuffle(img_list1)
    random.shuffle(img_list2)
    p1 = img_list1[-1]
    p2 = img_list2[-1]
    
    return (p1, p2, 0)

def __save_dataset(dataset: set, output_csv_path: str):
    """
    Function to save the set of dataset in the csv path

    
    Arguments:
        dataset {set} -- The set of tuples consisting the triplet dataset
        output_csv_path {str} -- The output csv path to save
    """

    dataset_list = list(dataset)
    dataset_df = pd.DataFrame(data = dataset_list, columns = ["Sample1", "Sample2", "Label"])
    # save it
    dataset_df.to_csv(output_csv_path, index = False)


# NOTE: Public APIs
def dataset_pair_generator(dataset_path: str, output_csv_path: str, count: int, checkpoint = 0):
    """
    Function to generate pairs of data for the dataset for the model training
    upto the count given
    NOTE: MAX count = 2000000 is advisable
    
    Arguments:
        dataset_path {str} -- The dataset path of the MARS dataset
        output_csv_path {str} -- The output csv save path
        count {int} -- The total amount of dataset to make
    
    Keyword Arguments:
        checkpoint {int} -- The checkpoint argument lets you to 
        checkpoint the dataset generation for safety purpose. Mention
        the len(dataset) after which we should do regular checkpoint. set = 0
        for no checkpoint(default: {0})
    """
    if count > 2000000:
        print("[WARN] Count is too high to create hard sampled dataset")
        print("[INFO] Still trying...")
    
    print("[INFO] Generating %d samples and storing it in %s" %(count, output_csv_path))
    
    # Get the global lookup dictionary
    data_dict = __data_dict_generator(dataset_path = dataset_path)

    # Generating data
    dataset = set()
    alternator = True
    person_list = list(data_dict.keys())

    with progressbar.ProgressBar(min_value = 0, max_value = count) as bar:
        while len(dataset) < count:
            random.shuffle(person_list)
            check = len(dataset)
    
            random.shuffle(person_list)
            person = person_list[-1]
            if alternator == True:
                if len(data_dict[person].keys()) < 2:
                    continue
                dataset.add(__positive_pair(data_dict, person))
            else:
                dataset.add(__negative_pair(data_dict, person))
            
            if check < len(dataset):
                alternator = not alternator
                if checkpoint != 0:
                    if len(dataset) % checkpoint == 0:
                        output_csv_path_name = ".".join(output_csv_path.split(".")[:-1])
                        __save_dataset(dataset, output_csv_path_name + "_" + str(len(dataset)) + ".csv")
                        print("[INFO] Checkpointed at %d in %s" %(len(dataset), output_csv_path_name + "_" + str(len(dataset)) + ".csv"))
            
            bar.update(len(dataset))
    __save_dataset(dataset, output_csv_path)
    print("[INFO] Final CSV File is generated and saved at: %s" %(output_csv_path))

    