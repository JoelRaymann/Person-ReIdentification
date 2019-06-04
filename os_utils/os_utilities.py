import os

def make_directory(list_of_directories: list):
    """
    Function to make all directories mentioned in the list of directories
    
    Arguments:      
        list_of_directories {list} -- A list of all directories to make
    """
    for directory in list_of_directories:
        try:
            os.makedirs(directory)
        except OSError:
            print("{WARN} Directory: %s already exist" %(directory))
        except Exception as err:
            print("{ERROR}: ", err)
            