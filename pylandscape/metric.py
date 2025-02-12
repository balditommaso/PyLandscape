import pickle
import os
import abc

class Metric(abc.ABC):
    """
    Class used to allow the loss landscape metrics to inherit basic methods and template
    """
    def __init__(self, name: str) -> None:       
        self.name = name
        self.results = {}
    
    
    def save_on_file(self, path: str) -> None:
        """
        Method used to save the dictionary with the results on pickle file. 

        Args:
            path (str): path to the destination directory.
        """
        path = os.path.join(path, self.name + ".pkl")
        directory, base_name = os.path.split(path)
        file_name, extension = os.path.splitext(base_name)
        dir_name = os.path.dirname(directory)
        if not os.path.exists(dir_name):
            os.makedirs(dir_name)
            
        # handle versioning of the file
        version = 1
        while os.path.exists(path):
            # Create a new file path with version number
            self.name = f"{file_name}_v{version}{extension}"
            path = os.path.join(directory, self.name)
            version += 1

        with open(path, "wb") as f:
            pickle.dump(self.results, f)
     
    
    def load_from_file(self, path: str) -> bool:
        """
        Method used to load results of metric from a pickle file.

        Args:
            path (str): path to the target pickle file.

        Returns:
            bool: True if the results have been loaded successfully.
        """
        file_path = os.path.join(path, self.name + ".pkl")
        assert os.path.isfile(file_path), f"File not found: {file_path}"
        with open(file_path, "rb") as f:
            data = pickle.load(f)

        self.results = data[self.name]
        return True
    
    
