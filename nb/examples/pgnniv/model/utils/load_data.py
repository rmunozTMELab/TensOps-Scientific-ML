import os
import pickle

def load_data(file_path):
    try:
        if not os.path.exists(file_path):
            print(f"File not found: {file_path}")
            return None
        with open(file_path, 'rb') as f:
            data = pickle.load(f)
            print(f"Data successfully loaded from: {file_path}")
            return data

    except FileNotFoundError:
        print(f"Error: The file '{file_path}' was not found.")
    except pickle.UnpicklingError:
        print("Error: Failed to unpickle the data. The file may be corrupted or not a valid pickle file.")
    except Exception as e:
        print(f"An unexpected error occurred: {e}")
    
