import os

def create_folder(path):
    """
    Create a folder at the specified path if it doesn't already exist.

    Args:
        path (str): The path of the folder to create.
    """
    try:
        if not os.path.exists(path):
            os.makedirs(path)
            print(f"Folder created at: {path}")
        else:
            print(f"Folder already exists at: {path}")
    except Exception as e:
        print(f"An error occurred while creating the folder: {e}")