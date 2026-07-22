import os

def create_folder(file_path):
    try:
        if not os.path.exists(file_path):
            os.makedirs(file_path)
            print(f"Folder successfully created at: {file_path}")
        else:
            print(f"Folder already exists at: {file_path}")
    
    except PermissionError:
        print(f"Permission denied: Unable to create folder at '{file_path}'. Please check your permissions.")
    except FileNotFoundError:
        print(f"Error: Invalid path '{file_path}'. It may contain invalid characters or refer to a non-existent directory.")
    except OSError as e:
        print(f"OS error: {e}")
    except Exception as e:
        print(f"An unexpected error occurred: {e}")
