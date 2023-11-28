import requests
import zipfile
import io
import os


def download_data():
    # Set the URL for the zip file
    url = "https://github.com/karoldvl/ESC-50/archive/master.zip"

    # Set the directory and folder names
    download_directory = "your_download_directory"
    extract_folder = "data"

    # Create the download directory if it doesn't exist
    os.makedirs(download_directory, exist_ok=True)

    # Check if the "data" folder is empty
    data_folder = os.path.join(download_directory, extract_folder)
    if not os.listdir(data_folder):
        # Download the zip file
        response = requests.get(url)
        zip_file = zipfile.ZipFile(io.BytesIO(response.content))

        # Extract the contents to the specified folder
        zip_file.extractall(data_folder)

        # Close the zip file
        zip_file.close()
        print("Download and extraction complete.")
    else:
        print("Data folder is not empty. Skipping download.")


if __name__ == "__main__":
    download_data()
