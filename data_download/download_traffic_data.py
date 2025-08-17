import pandas as pd
import urllib.request
import urllib.error
import zipfile
import os

def main():
    path = input("Enter the folder path where traffic data should be saved: ").strip()
    if not os.path.isdir(path):
        print("Invalid folder path. Exiting.")
        return

    file_path = input("Enter the full path to nuts3.xls: ").strip()
    if not os.path.isfile(file_path):
        print("Invalid file path. Exiting.")
        return

    url = "https://opentransportmap.info/download/nuts-3/"
    df = pd.read_excel(file_path)
    ids = df['NUTS 3 ID (2010)'].dropna().values

    for i in ids:
        print(f"downloading {i}...")
        link = url + i + "/"

        try:
            response = urllib.request.urlopen(link)
            if response.status == 200:
                zip_path = os.path.join(path, i + ".zip")
                urllib.request.urlretrieve(link, zip_path)
                with zipfile.ZipFile(zip_path, "r") as zip_ref:
                    zip_ref.extractall(os.path.join(path, i))
                os.remove(zip_path)
                print(f"download and extraction completed for {i}")
            else:
                print(f"No file found for {i}: Status code {response.status}")
        except urllib.error.HTTPError as e:
            print(f"HTTPError for {i}: {e.code} - {e.reason}")
        except urllib.error.URLError as e:
            print(f"URLError for {i}: {e.reason}")
        except Exception as e:
            print(f"An error occurred for {i}: {str(e)}")

if __name__ == "__main__":
    main()
