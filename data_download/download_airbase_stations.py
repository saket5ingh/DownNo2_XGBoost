import pandas as pd
import os
import requests

def main():
    file_path = input("Enter the full path to Airbase_links.txt: ").strip()
    if not os.path.isfile(file_path):
        print("Invalid file path. Exiting.")
        return

    path = "."
    linklist = pd.read_csv(file_path, header=None, dtype=str)

    for link in linklist.loc[:, 0]:
        file_name = link.split("/")[-1]
        year = link[-19:-15]
        target_folder = os.path.join(path, "data", "Airbase", year)
        os.makedirs(target_folder, exist_ok=True)
        target_file = os.path.join(target_folder, file_name)

        if not os.path.isfile(target_file):
            print(f"Downloading {link} to {target_file}")
            response = requests.get(link)
            if response.status_code == 200:
                with open(target_file, "wb") as f:
                    f.write(response.content)
            else:
                print(f"Failed to download {link}. Status code: {response.status_code}")

if __name__ == "__main__":
    main()
