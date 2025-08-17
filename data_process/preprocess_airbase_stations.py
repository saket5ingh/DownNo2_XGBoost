import os
import glob
import pandas as pd


def load_metadata(root):
    file = os.path.join(root, 'input', 'AIRBASE', 'metadata_AIRBASE.csv')
    df = pd.read_csv(file)
    return df[df['AirPollutant'] == 'NO2']


def get_roi_filters(df):
    bROI1 = (df.Longitude > 6) & (df.Longitude < 12) & (df.Latitude > 42) & (df.Latitude < 48)
    bROI2 = (df.Longitude > 2) & (df.Longitude < 8) & (df.Latitude > 48) & (df.Latitude < 54)
    return df[bROI1], df[bROI2]


def process_airbase_files(root, no2_ROI1, no2_ROI2):
    ROIlist = {"no2_ROI1": no2_ROI1, "no2_ROI2": no2_ROI2}

    datafiles = os.path.join(root, 'data', 'AIRBASE', '*2018_timeseries.csv')
    filenames = glob.glob(datafiles)

    for f in filenames:
        try:
            data = pd.read_csv(
                f,
                usecols=[4, 10, 11, 12, 13, 14, 15, 16],
                keep_date_col=True,
                encoding='utf-16'
            )
        except Exception as e:
            print(f"Encoding/reading error for {f}: {e}")
            continue

        stnname = data['AirQualityStationEoICode'].iloc[0]
        filename = os.path.basename(f)

        stnfiles = sorted(glob.glob(os.path.join(root, 'data', 'AIRBASE', filename[:10] + '*.csv')))

        if len(stnfiles) == 2:
            df1 = pd.read_csv(stnfiles[0], usecols=[4, 10, 11, 12, 13, 14, 15, 16], encoding='utf-16')
            df2 = pd.read_csv(stnfiles[1], usecols=[4, 10, 11, 12, 13, 14, 15, 16])
            df = pd.concat([df1, df2])

            newfilename = f"{stnname}_2018_2019.csv"
            saved = False

            for ROI, roidf in ROIlist.items():
                if (roidf.AirQualityStationEoICode == stnname).any():
                    outdir = os.path.join(root, 'data', 'AIRBASE', ROI)
                    os.makedirs(outdir, exist_ok=True)
                    df.to_csv(os.path.join(outdir, newfilename), index=False)
                    print(f"Saved {stnname} to {ROI}")
                    saved = True
                    break

            if not saved:
                outdir = os.path.join(root, 'data', 'AIRBASE', 'elsewhere')
                os.makedirs(outdir, exist_ok=True)
                df.to_csv(os.path.join(outdir, newfilename), index=False)
                print(f"Saved {stnname} to elsewhere")

        else:
            print(f"{stnname} :: number of files: {len(stnfiles)}")


def main():
    root = input("Enter the root folder path: ").strip()
    if not os.path.isdir(root):
        print("Invalid root folder path. Exiting.")
        return

    no2_airbase = load_metadata(root)
    no2_ROI1, no2_ROI2 = get_roi_filters(no2_airbase)
    process_airbase_files(root, no2_ROI1, no2_ROI2)


if __name__ == "__main__":
    main()
