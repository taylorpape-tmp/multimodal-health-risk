import pandas as pd
from pathlib import Path
import tarfile
import re
import io


TAR = Path("data/raw/wearable/Stanford_Wearables_data.tar")
OUT = Path("data/interim/wearable")
OUT.mkdir(parents=True, exist_ok=True)

#turn one raw subject CSV (a file object) into a clean 1-minute series:
#resample to 1-minute means (mean() skips NaN, so the 17% per-second gaps
#average away, only fully-unworn minutes stay NaN)
#drop minutes with no HR reading (watch off-wrist)
def clean_series(fileobj):
    df = pd.read_csv(fileobj)
    df.columns = [c.strip().strip('"') for c in df.columns]
    df['time'] = pd.to_datetime(df['time'], utc=True, errors='coerce')
    df = df.set_index('time').sort_index()
    for col in ['hr', 'accel_magnitude', 'skin_temp']:
        df[col] = pd.to_numeric(df[col], errors='coerce')
    per_min = df[['hr', 'accel_magnitude', 'skin_temp']].resample('1min').mean()
    per_min = per_min.dropna(subset=['hr'])
    return per_min

#primary recordings only: Basis_###.csv with no trailing 'b' 
def is_primary(member_name):
    base = member_name.split('/')[-1]
    return re.fullmatch(r"Basis_\d+\.csv", base) is not None

def main():
    summary = []
    with tarfile.open(TAR, 'r') as tar:
        members = [m for m in tar.getmembers()
                   if 'Basis_Watch_Data/' in m.name and m.name.endswith('.csv')
                   and is_primary(m.name)]
        members.sort(key=lambda m: m.name)
        print(f"{len(members)} primary subject files to process")

        for m in members:
            subject = m.name.split('/')[-1].replace('.csv', '') #e.g. 'Basis_002'
            raw_bytes = tar.extractfile(m).read() #one CSV into memory
            per_min = clean_series(io.BytesIO(raw_bytes))
            per_min.to_parquet(OUT / f"{subject}.parquet") #keep full 1-min series
            row = {"subject": subject,
                   "worn_minutes": len(per_min),
                   "worn_days": round(len(per_min) / 1440, 1)}
            summary.append(row)
            print(f"{subject}: {len(per_min):,} worn min ({row['worn_days']} days)")

    pd.DataFrame(summary).to_csv(OUT / "_wear_summary.csv", index=False)
    print(f"\ndone: {len(summary)} subjects saved to {OUT}")

if __name__ == "__main__":
    main()

#use the UTC timestamp as the index so pandas can resample by time
df['time'] = pd.to_datetime(df['time'], utc=True, errors='coerce')
df = df.set_index('time').sort_index()

#make sure the sensor columns are numeric (blanks -> NaN)
for col in ['hr', 'accel_magnitude', 'skin_temp']:
    df[col] = pd.to_numeric(df[col], errors='coerce')

#1-minute mean of each sensor, mean() skips NaN automatically
#only fully-unworn minutes stay NaN
per_min = df[['hr', 'accel_magnitude', 'skin_temp']].resample('1min').mean()
print("per-minute shape:", per_min.shape) #96k rows (66.6 days x 1440 min)
print("NaN minutes per sensor:")
print(per_min.isna().sum())
print(per_min.head())

#a minute with no HR reading = watch off-wrist
before = len(per_min)
per_min = per_min.dropna(subset=['hr'])
print(f"dropped {before - len(per_min):,} unworn minutes -> {len(per_min):,} worn minutes")
print("worn days:", round(len(per_min)/1440, 1))