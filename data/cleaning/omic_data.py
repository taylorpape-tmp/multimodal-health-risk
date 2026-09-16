import pandas as pd
from pathlib import Path
import re

raw = Path("data/raw/clinical")
workbook = raw / "time_molecule_comparison.xlsx"

#sheets we want to keep
wanted = ["S1_Subjects", "S9_ISIRassoc.", "S8_SSPGassoc.", "S4_HealthyIQR"]

def load_sheets(workbook, wanted):
    xl = pd.ExcelFile(workbook)
    sheets = {}
    for name in wanted:
        sheets[name] = pd.read_excel(xl, sheet_name = name)
    return sheets

sheets = load_sheets(workbook, wanted)
for name, df in sheets.items():
    print(f"{name:20s}  {df.shape[0]:>6} rows x {df.shape[1]:>4} cols")
    
#patient columns are Z-codes
def patient_cols(df):
    return [c for c in df.columns if re.fullmatch(r"Z[A-Z0-9]{6}", str(c))]


def split_sheet(df, label_name):
    pts = patient_cols(df)  #the Z-code columns
    row_labels = df.iloc[:, 0] #col 0 = IRIS/analyte names

    #row 0 is the target, rows 1+ are analytes
    labels = df.loc[row_labels == label_name, pts].iloc[0] #Series: patient -> IRIS/SSPG
    analytes = df.loc[row_labels != label_name, [df.columns[0]] + pts]

    #make analyte name the index, keep only patient columns -> analytes x patients
    analytes = analytes.set_index(df.columns[0])
    return labels, analytes

s9 = sheets["S9_ISIRassoc."]
labels9, analytes9 = split_sheet(s9, "IRIS")
print("label vector:", labels9.shape, labels9.head(3).to_dict())
print("analyte matrix:", analytes9.shape)   

#missing
miss_per_analyte = analytes9.isna().mean(axis=1)   #fraction of patients missing, per analyte
miss_per_patient = analytes9.isna().mean(axis=0)   #fraction of analytes missing, per patient

print("total NaN cells:", int(analytes9.isna().sum().sum()))
print("analytes >50% missing:", int((miss_per_analyte > 0.5).sum()))
print("analytes >20% missing:", int((miss_per_analyte > 0.2).sum()))
print("worst 5 analytes:")
print(miss_per_analyte.sort_values(ascending=False).head(5))
print("\npatients >50% missing:", int((miss_per_patient > 0.5).sum()))
     
#which patients
bad = miss_per_patient[miss_per_patient > 0.5].sort_values(ascending=False)
print("patients >50% missing:")
print(bad)
print("\nfull distribution of per-patient missingness:")
print(miss_per_patient.describe()) 

#drop patients missing >50% of analytes
#median-impute rest
keep = miss_per_patient[miss_per_patient <= 0.5].index      
analytes9_clean = analytes9[keep]                    

#median per analyte (row-wise): fill with median
analytes9_clean = analytes9_clean.apply(lambda row: row.fillna(row.median()), axis=1)

#drop those patients from the label vector so labels and features stay aligned
labels9_clean = labels9[keep]
print("NaN cells remaining:", int(analytes9_clean.isna().sum().sum()))

#Sheet 8
s8 = sheets["S8_SSPGassoc."]
labels8, analytes8 = split_sheet(s8, "SSPG")

miss_per_patient8 = analytes8.isna().mean(axis=0)
keep8 = miss_per_patient8[miss_per_patient8 <= 0.5].index
analytes8_clean = analytes8[keep8].apply(lambda r: r.fillna(r.median()), axis=1)
labels8_clean = labels8[keep8]

print("S8:", analytes8.shape, "->", analytes8_clean.shape, "| NaN:", int(analytes8_clean.isna().sum().sum()))
print("labels:", labels8_clean.shape, "| SSPG NaN in label row:", int(labels8_clean.isna().sum()))

#save cleaned matrices to data/interim fro SQL
#transpose -> patients as rows, analytes as columns, label as first column
interim = Path("data/interim")
interim.mkdir(parents=True, exist_ok=True)

def save_clean(analytes, labels, label_name, path):
    X = analytes.T  #patients become rows, analytes become columns
    X.insert(0, label_name, labels) #first column = the target
    X.index.name = "SubjectID"
    X.to_csv(path)
    print(f"{path.name}: {X.shape[0]} patients x {X.shape[1]-1} analytes (+{label_name})")

save_clean(analytes9_clean, labels9_clean, "IRIS", interim / "omics_S9_isir_clean.csv")
save_clean(analytes8_clean, labels8_clean, "SSPG", interim / "omics_S8_sspg_clean.csv")