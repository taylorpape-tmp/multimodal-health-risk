import pandas as pd
from pathlib import Path

#import Zhou clinical data
raw = Path("data/raw/clinical")
files = {
    "between_subject": "individual_molecule_comparison.xlsx",
    "within_subject":  "group_molecule_comparison.xlsx", 
}

def load(fname):
    xl = pd.ExcelFile(raw / fname)
    sheet = xl.sheet_names[0]
    return pd.read_excel(xl, sheet_name=sheet), sheet

def check(df):
    #missing values
    n_missing = int(df.isna().sum().sum())
    
    #r_vals
    r_cols = [c for c in df.columns if c.startswith("cor_r")]
    r_vals = df[r_cols]
    r_bad = int(((r_vals <-1) | (r_vals > 1)).sum().sum())
    
    #p_vals
    p_cols = [c for c in df.columns if "p" in c.lower() and "cor_char" not in c]
    p_cols = [c for c in p_cols if c.startswith(("cor_p", "adj.p", "LMM", "Inter"))]
    p_vals = df[p_cols]
    p_bad = int(((p_vals < 0) | (p_vals > 1)).sum().sum())
    
    return {"missing": n_missing, "r_cols": r_cols, "r_out_of_range": r_bad, "p_cols": p_cols, "p_out_of_range": p_bad}

for name, fname, in files.items():
    df, sheet = load(fname)
    res = check(df)
    print(f"\n{name} ({fname}, sheet {sheet})")
    print(f"shape: {df.shape[0]:,} x {df.shape[1]}")
    print(f"missing values: {res['missing']}")
    print(f"r columns {res['r_cols']}: {res['r_out_of_range']} out of [-1, 1]")
    print(f"p columns {res['p_cols']}: {res['p_out_of_range']} out of [0, 1]")
    ok = res["missing"] == 0 and res["r_out_of_range"] == 0 and res["p_out_of_range"] == 0

