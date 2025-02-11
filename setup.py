import pandas as pd
from sklearn.model_selection import train_test_split


def main():

    wcx_df = pd.read_csv("/app/test_data/hf_wcx.csv", encoding="utf-8-sig", converters={"relevant_laws": eval})
    tax_df = pd.read_csv("/app/test_data/hf_tax.csv",  encoding="utf-8-sig", converters={"relevant_laws": eval})
    
    tax_df = tax_df.rename(columns = {"question": "ข้อหารือ", "relevant_laws": "actual_relevant_laws"})
    
    #Then, select the ratio
    ratio = 0.2
    n_samples = round(0.2*wcx_df.shape[0])

    #Then, get the stats
    relevant_law_codes = wcx_df["relevant_laws"].apply(lambda x: x[0]["law"])
    
    train, test = train_test_split(wcx_df, test_size=n_samples, random_state=42, shuffle=True, stratify=relevant_law_codes)
    
    wcx_df.to_csv("/app/test_data/hf_wcx.csv", encoding="utf-8-sig", index=False)
    tax_df.to_csv("/app/test_data/hf_tax.csv", encoding="utf-8-sig", index=False)
    test.to_csv("/app/test_data/lclm_sample.csv", encoding="utf-8-sig", index=False)
    
    


if __name__ == "__main__":
    main()