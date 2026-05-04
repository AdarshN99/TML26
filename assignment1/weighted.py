from pathlib import Path
import pandas as pd

# config
BASE = Path(__file__).parent
INPUT1 = BASE / "submission_rmia_64.csv"
INPUT2 = BASE / "submission_lira_32.csv"
OUTPUT_CSV = BASE / "submission_lira_rmia_32_64.csv"

# weights (adjust as needed)
W1 = 0.5
W2 = 0.5

def main():
    # load csvs
    df1 = pd.read_csv(INPUT1)
    df2 = pd.read_csv(INPUT2)

    # merge on id
    df = df1.merge(df2, on='id', suffixes=('_1', '_2'))

    # weighted sum
    df['score'] = W1 * df['score_1'] + W2 * df['score_2']

    # keep only required columns
    output_df = df[['id', 'score']]

    # save
    output_df.to_csv(OUTPUT_CSV, index=False)

    print(f"Saved blended submission to {OUTPUT_CSV}")

if __name__ == "__main__":
    main()
