# scripts/make_test_split.py
import sys
import pandas as pd
from sklearn.model_selection import train_test_split

INPUT = "data/loan.csv"
TRAIN_OUT = "data/train.csv"
TEST_OUT = "data/test.csv"
TEST_SIZE = 0.2
RANDOM_STATE = 42

def main():
    print("Reading", INPUT)
    df = pd.read_csv(INPUT)

    # strip spaces around headers
    df.columns = [c.strip() for c in df.columns]

    print("Full dataset shape:", df.shape)
    print("Columns:", df.columns.tolist())

    if "loan_status" not in df.columns:
        print("ERROR: 'loan_status' column not found in", INPUT)
        sys.exit(1)

    train, test = train_test_split(
        df,
        test_size=TEST_SIZE,
        random_state=RANDOM_STATE,
        stratify=df["loan_status"]
    )
    train.to_csv(TRAIN_OUT, index=False)
    test.to_csv(TEST_OUT, index=False)
    print("Wrote", TRAIN_OUT, "->", train.shape)
    print("Wrote", TEST_OUT, "->", test.shape)

if __name__ == "__main__":
    main()
