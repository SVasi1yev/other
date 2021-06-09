import pandas as pd
from pathlib import Path
import os


def crop(input_df, N):
    bool_df = input_df.applymap(lambda x: len(x.split()) > N)
    idx = ~bool_df.sum(axis=1).apply(bool)
    return input_df[idx]


if __name__ == "__main__":
    N = 30
    sep = ","
    input_path = Path("../../dialog_parser_for_Simpsons/simpsons_homer_context_1.csv")
    output_path = os.path.splitext(input_path)[0] + "_cropped_{}".format(N) + os.path.splitext(input_path)[1]

    input_df = pd.read_csv(input_path, sep=sep, header=0)
    input_df = input_df.dropna()
    output_df = crop(input_df, N)
    output_df.to_csv(output_path, sep=sep, index=False)
