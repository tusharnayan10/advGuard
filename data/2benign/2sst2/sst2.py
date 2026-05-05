from datasets import load_dataset
import pandas as pd

def export_sst2_sentences(split="train", output_file="sst2_sentences.csv"):

    print(f"Loading stanfordnlp/sst2 ({split})...")
    ds = load_dataset("stanfordnlp/sst2", split=split)

    sentences = [row["sentence"] for row in ds]

    df = pd.DataFrame({"sentence": sentences})

    df.to_csv(output_file, index=False)
    print(f"Saved: {output_file}")


if __name__ == "__main__":
    export_sst2_sentences(split="train", output_file="sst2_train_sentences.csv")
    export_sst2_sentences(split="validation", output_file="sst2_validation_sentences.csv")
    export_sst2_sentences(split="test", output_file="sst2_test_sentences.csv")
