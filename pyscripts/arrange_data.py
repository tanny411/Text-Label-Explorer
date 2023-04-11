import os
import pandas as pd

pd.read_csv("data/embeddings_w_topics.csv").to_csv("public/embeddings_new.csv", index=False)
pd.read_csv("data/top_5_matches.csv").to_csv("public/top_5_matches.csv", index=False)


for filename in os.scandir("data/topic_models_all_words"):
    file = filename.path
    pd.read_csv(file).to_csv(f"public/text_label_explorer/{filename}", index=False)

for filename in os.scandir("data/data_topics_compressed"):
    file = filename.path
    pd.read_csv(file).to_csv(f"public/top_topics/top_topics+{filename}", index=False)