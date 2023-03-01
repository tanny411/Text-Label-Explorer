import os
import gzip
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

# for Universal Sentence Encoder (USE)
import tensorflow as tf
import tensorflow_hub as hub

# for Dimension reduction
from umap import UMAP
from pymde import preserve_neighbors

# sklearn methods for dimension reduction
from sklearn.decomposition import PCA, KernelPCA, TruncatedSVD, FactorAnalysis
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.ensemble import RandomTreesEmbedding
from sklearn.manifold import Isomap, LocallyLinearEmbedding, MDS, SpectralEmbedding, TSNE
from sklearn.neighbors import NeighborhoodComponentsAnalysis
from sklearn.pipeline import make_pipeline
from sklearn.random_projection import SparseRandomProjection

    # "PyMDE": (
    #     lambda x: preserve_neighbors(x).embed()
    # ),

n_neighbors = 30
embedding_functions = {
    "PCA": (
        lambda x, y: PCA(n_components=2).fit_transform(x)
    ),
    "KernelPCA": (
        lambda x, y: KernelPCA(n_components=2, kernel='rbf').fit_transform(x) # ['linear', 'poly', 'rbf', 'sigmoid', 'cosine']
    ),
    "UMAP": (
        lambda x, y: UMAP(2).fit_transform(x)
    ),
    "Random projection": (
        lambda x, y: SparseRandomProjection(
            n_components=2, random_state=42).fit_transform(x)
    ),
    "Truncated SVD": (
        lambda x, y: TruncatedSVD(n_components=2).fit_transform(x)
    ),
    "Linear Discriminant Analysis": (
        lambda x, y: LinearDiscriminantAnalysis(n_components=2).fit_transform(x, y)
    ),
    "Isomap": lambda x, y: (
        Isomap(n_neighbors=n_neighbors, n_components=2).fit_transform(x)
    ),
    "Standard LLE": (
        lambda x, y: LocallyLinearEmbedding(
            n_neighbors=n_neighbors, n_components=2, method="standard").fit_transform(x)
    ),
    "Modified LLE": (
        lambda x, y: LocallyLinearEmbedding(
            n_neighbors=n_neighbors, n_components=2, method="modified").fit_transform(x)
    ),
    "Hessian LLE": (
        lambda x, y: LocallyLinearEmbedding(
            n_neighbors=n_neighbors, n_components=2, method="hessian").fit_transform(x)
    ),
    "LTSA LLE": (
        lambda x, y: LocallyLinearEmbedding(
            n_neighbors=n_neighbors, n_components=2, method="ltsa").fit_transform(x)
    ),
    "MDS": (
        lambda x, y: MDS(n_components=2, n_init=1, max_iter=120,
                      n_jobs=2).fit_transform(x)
    ),
    "Random Trees": (
        lambda x, y: make_pipeline(
            RandomTreesEmbedding(n_estimators=200, max_depth=5, random_state=0),
            TruncatedSVD(n_components=2),
        ).fit_transform(x)
    ),
    "Spectral": (
        lambda x, y: SpectralEmbedding(
            n_components=2, random_state=0, eigen_solver="arpack").fit_transform(x)
    ),
    "t-SNE": (
        lambda x, y: TSNE(n_components=2, n_iter=500, n_iter_without_progress=150,
                       n_jobs=2, random_state=0, learning_rate="auto", init="pca").fit_transform(x)
    ),
    "NCA": (
        lambda x, y: NeighborhoodComponentsAnalysis(
            n_components=2, init="pca", random_state=0).fit_transform(x, y)
    ),
    "Factory Analysis": (
        lambda x, y: FactorAnalysis(n_components=2).fit_transform(x)
    )
}


def draw(filename, target_dir):
    df = pd.read_csv(filename)

    for name in embedding_functions:
        plt.figure(figsize=(7, 7))
        sns.scatterplot(x=df[name + "_1"], y=df[name + "_2"],
                        hue=df["tags"].astype("category")).set(title=name)
        plt.savefig(f"{target_dir}/{name}.png")


# given two vectors, find the cosine similarity
def cosine(u, v):
    return np.dot(u, v) / (np.linalg.norm(u) * np.linalg.norm(v))


# Finds top-n closest sentences to the given sentence (vec)
def top_n_matches(vec, sent_vecs, n=10):
    scores = []
    for ix, sent_vec in enumerate(sent_vecs):
        cos = cosine(vec, sent_vec)
        scores.append((ix, cos))

    return sorted(scores, key=lambda a: np.absolute(a[1]), reverse=True)[:n]

def get_top_matches(source_filename, target_filename, n):

    # the top sentence will be a duplicate, so we need to generate one more sentence
    n+=1
    
    data = pd.read_csv(source_filename, usecols=["text", "tags"]).reset_index()

    # Get sentence embedding
    module_url = "https://tfhub.dev/google/universal-sentence-encoder/4"
    model = hub.load(module_url)
    sentence_embeddings = model(data["text"].tolist())

    # If n=6, we want top sents 1 to 5git 
    for i in range(1,n):
        data[f"top_sent_{i}"] = ""
        data[f"top_sent_tag_{i}"] = ""
        data[f"top_sent_score_{i}"] = 0

    for ix, sent_vec in enumerate(sentence_embeddings):
        top_n_sents = top_n_matches(sent_vec, sentence_embeddings, n=n)[1:]
        for i, (index, score) in enumerate(top_n_sents): # ignore the first sentence, its the same exact sentence
            data.loc[ix, f"top_sent_{i+1}"] = data.loc[index, "text"]
            data.loc[ix, f"top_sent_tag_{i+1}"] = data.loc[index, "tags"]
            data.loc[ix, f"top_sent_score_{i+1}"] = score
    
    data.to_csv(target_filename, index=False)

# Loads data and calls all dimensionality reduction functions
def main(source_filename, target_filename):

    # Read Data
    data = pd.read_csv(source_filename, 
                       usecols=["text", "tags", "sub_tags", "data_source"])

    # Get sentence embedding
    module_url = "https://tfhub.dev/google/universal-sentence-encoder/4"
    model = hub.load(module_url)
    sentence_embeddings = model(data["text"].tolist())

    # Fomalize data
    X, y = sentence_embeddings, data["tags"]
    print(X.shape, y.shape)

    # dict to store all projections
    embeddings = {}

    for name, func in embedding_functions.items():
        embedding = func(X, y)
        embeddings[name + "_1"] = embedding[:, 0]  # first component
        embeddings[name + "_2"] = embedding[:, 1]  # second component

    # Gather all embeddings as pandas dataframe
    df = pd.DataFrame(embeddings)
    # print(df.head())

    # Combine labels with embeddings
    df = df.merge(data, left_index=True, right_index=True)

    # Save all embeddings as csv
    df.reset_index().to_csv(target_filename, index=False)


# Runs all dimensionality reduction algorithms and save as csv
# main(source_filename="data/data.csv" ,target_filename="data/embeddings.csv")

# # Plots and save all embeddings in "pyscripts/plots" folder
# draw(filename="data/embeddings.csv", target_dir="data/plots/dimension_reduction")

get_top_matches(source_filename="data/embeddings.csv", target_filename="data/top_5_matches.csv", n=5)
