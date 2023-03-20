from time import time
import matplotlib.pyplot as plt
import pandas as pd
from wordcloud import WordCloud
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
from sklearn.decomposition import NMF, MiniBatchNMF, LatentDirichletAllocation, TruncatedSVD

n_samples = 2000
n_features = 1000
n_components = 10  # number of topics
n_top_words = 20
batch_size = 128
init = "nndsvda"


def plot_top_words(model, feature_names, n_top_words, title):
    fig, axes = plt.subplots(2, 5, figsize=(30, 15), sharex=True)
    axes = axes.flatten()
    for topic_idx, topic in enumerate(model.components_):
        top_features_ind = topic.argsort()[: -n_top_words - 1: -1]
        top_features = [feature_names[i] for i in top_features_ind]
        weights = topic[top_features_ind]

        ax = axes[topic_idx]
        ax.barh(top_features, weights, height=0.7)
        ax.set_title(f"Topic {topic_idx +1}", fontdict={"fontsize": 30})
        ax.invert_yaxis()
        ax.tick_params(axis="both", which="major", labelsize=20)
        for i in "top right left".split():
            ax.spines[i].set_visible(False)
        fig.suptitle(title, fontsize=40)

    plt.subplots_adjust(top=0.90, bottom=0.05, wspace=0.90, hspace=0.3)
    plt.savefig("data/plots/topics/" + title + ".png")


topic_models = {
    "NMF model (Frobenius norm)": NMF(
        n_components=n_components,
        random_state=1,
        init=init,
        beta_loss="frobenius",
        alpha_W=0.00005,
        alpha_H=0.00005,
        l1_ratio=0,
    ),
    "NMF model (KL divergence)": NMF(
        n_components=n_components,
        random_state=1,
        init=init,
        beta_loss="kullback-leibler",
        solver="mu",
        max_iter=1000,
        alpha_W=0.00005,
        alpha_H=0.00005,
        l1_ratio=0.5,
    ),
    "MiniBatchNMF model (Frobenius norm)": MiniBatchNMF(
        n_components=n_components,
        random_state=1,
        batch_size=batch_size,
        init=init,
        beta_loss="frobenius",
        alpha_W=0.00005,
        alpha_H=0.00005,
        l1_ratio=0.5,
    ),
    "MiniBatchNMF model (KL divergence)": MiniBatchNMF(
        n_components=n_components,
        random_state=1,
        batch_size=batch_size,
        init=init,
        beta_loss="kullback-leibler",
        alpha_W=0.00005,
        alpha_H=0.00005,
        l1_ratio=0.5,
    ),
    "LDA": LatentDirichletAllocation(
        n_components=n_components,
        max_iter=5,
        learning_method="online",
        learning_offset=50.0,
        random_state=0,
    ),
    "LSA": TruncatedSVD(
        n_components=n_components,
    )
}


def get_tfidf(data):
    # Use tf-idf features for NMF.
    # Words occurring in only one document or in at least 95% of the documents are removed
    print("Extracting tf-idf features for NMF...")
    tfidf_vectorizer = TfidfVectorizer(
        lowercase=True, max_df=0.95, min_df=2, max_features=n_features, stop_words="english",
    )
    tfidf = tfidf_vectorizer.fit_transform(data)
    return tfidf_vectorizer, tfidf


def get_tf(data):
    # Use tf (raw term count) features for LDA.
    print("Extracting tf features for LDA...")
    tf_vectorizer = CountVectorizer(
        lowercase=True, max_df=0.95, min_df=2, max_features=n_features, stop_words="english"
    )
    tf = tf_vectorizer.fit_transform(data)
    return tf_vectorizer, tf

def get_wordcloud(name, words, weights):
    # frequencies : dict from string to float
    frequencies = {word:weight for word, weight in zip(words, weights)}
    word_cloud = WordCloud(collocations = False, background_color = 'white').fit_words(frequencies)
    plt.imshow(word_cloud, interpolation='bilinear')
    plt.axis("off")
    plt.tight_layout()
    plt.savefig("data/plots/wordclouds/" + name + ".png")

def normalize_topic_distribution_per_row(folder_name):
    def get_top_n_topics(row, n):
        top_topic_names = []
        top_topic_weights = []
        total = row.sum()
        
        for _ in range(n):
            max_ix = row.argmax()
            max_topic = row.index[max_ix]
            max_value = row[max_ix]*100/total
            row[max_ix] = -100000

            top_topic_names.append(max_topic)
            top_topic_weights.append(max_value)
        
        new_row = {}
        for i, (name, weight) in enumerate(zip(top_topic_names, top_topic_weights), 1):
            new_row[f"top_topic_name_{i}"] = name
            new_row[f"top_topic_weight_{i}"] = weight
        
        return new_row

    emebds = pd.read_csv("data/embeddings.csv")
    for topic_model in topic_models.keys():
        filename = folder_name + topic_model + ".csv"
        new_filename = folder_name[:-1] + "_compressed/" + topic_model + ".csv"
        df = pd.read_csv(filename)
        df = pd.DataFrame.from_records(df.apply(lambda x: get_top_n_topics(x, 3), axis=1).values).reset_index()
        df.to_csv(new_filename, index=False)

        # augment embeddings.csv
        emebds= emebds.merge(df[["top_topic_name_1", "index"]], on="index").rename(columns={"top_topic_name_1":f"top_topic_{topic_model}"})
        emebds.to_csv("data/embeddings_w_topics.csv", index=False)

def main():
    print("Loading dataset...")
    data = pd.read_csv("data/data.csv", usecols=["text"])["text"].tolist()

    tfidf_vectorizer, tfidf = get_tfidf(data)
    tf_vectorizer, tf = get_tf(data)

    for name, func in topic_models.items():
        x = tfidf
        vectorizer = tfidf_vectorizer
        if name == "LDA":
            x = tf
            vectorizer = tf_vectorizer
        
        topic_model = func.fit(x)
        feature_names = vectorizer.get_feature_names_out()
        
        # plot_top_words(topic_model, feature_names, n_top_words, name)

        # components_ : (n_components, n_features)
        topic_df = pd.DataFrame()
        # loop through topics
        for topic_num, topics in enumerate(topic_model.components_, 1):
            # get top_n words per topic
            top_features_ind = topics.argsort()[: -n_top_words - 1: -1]
            top_features = [feature_names[i] for i in top_features_ind]
            # get the weights of the words
            weights = topics[top_features_ind]
            topic_df[f"topic_{topic_num}_weights"] = weights
            topic_df[f"topic_{topic_num}_words"] = top_features

            get_wordcloud(name + "_" + str(topic_num), feature_names, topics)
        
        topic_df.to_csv("data/topic_models/" + name + ".csv", index=False)

        # transform(X) : (n_samples, n_components)
        sample2topics = pd.DataFrame(topic_model.transform(x), columns=[f"topic_{i}" for i in range(1, n_components+1)])
        sample2topics.to_csv("data/data_topics/" + name + ".csv", index=False)

# main()
normalize_topic_distribution_per_row("data/data_topics/")
