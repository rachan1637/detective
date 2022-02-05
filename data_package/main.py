from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.cluster import KMeans
from torch import cosine_similarity
from preprocess import preprocessing
import json
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.manifold import MDS
from sklearn.decomposition import TruncatedSVD

def prep_code_to_publication_year():
    merged_file = 'data/merged_form.csv'
    df = pd.read_csv(merged_file)

    code_to_publication_year = {}
    for _, row in df.iterrows():
        story_code = row['Story Code']
        year = row['publication_year']
        code_to_publication_year[story_code] = year

    return code_to_publication_year

def prep_cluster_result():
    code_to_partitions, code_to_plain_text = preprocessing()
    print("-----Start calculating tf-idf score-----")
    corpus = [text for text in code_to_plain_text.values()]
    story_codes = [code for code in code_to_plain_text.keys()]
    vectorizer = TfidfVectorizer(
        max_df = 0.7,
        min_df = 0.3,
        ngram_range=(1,3)
    )

    tfidf_matrix = vectorizer.fit_transform(corpus)
    feature_words = vectorizer.get_feature_names()
    print(len(feature_words))
    print(tfidf_matrix.shape)
    dist = 1 - cosine_similarity(tfidf_matrix)

    num_clusters = 3
    km = KMeans(n_clusters = num_clusters)
    km.fit(tfidf_matrix)
    clusters = km.labels_.tolist()
    
    code_to_publication_year = prep_code_to_publication_year()
    code_to_cluster = {}
    for story_code, cluster in zip(story_codes, clusters):
        code_to_cluster[story_code] = {'publication_year': int(code_to_publication_year[story_code]), 'group': cluster}

    with open("cluster_result.json", 'w') as file:
        json.dump(code_to_cluster, file)
    print("-----End dumping the cluster result json file.-----")

    print("Top terms per cluster:")

    order_centroids = km.cluster_centers_.argsort()[:, ::-1]
    for i in range(num_clusters):
        print(f"Cluster {i} words:")
        for ind in order_centroids[i, :10]:
            print(feature_words[ind])
    return tfidf_matrix, dist

if __name__ == "__main__":
    tfidf_matrix, dist = prep_cluster_result()

    with open("cluster_result.json", 'r') as file:
        code_to_cluster = json.load(file)

    cluster_names = {0: '0', 1: '1', 2: '2'}
    
    mds = MDS(n_components = 2, dissimilarity="precomputed", random_state = 1)
    pos = mds.fit_transform(dist)
    # svd = TruncatedSVD(n_components=2)
    # pos = svd.fit_transform(dist)

    xs, ys = pos[:, 0], pos[:, 1]
    cluster_colors = {0: '#1b9e77', 1: '#d95f02', 2: '#7570b3'}
    clusters = [cluster['group'] for cluster in code_to_cluster.values()]
    codes = [code for code in code_to_cluster.keys()]
    years = [cluster['publication_year'] for cluster in code_to_cluster.values()]

    df = pd.DataFrame(dict(x = xs, y = ys, label = clusters, publication_year = years, story_code = codes))
    df.to_csv("cluster_plot_df.csv")
    df = pd.read_csv("cluster_plot_df.csv")
    groups = df.groupby('label')

    # set up plot
    fig, ax = plt.subplots(figsize=(17, 9)) # set size
    ax.margins(0.05) # Optional, just adds 5% padding to the autoscaling

    #iterate through groups to layer the plot
    #note that I use the cluster_name and cluster_color dicts with the 'name' lookup to return the appropriate color/label
    for name, group in groups:
        ax.plot(group.x, group.y, marker='o', linestyle='', ms=12, 
                label=cluster_names[name], color=cluster_colors[name], 
                mec='none')
        ax.set_aspect('auto')
        ax.tick_params(\
            axis= 'x',          # changes apply to the x-axis
            which='both',      # both major and minor ticks are affected
            bottom='off',      # ticks along the bottom edge are off
            top='off',         # ticks along the top edge are off
            labelbottom='off')
        ax.tick_params(\
            axis= 'y',         # changes apply to the y-axis
            which='both',      # both major and minor ticks are affected
            left='off',      # ticks along the bottom edge are off
            top='off',         # ticks along the top edge are off
            labelleft='off')
    
    ax.legend(numpoints=1)  #show legend with only 1 point

    #add label in x,y position with the label as the film title
    for i in range(len(df)):
        ax.text(df.iloc[i]['x'], df.iloc[i]['y'], df.iloc[i]['publication_year'], size=8)  

    plt.savefig('cluster_plot_mds_dist_37.png')
    # plt.show() #show the plotf