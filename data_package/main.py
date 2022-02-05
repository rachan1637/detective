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

def prep_code_to_author():
    merged_file = 'data/merged_form.csv'
    df = pd.read_csv(merged_file)

    code_to_author = {}
    for _, row in df.iterrows():
        story_code = row['Story Code']
        author = row['Author Code']
        code_to_author[story_code] = author

    return code_to_author

def prep_cluster_result(base = 'tfidf', n_clusters = 5, part = 'main'):
    code_to_partitions, code_to_plain_text = preprocessing()
    print("-----Start calculating tf-idf score-----")
    if part == 'main':
        corpus = [text for text in code_to_plain_text.values()]
    elif part == 'first':
        corpus = [text['first'] for text in code_to_partitions.values()]
    elif part == 'second':
        corpus = [text['second'] for text in code_to_partitions.values()]

    story_codes = [code for code in code_to_plain_text.keys()]
    vectorizer = TfidfVectorizer(
        max_df = 0.8,
        min_df = 0.5,
        ngram_range=(1,1)
    )

    tfidf_matrix = vectorizer.fit_transform(corpus)
    feature_words = vectorizer.get_feature_names()
    print(len(feature_words))
    print(tfidf_matrix.shape)
    dist = 1 - cosine_similarity(tfidf_matrix)

    num_clusters = n_clusters
    km = KMeans(n_clusters = num_clusters)
    if base == 'tfidf':
        km.fit(tfidf_matrix)
    elif base == 'dist':
        km.fit(dist)

    clusters = km.labels_.tolist()
    
    code_to_publication_year = prep_code_to_publication_year()
    code_to_author = prep_code_to_author()
    code_to_cluster = {}
    for story_code, cluster in zip(story_codes, clusters):
        code_to_cluster[story_code] = {'publication_year': int(code_to_publication_year[story_code]), 'author': code_to_author[story_code], 'group': cluster}

    with open(f"cluster_result_{base}.json", 'w') as file:
        json.dump(code_to_cluster, file)
    print("-----End dumping the cluster result json file.-----")

    if base == 'tfidf':
        print("Top terms per cluster:")

        order_centroids = km.cluster_centers_.argsort()[:, ::-1]
        for i in range(num_clusters):
            print(f"Cluster {i} words:")
            for ind in order_centroids[i, :10]:
                print(feature_words[ind])

    return tfidf_matrix, dist

def save_csv_and_plot(base = 'tfidf', n_clusters = 5):
    # Base could be tfidf, depends on what do you want to cluster
    tfidf_matrix, dist = prep_cluster_result(base = base, n_clusters = n_clusters)

    with open(f"cluster_result_{base}.json", 'r') as file:
        code_to_cluster = json.load(file)

    cluster_names = {0: '0', 1: '1', 2: '2', 3: '3', 4: '4', 5: '5', 6: '6'}
    
    mds = MDS(n_components = 2, dissimilarity="precomputed", random_state = 1)
    pos = mds.fit_transform(dist)
    # svd = TruncatedSVD(n_components=2)
    # pos = svd.fit_transform(dist)

    xs, ys = pos[:, 0], pos[:, 1]
    cluster_colors = {0: '#1b9e77', 1: '#d95f02', 2: '#7570b3', 3: '#03fcc2', 4: '#be03fc', 5: '#f5429b', 6: '#f5b342'}
    # cluster_colors = {0: '#1b9e77', 1: '#d95f02', 2: '#7570b3', 3: '#03fcc2', 4: '#be03fc', 5: '#f5429b'}
    clusters = [cluster['group'] for cluster in code_to_cluster.values()]
    codes = [code for code in code_to_cluster.keys()]
    years = [cluster['publication_year'] for cluster in code_to_cluster.values()]
    authors = [cluster['author'] for cluster in code_to_cluster.values()]

    df = pd.DataFrame(dict(x = xs, y = ys, label = clusters, publication_year = years, story_code = codes, author_code = authors))
    df.to_csv(f"cluster_df_{base}.csv")
    df = pd.read_csv(f"cluster_df_{base}.csv")
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

    plt.title(f'Clustering Plot by {base} with Publication Year')
    plt.savefig(f'cluster_plot_year_{base}.png')

    # set up plot
    ax.cla()
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

    for i in range(len(df)):
        ax.text(df.iloc[i]['x'], df.iloc[i]['y'], df.iloc[i]['author_code'], size=8)

    plt.title(f'Clustering Plot by {base} with Author Code')
    plt.savefig(f'cluster_plot_author_{base}.png')
    # plt.show() #show the plotf

if __name__ == "__main__":
    save_csv_and_plot(base = 'tfidf', n_clusters = 6)
    save_csv_and_plot(base = 'dist', n_clusters = 7)