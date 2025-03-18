import pandas as pd
import numpy as np
import re
import openai
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.cluster import KMeans
from sklearn.metrics.pairwise import cosine_similarity
import hdbscan

file_path = "/Users/arvindsivaram/column-desc/tmp/new_columns.csv"
df = pd.read_csv(file_path)

# parse column desc from the input table
def extract_column_desc(text):
    match = re.match(r"(.+?): (.+)", str(text))
    if match:
        return match.group(1), match.group(2)
    return None, text  

df["column_name"], df["column_desc"] = zip(*df["column_desc"].apply(extract_column_desc))

# create embeddings for the column descs
def get_embedding(text):
    client = openai.OpenAI()
    response = client.embeddings.create(
        model="text-embedding-ada-002",
        input=[text]
    )
    return np.array(response.data[0].embedding)

df["column_embedding"] = df["column_desc"].apply(get_embedding)

# init TF-IDF scores
vectorizer = TfidfVectorizer(stop_words="english")
tfidf_matrix = vectorizer.fit_transform(df["column_name"].fillna(""))

final_results = []

# process each source_id separately
for source_id, group in df.groupby("source_id"):
    
    # combine TF-IDF scores and OpenAI Embeddings to get better contextual understanding
    embedding_matrix = np.vstack(group["column_embedding"].values)
    tfidf_matrix_group = tfidf_matrix[group.index].toarray()

    # ensure same size when combining
    if tfidf_matrix_group.shape[1] < embedding_matrix.shape[1]:
        padding = np.zeros((tfidf_matrix_group.shape[0], embedding_matrix.shape[1] - tfidf_matrix_group.shape[1]))
        tfidf_matrix_group = np.hstack((tfidf_matrix_group, padding))

    combined_features = np.hstack((embedding_matrix, tfidf_matrix_group))

    # pre-clustering with KMeans to get rough idea of groups. helps to prevent geospatial/temporal data getting mixed up
    n_clusters = min(15, max(3, len(group) // 3))
    kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
    group["pre_cluster"] = kmeans.fit_predict(combined_features)

    # create distance matrix for clustering
    similarity_matrix = cosine_similarity(combined_features)
    distance_matrix = np.maximum(0, 1 - similarity_matrix)

    # run HDBSCAN on pre-clusters
    group["cluster"] = -1
    for cluster_id in group["pre_cluster"].unique():
        sub_df = group[group["pre_cluster"] == cluster_id]

        # assign outliers/pairs as clusters, do not drop them
        if len(sub_df) < 2:
            group.loc[sub_df.index, "cluster"] = sub_df.index  
            continue
        
        # regulate size of clusters. dont let HDBSCAN break the groups into singleton clusters
        sub_indices = np.array([group.index.get_loc(i) for i in sub_df.index])
        sub_distance_matrix = distance_matrix[np.ix_(sub_indices, sub_indices)]


        min_cluster_size = max(2, int(len(sub_df) * 0.3))
        min_samples = max(1, int(len(sub_df) * 0.15))

        sub_clusterer = hdbscan.HDBSCAN(metric="precomputed", min_cluster_size=min_cluster_size, min_samples=min_samples, cluster_selection_method='eom')
        sub_labels = sub_clusterer.fit_predict(sub_distance_matrix)
        group.loc[sub_df.index, "cluster"] = sub_labels + (cluster_id * 100)

    # each column should only appear once across clusters
    group["cluster"] = group.groupby("column_name")["cluster"].transform("min") 

    group_sorted = group.sort_values(by=["cluster"], ascending=True)

    # store as lists bc there's an issue splitting if names have commas in them
    assigned_columns = set()
    grouped_chunks = []

    # SET K = MAX NUMBER OF COLUMNS IN A CHUNK
    K = 20
    valid_clusters = group_sorted["cluster"].unique()

    for cluster_id in valid_clusters:
        cluster_group = group_sorted[group_sorted["cluster"] == cluster_id]["column_name"].tolist()

        unique_cluster_group = [col for col in cluster_group if col not in assigned_columns]
        assigned_columns.update(unique_cluster_group) 

        if unique_cluster_group:
            while len(unique_cluster_group) > 0:
                chunk_size = min(K, len(unique_cluster_group))  
                grouped_chunks.append(unique_cluster_group[:chunk_size])
                unique_cluster_group = unique_cluster_group[chunk_size:]

    # missing column check. every col should be in some chunk
    # should be resolved now with the comma fix
    all_columns = set(group["column_name"])
    chunked_columns = set(sum(grouped_chunks, [])) 

    missing_columns = all_columns - chunked_columns
    if missing_columns:
        grouped_chunks.append(list(missing_columns)) 

    # store results for each source_id
    for chunk in grouped_chunks:
        final_results.append({
            "source_id": source_id,
            "grouped_columns": ", ".join(chunk)
        })

# Convert results to DataFrame and save to CSV
final_df = pd.DataFrame(final_results)
output_grouped_path = "/Users/arvindsivaram/column-desc/tmp/grouped_columns.csv"
final_df.to_csv(output_grouped_path, index=False)
