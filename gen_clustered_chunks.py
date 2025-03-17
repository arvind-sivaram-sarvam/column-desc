import pandas as pd
import numpy as np
import re
import openai
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.decomposition import PCA
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

# combine TF-IDF scores and OpenAI Embeddings to get better contextual understanding
embedding_matrix = np.vstack(df["column_embedding"].values)
tfidf_matrix = tfidf_matrix.toarray()

# ensure same size when combining
if tfidf_matrix.shape[1] < embedding_matrix.shape[1]:
    padding = np.zeros((tfidf_matrix.shape[0], embedding_matrix.shape[1] - tfidf_matrix.shape[1]))
    tfidf_matrix = np.hstack((tfidf_matrix, padding))

combined_features = np.hstack((embedding_matrix, tfidf_matrix))

# pre-clustering with KMeans to get rough idea of groups. helps to prevent geospatial/temporal data getting mixed up
n_clusters = min(15, max(3, len(df) // 3))
kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
df["pre_cluster"] = kmeans.fit_predict(combined_features)

# create distance matrix for clustering
similarity_matrix = cosine_similarity(combined_features)
distance_matrix = np.maximum(0, 1 - similarity_matrix)

# run HDBSCAN on pre-clusters
df["cluster"] = -1
for cluster_id in df["pre_cluster"].unique():
    sub_df = df[df["pre_cluster"] == cluster_id]

    # assign outliers/pairs as clusters, do not drop them
    if len(sub_df) < 2:
        df.loc[sub_df.index, "cluster"] = sub_df.index  
        continue
    
    # regulate size of clusters. dont let HDBSCAN break the groups into singleton clusters
    sub_distance_matrix = distance_matrix[sub_df.index][:, sub_df.index]
    min_cluster_size = max(2, int(len(sub_df) * 0.3))
    min_samples = max(1, int(len(sub_df) * 0.15))

    sub_clusterer = hdbscan.HDBSCAN(metric="precomputed", min_cluster_size=min_cluster_size, min_samples=min_samples, cluster_selection_method='eom')
    sub_labels = sub_clusterer.fit_predict(sub_distance_matrix)
    df.loc[sub_df.index, "cluster"] = sub_labels + (cluster_id * 100)

# each column should only appear once across clusters
df["cluster"] = df.groupby("column_name")["cluster"].transform("min") 

df_sorted = df.sort_values(by=["cluster"], ascending=True)

# store as lists bc there's an issue splitting if names have commas in them
assigned_columns = set()
final_grouped_chunks = []

# SET K = MAX NUMBER OF COLUMNS IN A CHUNK
K = 20
valid_clusters = df_sorted["cluster"].unique()

for cluster_id in valid_clusters:
    cluster_group = df_sorted[df_sorted["cluster"] == cluster_id]["column_name"].tolist()

    unique_cluster_group = [col for col in cluster_group if col not in assigned_columns]
    assigned_columns.update(unique_cluster_group) 

    if unique_cluster_group:
        while len(unique_cluster_group) > 0:
            chunk_size = min(K, len(unique_cluster_group))  
            final_grouped_chunks.append(unique_cluster_group[:chunk_size])
            unique_cluster_group = unique_cluster_group[chunk_size:]

# missing column check. every col should be in some chunk
# should be resolved now with the comma fix
all_columns = set(df["column_name"])
chunked_columns = set(sum(final_grouped_chunks, [])) 

missing_columns = all_columns - chunked_columns
if missing_columns:
    print(f"Warning: These columns were missing from final output: {missing_columns}")
    final_grouped_chunks.append(list(missing_columns)) 

final_grouped_df = pd.DataFrame({
    "source_id": df["source_id"].iloc[0], 
    "grouped_columns": [", ".join(group) for group in final_grouped_chunks] 
})

output_grouped_path = "/Users/arvindsivaram/column-desc/tmp/grouped_columns.csv"
final_grouped_df.to_csv(output_grouped_path, index=False)
