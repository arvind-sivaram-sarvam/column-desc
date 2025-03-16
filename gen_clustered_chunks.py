import pandas as pd
import numpy as np
import re
import openai
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.cluster import AgglomerativeClustering
from sklearn.preprocessing import StandardScaler

file_path = "/Users/arvindsivaram/column-desc/tmp/new_columns.csv"
df = pd.read_csv(file_path)

def extract_column_desc(text):
    match = re.match(r"(.+?): (.+)", str(text))
    if match:
        return match.group(1), match.group(2)
    return None, text  

df["column_name"], df["column_desc"] = zip(*df["column_desc"].apply(extract_column_desc))

def get_embedding(text):
    client = openai.OpenAI()
    response = client.embeddings.create(
        model="text-embedding-ada-002",
        input=[text]
    )
    return np.array(response.data[0].embedding)

df["table_metadata"] = df["table_name"] + " " + df["metadata"]
table_embedding = get_embedding(df["table_metadata"].iloc[0])
df["column_embedding"] = df["column_desc"].apply(get_embedding)

df["relevance_score"] = df["column_embedding"].apply(
    lambda x: np.dot(x, table_embedding) / (np.linalg.norm(x) * np.linalg.norm(table_embedding))
)

vectorizer = TfidfVectorizer(stop_words="english")
tfidf_matrix = vectorizer.fit_transform(df["column_desc"])

combined_features = np.hstack((tfidf_matrix.toarray(), np.vstack(df["column_embedding"].values)))

scaler = StandardScaler()
combined_features_scaled = scaler.fit_transform(combined_features)

num_clusters = max(2, len(df) // 4)
agglom = AgglomerativeClustering(n_clusters=num_clusters, metric="euclidean", linkage="ward")
df["cluster"] = agglom.fit_predict(combined_features_scaled)

df_sorted = df.sort_values(by=["cluster", "relevance_score"], ascending=[True, False])

K = 4  
grouped_chunks = []
valid_clusters = df_sorted["cluster"].unique()

for cluster_id in valid_clusters:
    cluster_group = df_sorted[df_sorted["cluster"] == cluster_id]["column_name"].tolist()

    while len(cluster_group) > 0:
        chunk_size = min(K, len(cluster_group))  
        grouped_chunks.append(", ".join(cluster_group[:chunk_size]))
        cluster_group = cluster_group[chunk_size:]

final_grouped_df = pd.DataFrame({"source_id": df["source_id"].iloc[0], "grouped_columns": grouped_chunks})

output_grouped_path = "/Users/arvindsivaram/column-desc/tmp/grouped_columns.csv"
final_grouped_df.to_csv(output_grouped_path, index=False)
