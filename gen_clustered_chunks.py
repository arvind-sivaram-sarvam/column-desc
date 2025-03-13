import pandas as pd
import numpy as np
import re
from sklearn.cluster import AgglomerativeClustering
import openai

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

df["relevance_score"] = df["column_embedding"].apply(lambda x: np.dot(x, table_embedding) / (np.linalg.norm(x) * np.linalg.norm(table_embedding)))

df_sorted = df.sort_values(by="relevance_score", ascending=False)

K = 4  

num_clusters = max(1, len(df_sorted) // K)  

clustering = AgglomerativeClustering(n_clusters=num_clusters, metric="euclidean", linkage="ward")
cluster_labels = clustering.fit_predict(np.vstack(df_sorted["column_embedding"].values))

df_sorted["cluster"] = cluster_labels

df_clustered = df_sorted.sort_values(by=["cluster", "relevance_score"], ascending=[True, False])

grouped_chunks = []
for cluster_id in df_clustered["cluster"].unique():
    cluster_group = df_clustered[df_clustered["cluster"] == cluster_id]["column_name"].tolist()
    
    for i in range(0, len(cluster_group), K):
        grouped_chunks.append(", ".join(cluster_group[i:i + K]))

final_grouped_df = pd.DataFrame({"source_id": df["source_id"].iloc[0], "grouped_columns": grouped_chunks})

output_grouped_path = "/Users/arvindsivaram/column-desc/tmp/grouped_columns.csv"
final_grouped_df.to_csv(output_grouped_path, index=False)
