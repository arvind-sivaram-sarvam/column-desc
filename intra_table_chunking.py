import pandas as pd
import json
import re
import time
import os
from concurrent.futures import ThreadPoolExecutor, as_completed
from litellm import completion
import threading

input_excel = "/Users/arvindsivaram/column-desc/augmented_input.xlsx"
grouping_checkpoint_file = "/Users/arvindsivaram/column-desc/keyed_grouping_checkpoint2.csv"
final_output_file = "/Users/arvindsivaram/column-desc/tmp/keyed_grouped_columns2.xlsx"

df = pd.read_excel(input_excel)

# pick up from checkpoint, if it exists
if os.path.exists(grouping_checkpoint_file):
    try:
        checkpoint_df = pd.read_csv(grouping_checkpoint_file)
        processed_source_ids = set(checkpoint_df["source_id"].astype(str).unique())
        new_data = checkpoint_df.to_dict(orient="records")
        print(f"Resuming grouping from checkpoint. {len(processed_source_ids)} tables already processed.")
    except pd.errors.EmptyDataError:
        print("Checkpoint file exists but is empty. Starting from scratch.")
        processed_source_ids = set()
        new_data = []
else:
    processed_source_ids = set()
    new_data = []

lock = threading.Lock()

# group each source id
grouped_columns = {}
table_info = {}
for idx, row in df.iterrows():
    src = str(row["source_id"])  
    table_name = row["table_name"]
    table_desc = row["table_desc"] 
    if src not in table_info:
        table_info[src] = {"table_name": table_name, "metadata": row["metadata"]}
    col_desc = row["column_desc"]
    if isinstance(col_desc, str):
        col_desc = col_desc.strip()
    else:
        col_desc = ""
    col_item = {
        "uid": row["uid"],
        "column_name": row["column_name"],
        "column_description": col_desc
    }
    grouped_columns.setdefault(src, []).append(col_item)

# remove duplicate columns (by UID) for each source_id.
for src in grouped_columns:
    seen = set()
    unique_list = []
    for item in grouped_columns[src]:
        if item["uid"] not in seen:
            unique_list.append(item)
            seen.add(item["uid"])
    grouped_columns[src] = unique_list

max_threads = 6
api_semaphore = threading.Semaphore(max_threads)

def group_columns_by_relevance_with_uid(source_id, table_name, columns_list, max_group_size):
    prompt = f"""You are a data organization expert. Your task is to group the following columns into clusters of at most {max_group_size} columns, ensuring the columns in each cluster are most relevant to each other.
Your grouping should be based on the column names and their descriptions. You may choose how many clusters to return, and which/how many columns are in each cluster.
The columns in a cluster must be highly relevant to each other. There shouldn't be any cases where (for example) geospatial data is grouped with temporal data.
The connection that causes a grouping to be appropriate should be obvious from the column name and description, and its context in the table. Upon identifying
a connection between a set of columns, you must make sure that it applies to every single column in that grouping.
IMPORTANT: Do NOT modify the provided column names or descriptions. Each column is identified by a unique UID.
In your output, for each grouping, provide the UIDs of the columns in that grouping.

Table Name: {table_name}
Source ID: {source_id}

List of Columns:"""
    for col in columns_list:
        prompt += f"\n- UID: {col['uid']}, Column: {col['column_name']}, Description: {col['column_description']}"
    prompt += """
    
Output your answer as a JSON array of objects. Each object should represent one grouping and be in the format:
[
    {
        "source_id": <source_id>,
        "grouping": "<brief description of the grouping basis>",
        "grouped_uids": ["<uid1>", "<uid2>", ... ]
    }
]
Do not include any extra commentary.
"""
    max_retries = 3
    retry_delay = 20
    for attempt in range(max_retries):
        try:
            with api_semaphore:
                response = completion(
                    model="claude-3-5-sonnet-20241022",
                    messages=[{"role": "user", "content": prompt}]
                ).choices[0].message.content
            time.sleep(1) 
            groups = json.loads(response)
            return groups
        except Exception as e:
            print(f"Grouping API Error for source_id {source_id}: {e} | Retrying in {retry_delay} seconds (Attempt {attempt+1}/{max_retries})")
            time.sleep(retry_delay)
            retry_delay *= 2
    print(f"Grouping failed for source_id {source_id} after multiple retries.")
    return None

def update_checkpoint(new_groupings, checkpoint_file):
    with lock:
        if os.path.exists(checkpoint_file) and os.path.getsize(checkpoint_file) > 0:
            try:
                existing_df = pd.read_csv(checkpoint_file)
                existing_data = existing_df.to_dict(orient="records")
            except pd.errors.EmptyDataError:
                existing_data = []
        else:
            existing_data = []
        combined_data = existing_data + new_groupings
        temp_df = pd.DataFrame(combined_data)
        temp_df.to_csv(checkpoint_file, index=False)
        print(f"Checkpoint updated: {len(combined_data)} rows processed.")

# list of source ids which need to be processed again/for the first time
all_source_ids = sorted(grouped_columns.keys(), key=lambda x: int(x) if x.isdigit() else x)
pending_ids = [src for src in all_source_ids if src not in processed_source_ids]
print(f"Total source_ids to process: {len(pending_ids)}")

final_groupings = []
processed_count = 0
failed_source_ids = []

# multithreaded approach for getting multiple source ids
with ThreadPoolExecutor(max_workers=max_threads) as executor:
    future_to_src = {}
    for src in pending_ids:
        if src not in table_info:
            print(f"Warning: No table info for source_id {src}. Skipping.")
            continue
        table_name = table_info[src]["table_name"]
        future = executor.submit(group_columns_by_relevance_with_uid, src, table_name, grouped_columns[src], max_group_size=20)
        future_to_src[future] = src

    for future in as_completed(future_to_src):
        src = future_to_src[future]
        try:
            groups = future.result()
            if groups is None:
                print(f"No grouping result for source_id {src}.")
                failed_source_ids.append(src)
            else:
                assigned_uids = set()
                for grouping in groups:
                    filtered_columns = []
                    for uid in grouping.get("grouped_uids", []):
                        if uid in assigned_uids:
                            continue
                        assigned_uids.add(uid)
                        original = next(
                                (item for item in grouped_columns[src] 
                                if str(item["uid"]).strip() == str(uid).strip()),
                                None
                            )
                        if original:
                            filtered_columns.append({
                                "uid": original["uid"],
                                "column_name": original["column_name"],
                                "column_description": original["column_description"]
                            })
                        else:
                            filtered_columns.append({
                                "uid": uid,
                                "column_name": "",
                                "column_description": ""
                            })
                    if filtered_columns:
                        chunk_text = "Columns:\n" + "\n".join(
                            [f"- {item['column_name']} --> {item['column_description']}" for item in filtered_columns]
                        )
                        # chunk_text = "Columns:\n" + "\n".join(
                        #     [f"- Column Name: {item['column_name']}, Column Description: {item['column_description']}" for item in filtered_columns]
                        # )
                        final_groupings.append({
                            "source_id": src,
                            "chunk": chunk_text,
                            "metadata": table_info[src]["metadata"]
                        })
                processed_source_ids.add(src)
                print(f"Processed grouping for source_id {src}.")
            processed_count += 1
            if processed_count % 10 == 0:
                update_checkpoint(final_groupings, grouping_checkpoint_file)
                final_groupings = []  # Clear after checkpointing.
        except Exception as e:
            print(f"Error processing grouping for source_id {src}: {e}")
            failed_source_ids.append(src)

print(f"Failed source IDs: {failed_source_ids}")

if final_groupings:
    new_data.extend(final_groupings)
    update_checkpoint(final_groupings, grouping_checkpoint_file)

if os.path.exists(grouping_checkpoint_file) and os.path.getsize(grouping_checkpoint_file) > 0:
    existing_df = pd.read_csv(grouping_checkpoint_file)
    try:
        existing_df = existing_df.sort_values(by="source_id", key=lambda x: x.astype(int))
    except Exception as e:
        print("Sorting source_id as int failed, sorting as string:", e)
        existing_df = existing_df.sort_values(by="source_id")
    existing_df.to_csv(grouping_checkpoint_file, index=False)
    existing_df.to_excel(final_output_file, index=False)
    print(f"Final grouping checkpoint saved: {len(existing_df)} rows processed.")
    print(f"Final grouped output written to {final_output_file}.")
else:
    print("No grouping results to write.")
