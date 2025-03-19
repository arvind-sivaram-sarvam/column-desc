import pandas as pd
import os
import json
import time
from litellm import completion
import re
import threading
from concurrent.futures import ThreadPoolExecutor, as_completed

file_path = "/Users/arvindsivaram/column-desc/NDAP_Data_SourceMaster_1.xlsx"
checkpoint_file = "/Users/arvindsivaram/column-desc/checkpoint_results.csv"
column_desc_checkpoint = "/Users/arvindsivaram/column-desc/column_desc_checkpoint.json"

df = pd.read_excel(file_path)
print(f"Logical CPUs: {os.cpu_count()}")

# Load checkpoint data for grouping results
if os.path.exists(checkpoint_file):
    checkpoint_df = pd.read_csv(checkpoint_file)
    processed_source_ids = set(checkpoint_df["source_id"].unique())
    new_data = checkpoint_df.to_dict(orient="records")
    print(f"Resuming from checkpoint. {len(processed_source_ids)} tables already processed.")
else:
    processed_source_ids = set()
    new_data = []

# Load column description cache to skip cols that were processed already
if os.path.exists(column_desc_checkpoint):
    with open(column_desc_checkpoint, "r") as f:
        column_desc_cache = json.load(f)
    # Normalize the checkpoint format to be a list of dicts
    for key in list(column_desc_cache.keys()):
        value = column_desc_cache[key]
        if isinstance(value, str):
            parts = value.split(":", 1)
            if len(parts) == 2:
                column_name = parts[0].strip()
                column_desc = parts[1].strip()
            else:
                column_name = value.strip()
                column_desc = value.strip()
            column_desc_cache[key] = [{"column_name": column_name, "column_description": column_desc}]
        elif isinstance(value, list):
            new_list = []
            for item in value:
                if isinstance(item, dict):
                    new_list.append(item)
                elif isinstance(item, str):
                    parts = item.split(":", 1)
                    if len(parts) == 2:
                        column_name = parts[0].strip()
                        column_desc = parts[1].strip()
                    else:
                        column_name = item.strip()
                        column_desc = item.strip()
                    new_list.append({"column_name": column_name, "column_description": column_desc})
            column_desc_cache[key] = new_list
else:
    column_desc_cache = {}

table_info = {}
max_threads = 6
batch_size = 12
api_semaphore = threading.Semaphore(max_threads)
lock = threading.Lock()

def parse_chunk(chunk):
    """Parses the chunk text to extract table name, table description and list of columns."""
    name, description, columns = "", "", []
    lines = chunk.strip().split("\n")
    current_section = None
    for line in lines:
        if line.startswith("Name:"):
            current_section = "name"
            name += line.replace("Name:", "").strip()
        elif line.startswith("Description:"):
            current_section = "description"
            description = line.replace("Description:", "").strip()
        elif line.startswith("Columns:"):
            current_section = "columns"
        elif re.match(r"^- ", line) and current_section == "columns":
            columns.append(line.replace("- ", "").strip())
        else:
            if current_section == "name":
                name += " " + line.strip()
            elif current_section == "description" and not line.startswith("Columns:"):
                description += " " + line.strip()
    return name.strip(), description.strip(), columns

def generate_column_description(name, description, column, metadata):
    prompt = f"""
    You are a summarization expert, capable of consuming multiple sources of information and describing their features and usage 
    in the given context. You are provided with a table name, table description, column name and associated metadata. Your task is to 
    generate a column description, which accurately reflects the data in the column as well as its purpose. Here are some guidelines:
    
    - Use only the provided information.
    - Do not change the column name.
    - The description must accurately reflect the column in context.
    - The description should add value beyond the basic metadata.
    - Balance between conciseness and completeness.
    
    Provided Information:
    - Table Name: {name}
    - Description: {description}
    - Column: {column}
    - Metadata: {metadata}
    
    Enclose the final column description in the format @COLUMN_NAME: COLUMN_DESCRIPTION@
    """
    
    max_retries = 5
    retry_delay = 20  
    for attempt in range(max_retries):
        try:
            with api_semaphore:
                response = completion(
                    model="claude-3-5-sonnet-20241022",
                    messages=[{"role": "user", "content": prompt}]
                ).choices[0].message.content
            match = re.search(r'@(.*?)@', response)
            return match.group(1) if match else None
        except Exception as e:
            print(f"API Error: {e} | Retrying in {retry_delay} seconds... (Attempt {attempt+1}/{max_retries})")
            time.sleep(retry_delay)
            retry_delay *= 2  
    print("Failed after multiple retries.")
    return None

def group_columns_by_relevance(source_id, table_name, column_descriptions, max_group_size):
    prompt = f"""
    You are a data organization expert. Your task is to group the following columns into clusters of at most {max_group_size} columns, ensuring the columns in each cluster are most relevant to each other.
    Your grouping should be based on the column names and their descriptions. You may choose how many clustered groups to return, and which/how many columns are in each cluster.
    The columns in a group must be highly relevant to each other. There shouldn't be any cases where (for example) geospatial data is grouped with temporal data.
    The connection that causes a grouping to be appropriate should be obvious from the column name and description, and its context in the table. Upon identifying
    a connection between a set of columns, you must make sure that it applies to every single column in that grouping. Ensure that you don't edit 
    the column names and descriptions provided to you in any way. They must appear in the output as they did in the input.

    Table Name: {table_name}
    Source ID: {source_id}
    
    Columns and Descriptions:
    """
    for col_desc in column_descriptions:
        prompt += f"\n- {col_desc}"
    prompt += """
    
    Do not provide any additional explanation in the answer.
    Your output should be in a parseable JSON format, where each row corresponds to a grouping of columns in that source_id.
    Below is the sample format you should stick to:
    [
        {
            "source_id": <source_id>,
            "grouping": "<basis for grouping>",
            "grouped_columns": [
                {"column_name": "Column1", "column_description": "Description1"},
                {"column_name": "Column2", "column_description": "Description2"}
            ]
        }
    ]
    """
    
    max_retries = 5
    retry_delay = 20
    for attempt in range(max_retries):
        try:
            with api_semaphore:
                response = completion(
                    model="claude-3-5-sonnet-20241022",
                    messages=[{"role": "user", "content": prompt}]
                ).choices[0].message.content
            return json.loads(response)
        except Exception as e:
            print(f"Grouping API Error: {e} | Retrying in {retry_delay} seconds... (Attempt {attempt+1}/{max_retries})")
            time.sleep(retry_delay)
            retry_delay *= 2 
    print("Grouping failed after multiple retries.")
    return []

try:
    batch_executor = ThreadPoolExecutor(max_workers=max_threads)
    batch_futures = []
    total_rows = len(df)
    batch_counter = 0

    for i in range(0, total_rows, batch_size):
        
        batch_df = df.iloc[i:i+batch_size]
        batch_entries = []
        for idx, row in batch_df.iterrows():
            source_id = row["source_id"]
            chunk = row["chunk"]
            metadata = row["metadata"]
            table_name, table_desc, columns = parse_chunk(chunk)
            with lock:
                if source_id not in table_info:
                    table_info[source_id] = {"table_name": table_name, "description": table_desc, "metadata": metadata}
            with lock:
                processed_columns = set()
                if str(source_id) in column_desc_cache:
                    processed_columns = {item["column_name"] for item in column_desc_cache[str(source_id)]}
            for col in columns:
                if col not in processed_columns:
                    batch_entries.append({
                        "source_id": source_id,
                        "table_name": table_name,
                        "description": table_desc,
                        "column": col,
                        "metadata": metadata
                    })
        if batch_entries:
            future = batch_executor.submit(
                lambda entries=batch_entries: [
                    {
                        "source_id": entry["source_id"],
                        "table_name": entry["table_name"],
                        "column_name": entry["column"],
                        "column_description": generate_column_description(
                            entry["table_name"], entry["description"], entry["column"], entry["metadata"]
                        )
                    }
                    for entry in entries
                ]
            )
            batch_futures.append((future, len(batch_entries)))
        batch_counter += 1
        # Sleep briefly to avoid overloading the LLM
        time.sleep(1)
        if batch_counter % 10 == 0:
            for future in as_completed([f for f, cnt in batch_futures]):
                try:
                    results = future.result()
                    with lock:
                        for res in results:
                            src = str(res["source_id"])
                            if src not in column_desc_cache:
                                column_desc_cache[src] = []
                            if not any(item["column_name"] == res["column_name"] for item in column_desc_cache[src]):
                                column_desc_cache[src].append({
                                    "column_name": res["column_name"],
                                    "column_description": res["column_description"]
                                })
                except Exception as e:
                    print(f"Error processing future result: {e}")
            batch_futures = [(f, cnt) for f, cnt in batch_futures if not f.done()]
            with open(column_desc_checkpoint, "w") as f:
                json.dump(column_desc_cache, f)
            print(f"Checkpoint after processing {i+batch_size} rows.")
    # Wait for any remaining futures
    for future in as_completed([f for f, cnt in batch_futures]):
        try:
            results = future.result()
            with lock:
                for res in results:
                    src = str(res["source_id"])
                    if src not in column_desc_cache:
                        column_desc_cache[src] = []
                    if not any(item["column_name"] == res["column_name"] for item in column_desc_cache[src]):
                        column_desc_cache[src].append({
                            "column_name": res["column_name"],
                            "column_description": res["column_description"]
                        })
        except Exception as e:
            print(f"Error in final future processing: {e}")
    batch_executor.shutdown(wait=True)
    with open(column_desc_checkpoint, "w") as f:
        json.dump(column_desc_cache, f)
    print("Finished processing all batches for column descriptions.")
except KeyboardInterrupt:
    print("Interrupted during batch processing. Shutting down executor.")
    batch_executor.shutdown(wait=False)
    exit(1)
