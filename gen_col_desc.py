import pandas as pd
import os
import json
import time
from litellm import completion
import re
import threading
from concurrent.futures import ThreadPoolExecutor, as_completed


file_path = "/Users/arvindsivaram/column-desc/NDAP_Data_SourceMaster_1.xlsx"
df = pd.read_excel(file_path)
print(f"Logical CPUs: {os.cpu_count()}")

checkpoint_file = "/Users/arvindsivaram/column-desc/checkpoint_results.csv"
column_desc_checkpoint = "/Users/arvindsivaram/column-desc/column_desc_checkpoint.json"

# Load checkpoint data
if os.path.exists(checkpoint_file):
    checkpoint_df = pd.read_csv(checkpoint_file)
    processed_source_ids = set(checkpoint_df["source_id"].unique())
    new_data = checkpoint_df.to_dict(orient="records")
    print(f"Resuming from checkpoint. {len(processed_source_ids)} tables already processed.")
else:
    processed_source_ids = set()
    new_data = []

if os.path.exists(column_desc_checkpoint):
    with open(column_desc_checkpoint, "r") as f:
        column_desc_cache = json.load(f)
else:
    column_desc_cache = {}

max_threads = 5
api_semaphore = threading.Semaphore(max_threads)

def parse_chunk(chunk):
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
    generate a column description, which accurately reflects the data in the column as well as its purpose. Here are some guidelines 
    to note:

    - You must only use the provided information to generate your description
    - You must not change the column names in any way. They should appear in your output as they do in the input.
    - The column description must be accurate keeping in mind the column in the context of the entire table and the provided metadata. 
    - The column description's should be made in such a way that there is a value-add when a person is provided with
    just the column description, versus being provided with the table description, column name and metadata.
    - The column description must balance retaining all pertinent information while also not being unnecessarily long.

    You are provided with only the below information to create the column description:

    - Table Name: {name}
    - Description: {description}
    - Column: {column}
    - Metadata: {metadata}

    Enclose the final column description in the following format @COLUMN_NAME: COLUMN_DESCRIPTION@
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
            "source_id": {source_id},
            "grouping": "(basis for grouping)",
            "grouped_columns": [
    {"column_name": "Column1", "column_description": "Description of Column1"},
    {"column_name": "Column2", "column_description": "Description of Column2"}
]
        }
    ]
    """
    
    max_retries = 5
    retry_delay = 20

    for attempt in range(max_retries):
        try:
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

counter = 0
for source_id, group in df.groupby("source_id"):
    if source_id in processed_source_ids:
        print(f"Skipping {source_id}, already processed.")
        continue

    print(f"Running for Table {source_id}")
    column_descriptions = column_desc_cache.get(str(source_id), [])
    name, description = "", ""

    for _, row in group.iterrows():
        chunk = row["chunk"]
        metadata = row["metadata"]
        name, description, columns = parse_chunk(chunk)

        if name and description and columns:
            remaining_columns = [col for col in columns if col not in [desc.split(":")[0] for desc in column_descriptions]]

            with ThreadPoolExecutor(max_workers=max_threads) as executor:
                future_to_column = {
                    executor.submit(generate_column_description, name, description, column, metadata): column
                    for column in remaining_columns
                }

                for future in as_completed(future_to_column):
                    column = future_to_column[future]
                    col_desc = future.result()
                    if col_desc:
                        column_descriptions.append(col_desc)


            column_desc_cache[str(source_id)] = column_descriptions
            with open(column_desc_checkpoint, "w") as f:
                json.dump(column_desc_cache, f)
    
    if column_descriptions:
        time.sleep(1)
        grouped_columns_list = group_columns_by_relevance(source_id, name, column_descriptions, max_group_size=20)
        for grouping in grouped_columns_list:
            chunk_content = "Columns:\n" + "\n".join(
                [f"- {col['column_name']}: {col['column_description']}" for col in grouping.get("grouped_columns", [])]
            )
            new_data.append({
                "source_id": source_id,
                "chunk": chunk_content,
                "metadata": row["metadata"]
            })

    if (counter % 10 == 0):
        temp_df = pd.DataFrame(new_data)
        temp_df.to_csv(checkpoint_file, index=False)
        temp_df.to_excel("/Users/arvindsivaram/column-desc/tmp/checkpoint_results.xlsx", index=False)
        print(f"Checkpoint saved: {len(new_data)} rows processed.")
    print(f"Finished Table {source_id}")
    counter += 1

new_df = pd.DataFrame(new_data)
output_path = "/Users/arvindsivaram/column-desc/tmp/grouped_columns.xlsx"
new_df.to_excel(output_path, index=False)
