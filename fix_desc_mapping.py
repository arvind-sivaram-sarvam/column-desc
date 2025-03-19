import pandas as pd
import json
import re
import time
import os
from concurrent.futures import ThreadPoolExecutor, as_completed
from litellm import completion
import threading

input_excel = "/Users/arvindsivaram/column-desc/NDAP_Data_SourceMaster_1.xlsx"
col_desc_json_path = "/Users/arvindsivaram/column-desc/column_desc_checkpoint.json"
output_excel = "/Users/arvindsivaram/column-desc/augmented_source_master.xlsx"

df = pd.read_excel(input_excel)

if os.path.exists(col_desc_json_path):
    with open(col_desc_json_path, "r") as f:
        col_desc_data = json.load(f)
else:
    col_desc_data = {}

lock = threading.Lock()

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

def extract_description(entry):
    if isinstance(entry, dict):
        text = entry.get("column_description", "")
    else:
        text = str(entry)
    parts = text.split(":", 1)
    if len(parts) == 2:
        return parts[1].strip()
    return text.strip()

def generate_column_description_api(table_name, table_desc, column, metadata):
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
    - Table Name: {table_name}
    - Table Description: {table_desc}
    - Column: {column}
    - Metadata: {metadata}
    
    Enclose the final column description in the format @COLUMN_NAME: COLUMN_DESCRIPTION@
    """

    max_retries = 5
    retry_delay = 20
    for attempt in range(max_retries):
        try:
            response = completion(
                model="claude-3-5-sonnet-20241022",
                messages=[{"role": "user", "content": prompt}]
            ).choices[0].message.content
            match = re.search(r'@(.*?)@', response)
            if match:
                return match.group(1).strip()
            else:
                return None
        except Exception as e:
            print(f"API Error for column '{column}': {e} | Retrying in {retry_delay} seconds (Attempt {attempt+1}/{max_retries})")
            time.sleep(retry_delay)
            retry_delay *= 2
    print(f"Failed to generate column description for '{column}' after multiple retries.")
    return "Not Generated"

def process_row(row):
    src = str(row["source_id"])
    chunk = row["chunk"]
    metadata = row["metadata"]
    table_name, table_desc, columns = parse_chunk(chunk)
    # exactly one column in the "Columns:" section.
    if not columns:
        return (row.name, src, None, "No Column Found")
    col = columns[0].strip()
    # check for an exact match in the checkpoint JSON for this source id.
    with lock:
        entries = col_desc_data.get(src, [])
        matching_entry = None
        for entry in entries:
            if isinstance(entry, dict):
                if entry.get("column_name", "").strip() == col:
                    matching_entry = entry
                    break
            else:
                if entry.strip().startswith(col + ":"):
                    matching_entry = entry
                    break
    if matching_entry is not None:
        desc = extract_description(matching_entry)
        return (row.name, src, col, desc)
    else:
        # call the API to generate a new description.
        new_desc = generate_column_description_api(table_name, table_desc, col, metadata)
        if not new_desc or new_desc.strip() == "":
            new_desc = "Not Generated"
        # store in checkpoint JSON in the same format as before.
        new_entry = {"column_name": col, "column_description": f"{col}: {new_desc}"}
        with lock:
            if src in col_desc_data:
                col_desc_data[src].append(new_entry)
            else:
                col_desc_data[src] = [new_entry]
        return (row.name, src, col, new_desc)

results = {}
max_threads = 6
with ThreadPoolExecutor(max_workers=max_threads) as executor:
    futures = {executor.submit(process_row, row): row.name for idx, row in df.iterrows()}
    for future in as_completed(futures):
        try:
            index, src, col, desc = future.result()
            results[index] = desc
        except Exception as e:
            print("Error processing row:", e)

df["column_description"] = df.index.map(lambda idx: results.get(idx, ""))
df.to_excel(output_excel, index=False)
print(f"Augmented sheet written to {output_excel}.")

with open(col_desc_json_path, "w") as f:
    json.dump(col_desc_data, f, indent=2)
    f.flush()
    os.fsync(f.fileno())
print(f"Column description checkpoint updated at {col_desc_json_path}.")
