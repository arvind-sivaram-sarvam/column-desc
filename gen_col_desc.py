import pandas as pd
import os
import json
import time
from litellm import completion
import re
import threading
from concurrent.futures import ThreadPoolExecutor, as_completed

input_file = "/Users/arvindsivaram/column-desc/NDAP_Data_SourceMaster_1.xlsx"
checkpoint_file = "/Users/arvindsivaram/column-desc/column_desc_checkpoint.csv"
output_file = "/Users/arvindsivaram/column-desc/augmented_input.xlsx"

max_threads = 6        
api_semaphore = threading.Semaphore(max_threads)
checkpoint_lock = threading.Lock()
processed_since_checkpoint = 0

def parse_chunk(chunk):
    name, description, column_name = "", "", ""
    lines = chunk.strip().split("\n")
    current_section = None
    for line in lines:
        if line.startswith("Name:"):
            current_section = "name"
            name = line.replace("Name:", "").strip()
        elif line.startswith("Description:"):
            current_section = "description"
            description = line.replace("Description:", "").strip()
        elif line.startswith("Columns:"):
            current_section = "columns"
        elif re.match(r"^- ", line) and current_section == "columns":
            column_name = re.sub(r"^- ", "", line, count=1).strip()
    return name, description, column_name

# copy() gets rid of SettingWithCopyWarning?
df = pd.read_excel(input_file).copy()  
# add UID for each row unless already present
if 'uid' not in df.columns:
    df.loc[:, 'uid'] = df.index

# parse the chunk field for each row and directly store table details and column name.
df[['table_name', 'table_desc', 'column_name']] = df.apply(
    lambda row: pd.Series(parse_chunk(row['chunk'])), axis=1)


# checkpoint rows: uid, source_id, table_name, table_desc, column_name, column_desc, metadata
checkpoint_results = {}
if os.path.exists(checkpoint_file):
    checkpoint_df = pd.read_csv(checkpoint_file)
    for _, row in checkpoint_df.iterrows():
        checkpoint_results[row['uid']] = row.to_dict()
    print(f"Loaded checkpoint for {len(checkpoint_results)} rows.")
else:
    print("No existing checkpoint found.")


# skip rows that have a valid column_desc (non-empty and not "NOT GENERATED")
rows_to_process = []
for _, row in df.iterrows():
    uid = row["uid"]
    if uid in checkpoint_results:
        desc = checkpoint_results[uid].get("column_desc", "")
        if pd.notna(desc) and desc != "NOT GENERATED" and desc != "":
            continue
    rows_to_process.append({
        "uid": uid,
        "source_id": row["source_id"],
        "table_name": row["table_name"],
        "table_desc": row["table_desc"],
        "column_name": row["column_name"],
        "metadata": row["metadata"]
    })

print(f"Total rows to process: {len(rows_to_process)}")


def generate_description_for_row(entry):
    prompt = f"""
    You are a summarization expert, capable of consuming multiple sources of information and describing their features and usage 
    in the given context. You are provided with a table name, table description, column name and associated metadata. Your task is to 
    generate a column description, which accurately reflects the data in the column as well as its purpose. Here some guidelines 
    to note:

    - You must only use the provided information to generate your description,
    - The column description must be accurate keeping in mind the column in the context of the entire table and the provided metadata. 
    - The column description's should be made in such a way that there is a value-add when a person is provided with
    just the column description, versus being provided with the table description, column name and metadata.
    - The column description must balance retaining all pertinent information while also not being unnecessarily long.

    You are provided with only the below information to create the column description:
    
    UID: {entry['uid']}
    Table Name: {entry['table_name']}
    Table Description: {entry['table_desc']}
    Column Name: {entry['column_name']}
    Metadata: {entry['metadata']}

    Return your answer strictly as a JSON object with exactly two keys:
    "uid": the provided UID (must be exactly {entry['uid']} as an integer),
    "column_description": the generated column description.
    Do not include any extra text, explanation, or formatting. The output must be a valid JSON object.
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
            result = json.loads(response)
            returned_uid = result.get("uid")
            # if the returned uid is missing or does not match, give a warning and overwrite it
            if returned_uid is None or int(returned_uid) != int(entry["uid"]):
                print(f"UID mismatch: expected {entry['uid']}, got {returned_uid}")
                result["uid"] = entry["uid"]
            return result
        except Exception as e:
            print(f"API Error for uid {entry['uid']}: {e} | Retrying in {retry_delay} seconds... (Attempt {attempt+1}/{max_retries})")
            time.sleep(retry_delay)
            retry_delay *= 2
    print(f"API call failed after multiple retries for uid {entry['uid']}. Marking as NOT GENERATED.")
    return {"uid": entry["uid"], "column_description": "NOT GENERATED"}

def process_rows(entries):
    global processed_since_checkpoint
    with ThreadPoolExecutor(max_workers=max_threads) as executor:
        future_to_uid = {executor.submit(generate_description_for_row, entry): entry["uid"] for entry in entries}
        for future in as_completed(future_to_uid):
            try:
                result = future.result()
                uid = int(result["uid"])
                # update checkpoint_results with the new description.
                with checkpoint_lock:
                    # retrieve all details from the original entry.
                    original = next((entry for entry in entries if entry["uid"] == uid), {})
                    checkpoint_results[uid] = {
                        "uid": uid,
                        "source_id": original.get("source_id", ""),
                        "table_name": original.get("table_name", ""),
                        "table_desc": original.get("table_desc", ""),
                        "column_name": original.get("column_name", ""),
                        "column_desc": result["column_description"],
                        "metadata": original.get("metadata", "")
                    }
                    processed_since_checkpoint += 1
                    # checkpoint every 10 rows processed.
                    if processed_since_checkpoint % 10 == 0:
                        pd.DataFrame(list(checkpoint_results.values())).to_csv(checkpoint_file, index=False)
                        print(f"Checkpoint updated after processing {processed_since_checkpoint} rows in this run.")
            except Exception as e:
                print(f"Error processing future: {e}")
    return

if rows_to_process:
    process_rows(rows_to_process)
else:
    print("No new rows to process.")

with checkpoint_lock:
    pd.DataFrame(list(checkpoint_results.values())).to_csv(checkpoint_file, index=False)
    print("Final checkpoint updated.")

def lookup_details(uid):
    # returns a dict with all desired details for a given uid.
    entry = checkpoint_results.get(uid, {})
    return {
        "uid": uid,
        "source_id": entry.get("source_id", ""),
        "table_name": entry.get("table_name", ""),
        "table_desc": entry.get("table_desc", ""),
        "column_name": entry.get("column_name", ""),
        "column_desc": entry.get("column_desc", "NOT GENERATED"),
        "metadata": entry.get("metadata", "")
    }

final_df = pd.DataFrame([lookup_details(uid) for uid in df['uid']])
final_df.to_excel(output_file, index=False)
print(f"Augmented output written to {output_file}")
