import pandas as pd
import re

input_file = "/Users/arvindsivaram/column-desc/tmp/no_key_grouped_columns2.xlsx"
output_file = "/Users/arvindsivaram/column-desc/tmp/no_key_regrouped_g500.csv"

df = pd.read_excel(input_file)

def parse_chunk(chunk_str):
    """
    Given a chunk string of the form:
    
    Columns:
    - <column_name> --> <column_description>
    - <column_name> --> <column_description>
    ...
    
    Returns a list of tuples: [(column_name, column_description), ...]
    """
    lines = chunk_str.strip().splitlines()
    columns = []
    for line in lines:
        if line.strip().startswith("Columns:"):
            continue
        line = line.strip()
        # line starting with "- " followed by column info
        if line.startswith("- "):
            content = line[2:].strip() # remove the hyphen
            # split on the "-->"
            parts = content.split("-->")
            if len(parts) == 2:
                col_name = parts[0].strip()
                col_desc = parts[1].strip()
                columns.append((col_name, col_desc))
            else:
                continue
    return columns

df['columns_list'] = df['chunk'].apply(parse_chunk)
max_chunk_size = 500
grouped = df.groupby('source_id')

new_rows = []

for source_id, group in grouped:
    group_sorted = group.sort_index()
    current_chunk = []  
    current_count = 0
    
    for idx, row in group_sorted.iterrows():
        group_columns = row['columns_list']
        group_count = len(group_columns)
        
        # add groups to the chunk if it doesn't exceed 20 columns
        if current_count + group_count <= max_chunk_size:
            current_chunk.extend(group_columns)
            current_count += group_count
        else:
            # limit breached, end the chunk
            if current_chunk:
                chunk_text = "Columns:\n" + "\n".join(
                    [f"- {col_name} --> {col_desc}" for col_name, col_desc in current_chunk]
                )
                metadata = group_sorted.iloc[0]['metadata']
                new_rows.append({
                    'source_id': source_id,
                    'chunk': chunk_text,
                    'metadata': metadata
                })
            # the next chunk starts from the group we just excluded from the finished chunk
            current_chunk = group_columns.copy()
            current_count = group_count

    # if a leftover chunk hasn't been terminated, finish that source_id with that chunk
    if current_chunk:
        chunk_text = "Columns:\n" + "\n".join(
            [f"- {col_name} --> {col_desc}" for col_name, col_desc in current_chunk]
        )
        metadata = group_sorted.iloc[0]['metadata']
        new_rows.append({
            'source_id': source_id,
            'chunk': chunk_text,
            'metadata': metadata
        })

new_df = pd.DataFrame(new_rows)

# validation: for each source_id the total number of columns should be preserved.
def total_columns_for_source(rows):
    total = 0
    for r in rows:
        cols = parse_chunk(r)
        total += len(cols)
    return total

orig_counts = df.groupby('source_id')['columns_list'].apply(lambda lists: sum(len(x) for x in lists)).to_dict()
new_counts = new_df.groupby('source_id')['chunk'].apply(lambda chunks: sum(len(parse_chunk(c)) for c in chunks)).to_dict()

mismatch_count = 0
for sid in orig_counts:
    ocount = orig_counts[sid]
    ncount = new_counts.get(sid, 0)
    if ocount != ncount:
        mismatch_count += 1
        print(f"Mismatch for source_id {sid}: original {ocount} columns, regrouped {ncount} columns")

#new_df.to_excel(output_file, index=False)
new_df.to_csv(output_file, index=False)
print(f"There were {mismatch_count} source_ids with mismatched column counts.")
print(f"Regrouped output written to {output_file}")
