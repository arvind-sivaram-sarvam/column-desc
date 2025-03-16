import pandas as pd
import os
from litellm import completion
import re

file_path = "/Users/arvindsivaram/column-desc/NDAP_Data_SourceMaster_1.xlsx"
df = pd.read_excel(file_path)

# extract Name, Description, and Columns from chunk 
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

def generate_column_description(name, description, columns, metadata): #  no length constraint,
    prompt1 = f"""
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

    - Table Name: {name}
    - Description: {description}
    - Columns: {", ".join(columns)}
    - Metadata: {metadata}

    Enclose the final column description in the following format @COLUMN_NAME: COLUMN_DESCRIPTION@
    """

    prompt2 = f"""
    You are a simulation expert, capable of understanding and accurately depicting the behaviour of a given character.
    Your task is to simulate a discussion between the following experts, who deliberate until they come up with a mutually
    agreed upon short description for a column.

    1) Context expert: has a deep understanding of the data in the table, the implications of the metadata, and thus the
    kind of information stored in the column being discussed. They will make sure the column description is relevant within the context
    of the table's purpose, given the information they can derive from the column name and metadata.
    2) Summarization expert: master of balancing brevity while still retaining all pertinent information in the column description.
    3) Evaluation expert: adept at telling if a proposed description aptly and accurately captures the essence of the column. They
    can tell if there is a value-add when provided with just the column description versus being provided with the 
    table description, column name and metadata.

    The experts are provided with only the below information to create a column description:

    - Table Name: {name}
    - Description: {description}
    - Columns: {", ".join(columns)}
    - Metadata: {metadata}

    Enclose the final column description within @(column description)@
    """

    response = completion(
                model="claude-3-5-sonnet-20241022",
                messages=[
                    {
                        "role": "user",
                        "content": prompt1
                    }
                ],
            ).choices[0].message.content
    
    match = re.search(r'@(.*?)@', response)
    return match.group(1) if match else None

new_data = []
for _, row in df[df["source_id"] == 7999].iterrows():
    source_id = row["source_id"]
    chunk = row["chunk"]
    metadata = row["metadata"]
    
    name, description, columns = parse_chunk(chunk)
    if name and description and columns:
        new_column_name = generate_column_description(name, description, columns, metadata)
        new_data.append({"source_id": source_id, "table_name": name, "metadata": metadata, "column_desc": new_column_name})

new_df = pd.DataFrame(new_data)
output_path = "/Users/arvindsivaram/column-desc/tmp/new_columns.csv"
new_df.to_csv(output_path, index=False)
