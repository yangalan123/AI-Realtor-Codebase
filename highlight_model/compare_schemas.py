import json
def compare_schemas(schema1, schema2, chain=""):
    for key in schema1:
        if key not in schema2:
            print(f"Key {chain + key} is not in the updated schema.")
        else:
            if isinstance(schema1[key], dict):
                compare_schemas(schema1[key], schema2[key], chain + key + ".")
            elif set(schema1[key]) != set(schema2[key]):
                print(f"Value of key {chain + key} is different.")
                print("schema1 - schema2: ", set(schema1[key]) - set(schema2[key]))
                print("schema2 - schema1: ", set(schema2[key]) - set(schema1[key]))
    return
if __name__ == '__main__':
    updated_schema_file = "claude_output/claude_merged.json"
    original_schema_file = "claude_output/claude_merged_old.json"
    with open(original_schema_file, "r", encoding='utf-8') as f_in:
        original_schema = json.load(f_in)
    with open(updated_schema_file, "r", encoding='utf-8') as f_in:
        updated_schema = json.load(f_in)
    # for key in original_schema:
    #     if key not in updated_schema:
    #         print(f"Key {key} is not in the updated schema.")
    compare_schemas(original_schema, updated_schema)
    print("------")
    compare_schemas(updated_schema, original_schema)

