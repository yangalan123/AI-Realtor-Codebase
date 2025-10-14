import json
def simplify_schema(schema):
    updated_schema = {"last_to_leaf_nodes": []}
    if isinstance(schema, dict):
        for key in schema:
            ret = simplify_schema(schema[key])
            if ret:
                assert key != "last_to_leaf_nodes", "last_to_leaf_nodes is a reserved key in the schema."
                updated_schema[key] = ret
            else:
                updated_schema["last_to_leaf_nodes"].append(key)
    else:
        return None
    if len(updated_schema) == 1:
        return updated_schema["last_to_leaf_nodes"]
    else:
        return updated_schema

if __name__ == '__main__':
    with open("claude_output/claude_merged.json", "r", encoding='utf-8') as f_in:
        schema = json.load(f_in)
        simplified_schema = simplify_schema(schema)
    with open("claude_output/claude_merged_simplified_for_pre.json", "w", encoding='utf-8') as f_out:
        json.dump(simplified_schema, f_out, indent=4)