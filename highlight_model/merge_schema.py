import json
import os.path


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
    # save the merged schema
    if not os.path.exists("claude_merged.json"):
        json_files = ["claude_output/claude_100.json", "claude_schema_600.json", "claude_non_batch.json"]
        # load and merge these files
        schema = {}
        for json_file in json_files:
            with open(json_file, "r", encoding='utf-8') as f_in:
                schema.update(json.load(f_in))
        with open("claude_merged.json", "w", encoding='utf-8') as f_out:
            json.dump(schema, f_out, indent=4)
        # give a simplified version of the schema, do not keep the values/leaf nodes
        simplified_schema = simplify_schema(schema)
        with open("claude_merged_simplified.json", "w", encoding='utf-8') as f_out:
            json.dump(simplified_schema, f_out, indent=4)
    else:
        schema = json.load(open("claude_merged.json", "r", encoding='utf-8'))
    # load high_freq_vocab to check what is remained
    with open("collect_vocab.out.high_freq_50.txt", "r", encoding='utf-8') as f_in:
        buf = f_in.read()
        word_list = buf.strip().split(", ")
        # filter single-character words
        word_list = [word for word in word_list if len(word) > 1]
        word_set = set(word_list)
        print("word count", len(word_list))
    # check the coverage
    with open("claude_merged.json", "r", encoding='utf-8') as f_in:
        schema = json.load(f_in)
        value_set = set()
        value_list = []
        for key in schema:
            item = schema[key]
            value_set |= set(item)
            value_list.extend(item)
        print("value count", len(value_set))
        print("value count (list)", len(value_list))
    # check the intersection with word_list
    intersection = word_set & value_set
    print("intersection count", len(intersection))
    print(f"collect_vocab.out.high_freq_50.txt - claude_merged.json: ", len(word_set - value_set))
    print(f"claude_merged.json - collect_vocab.out.high_freq_50.txt: ", len(value_set - word_set))
    # save the remaining words in the word_set, one word per line
    with open("word_set_minus_value_set.txt", "w", encoding='utf-8') as f_out:
        f_out.write("\n".join(word_set - value_set))
