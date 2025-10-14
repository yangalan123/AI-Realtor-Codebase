import json
def level_travel(schema, level_dict, level=0):
    if isinstance(schema, dict):
        for key in schema:
            if level not in level_dict:
                level_dict[level] = 0
            level_dict[level] += 1
            level_travel(schema[key], level_dict, level + 1)
    return
if __name__ == '__main__':
    with open("claude_output/claude_merged.json", "r") as f_in:
        schema = json.load(f_in)
        # do level traversal to print level-wise node counts
        level_dict = {}
        level_travel(schema, level_dict)
        for level in level_dict:
            print(f"Level {level}: {level_dict[level]} nodes.")

