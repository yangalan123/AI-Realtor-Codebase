import json
if __name__ == '__main__':
    with open("collect_vocab.out.high_freq_50.txt", "r", encoding='utf-8') as f_in:
        buf = f_in.read()
        word_list = buf.strip().split(", ")
        word_set = set(word_list)
        print("word count", len(word_list))

    schema_path = "claude_output/claude_100.json"
    with open(schema_path, "r", encoding='utf-8') as f_in:
        schema = json.load(f_in)
        value_set = set()
        value_list = []
        for key in schema:
            item = schema[key]
            value_set |= set(item)
            value_list.extend(item)
        print("value count", len(value_set))
        print("value count (list)", len(value_list))

    word_set_50 = set(word_list[:50])
    print("intersection count", len(word_set_50 & value_set))
    print(f"collect_vocab.out.high_freq_50.txt - {schema_path}: ", len(word_set_50 - value_set))
    print(f"{schema_path} - collect_vocab.out.high_freq_50.txt: ", len(value_set - word_set_50))
