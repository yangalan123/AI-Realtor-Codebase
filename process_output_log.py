import json
if __name__ == '__main__':
    lineid2user_and_shot = {}
    max_shot = 0
    user_dict = {}
    with open("collected_human_data.json", "r") as f_in:
        users = json.load(f_in)
        counter = 0
        for user in users:
            user_id = user["id"]
            shot_num = 0
            for response in user["responses"]:
                max_shot = max(max_shot, shot_num)
                lineid2user_and_shot[counter] = (user_id, shot_num)
                counter += 1
                shot_num += 1
                if user_id not in user_dict:
                    user_dict[user_id] = True
    shot_accuracy = dict()
    for shot_i in range(max_shot + 1):
        shot_accuracy[shot_i] = [0, 0]
    with open("output_log", "r") as f_in:
        lines = f_in.readlines()
        score = 0
        for i, line in enumerate(lines):
            predict, real = line.strip().split()
            user, shot = lineid2user_and_shot[i]
            if predict == real:
                score += 1
                shot_accuracy[shot][0] += 1
            else:
                user_dict[user] = False
            shot_accuracy[shot][1] += 1
        print("overall accuracy: ", score / len(lines))
        print("shot-wise accuracy: ", {k: v[0] / v[1] for k, v in shot_accuracy.items() if v[1] > 0})
        print("user-wise accuracy: ", sum(user_dict.values()) / len(user_dict))