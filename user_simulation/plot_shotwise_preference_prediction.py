from matplotlib import pyplot as plt
if __name__ == '__main__':
    vanilla_prediction = {0: 0.4, 1: 0.8, 2: 0.6, 3: 0.6, 4: 0.6, 5: 0.6, 6: 0.8, 7: 0.5, 8: 0.6, 9: 0.6}
    gpt4_reflection_comment_prediction = {0: 0.7, 1: 0.7, 2: 0.7, 3: 0.6, 4: 0.9, 5: 0.7, 6: 0.7, 7: 0.6, 8: 0.7, 9: 0.4}
    llama3_reflection_comment_prediction = {0: 0.8, 1: 0.5, 2: 0.7, 3: 0.5, 4: 0.6, 5: 0.8, 6: 0.8, 7: 0.5, 8: 0.8, 9: 0.7}

    # plot the prediction
    xs = list(vanilla_prediction.keys())
    xs.sort()
    vanilla_ys = [vanilla_prediction[x] for x in xs]
    gpt4_reflection_comment_ys = [gpt4_reflection_comment_prediction[x] for x in xs]
    llama3_reflection_comment_ys = [llama3_reflection_comment_prediction[x] for x in xs]
    plt.plot(xs, vanilla_ys, label="Vanilla Preference Simulation")
    plt.plot(xs, gpt4_reflection_comment_ys, label="Reflection Comment Preference Simulation")
    plt.plot(xs, llama3_reflection_comment_ys, label="Llama3 Reflection Comment Preference Simulation")
    plt.xlabel("Shot Index")
    plt.ylabel("Preference Prediction Accuracy")
    plt.legend()
    plt.savefig("shotwise_preference_prediction.pdf")