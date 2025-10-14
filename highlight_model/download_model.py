from transformers import AutoModel


if __name__ == '__main__':
    model = AutoModel.from_pretrained("mistralai/Mixtral-8x22B-Instruct-v0.1")