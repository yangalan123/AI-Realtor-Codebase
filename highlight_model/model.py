import torch
class SimpleClassifier(torch.nn.Module):
    def __init__(self, input_size, intermediate_dims, output_size):
        super(SimpleClassifier, self).__init__()
        self.intermediate_dims = intermediate_dims
        if intermediate_dims is not None:
            self.ff = torch.nn.Sequential(
                torch.nn.Linear(input_size, intermediate_dims),
                torch.nn.ReLU(),
                torch.nn.Linear(intermediate_dims, output_size),
                torch.nn.Sigmoid()
            )
        else:
            # self.ff = torch.nn.Sequential(
            #     torch.nn.Linear(input_size, output_size),
            #     torch.nn.Sigmoid()
            # )
            self.fc = torch.nn.Linear(input_size, output_size)
            self.sigmoid = torch.nn.Sigmoid()

    def forward(self, x):
        if self.intermediate_dims is not None:
            x = self.ff(x)
        else:
            x = self.sigmoid(self.fc(x))
        return x
        # return self.sigmoid(x)


def collate_fn_mlp(examples):
    inputs = []
    labels = []
    ids = []
    scores = []
    for example in examples:
        inputs.append(torch.tensor(example["input"]))
        labels.append(torch.tensor(example["labels"]))
        ids.append(example["id"])
        scores.append(example["score"])
        # scores.append(torch.tensor([example["score"] for _ in range(len(example["labels"]))]))

    inputs = torch.stack(inputs)
    labels = torch.stack(labels)
    scores = torch.tensor(scores)
    return {"input": inputs, "labels": labels, "id": ids, "score": scores}

class XFMRClassifier(torch.nn.Module):
    def __init__(self, input_size, output_size, num_heads=8, num_layers=6):
        super(XFMRClassifier, self).__init__()
        encoder_layer = torch.nn.TransformerEncoderLayer(d_model=input_size, nhead=num_heads, batch_first=True)
        self.xfmr = torch.nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        self.fc = torch.nn.Linear(input_size, output_size)
        self.sigmoid = torch.nn.Sigmoid()

    def forward(self, x, mask):
        x = self.xfmr(x, mask=mask)
        # get first position of x
        x_aggr = x[:, 0, :]
        x_fc = self.fc(x_aggr)
        return self.sigmoid(x_fc)
