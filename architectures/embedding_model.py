import torch.nn as nn

class EmbeddingModel(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super(EmbeddingModel, self).__init__()

        self.hidden = nn.Linear(input_size, hidden_size)
        self.output = nn.Linear(hidden_size, output_size)

    def forward(self, x):
        x = self.hidden(x)
        x = self.output(x)
        return x
    
    def get_embedding(self, x):
        emb = self.hidden(x)
        return emb
    