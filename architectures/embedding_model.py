import torch.nn as nn
import torch.nn.functional as F

class EmbeddingSkipGramModel(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super(EmbeddingSkipGramModel, self).__init__()

        self.hidden = nn.Linear(input_size, hidden_size)
        self.output = nn.Linear(hidden_size, output_size)

    def forward(self, x):
        x = self.hidden(x)
        x = self.output(x)
        return x
    
    def get_embedding(self, x):
        emb = self.hidden(x)
        return emb
    
class EmbeddingCBOWModel(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super(EmbeddingCBOWModel, self).__init__()

        self.hidden = nn.Linear(input_size, hidden_size)
        self.output = nn.Linear(hidden_size, output_size)

    def forward(self, x):
        x = self.hidden(x)
        x = self.output(x)
        return x
    
    def get_embedding(self, x):
        x = list(x).index(1)

        emb = self.output.weight[x]
        return emb