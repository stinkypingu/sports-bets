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
    


class PredictionModel(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super(PredictionModel, self).__init__()

        self.hidden = nn.Linear(input_size, hidden_size)
        self.relu = nn.ReLU()  # Adding ReLU activation function
        self.output = nn.Linear(hidden_size, output_size)

    def forward(self, x):
        x = self.hidden(x)  # Pass through the hidden layer
        x = self.relu(x)  # Apply ReLU activation to hidden layer output
        out = self.output(x)  # Final output layer, no activation (linear)
        return out
    

class StatModel(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super(StatModel, self).__init__()

        self.hidden = nn.Linear(input_size, hidden_size)
        self.relu = nn.ReLU()  # Adding ReLU activation function
        self.output = nn.Linear(hidden_size, output_size)

    def forward(self, x):
        x = self.hidden(x)  # Pass through the hidden layer
        x = self.relu(x)  # Apply ReLU activation to hidden layer output
        out = self.output(x)  # Final output layer, no activation (linear)
        return out



class OverUnderModel(nn.Module):
    def __init__(self, input_size, hidden_size, output_size=2):
        super(OverUnderModel, self).__init__()
        self.hidden = nn.Linear(input_size, hidden_size)
        self.relu = nn.ReLU()  # Adding ReLU activation function
        self.output = nn.Linear(hidden_size, output_size)  #two output neurons for over and under

    def forward(self, x):
        x = self.hidden(x)  # Pass through the hidden layer
        x = self.relu(x)  # Apply ReLU activation
        out = self.output(x)  # Final output layer
        return out