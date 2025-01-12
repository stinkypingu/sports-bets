import torch.nn as nn

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
