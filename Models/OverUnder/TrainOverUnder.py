import torch
import torch.nn as nn
import torch.optim as optim
from DatasetBuilders.OverUnderDatasetBuilder import OverUnderDatasetBuilder
from Models.Architectures import OverUnderModel
import pandas as pd
import numpy as np
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix

max_games=10
offset=1
required_minutes=20
significant_minutes=15
hidden_size = 100

#
#format is max_games, offset, req_min, sig_min, hidden_size
embedding_file = f'emb_player_{max_games}_{offset}_{required_minutes}_{significant_minutes}_{hidden_size}.csv'

metric='PTS'
#threshold=13.5
threshold=19.5

max_games=20

pdb = OverUnderDatasetBuilder()
pdb.set_batches(embeddings_file=embedding_file, metric=metric, threshold=threshold, max_games=max_games, offset=offset)

batch_size = 16
train_loader, valid_loader, test_loader = pdb.get_dataloaders(batch_size=batch_size)
print(len(train_loader) * batch_size)
print(len(valid_loader) * batch_size)
print(len(test_loader) * batch_size)

input_size, output_size = pdb.get_io_sizes()
hidden_size = 100


model = OverUnderModel(input_size=input_size, hidden_size=hidden_size, output_size=output_size)

#under, over weighting to improve precision?
class_weights = torch.tensor([1.0, 1.0])
criterion = nn.CrossEntropyLoss(weight=class_weights)
optimizer = optim.Adam(model.parameters(), lr= 0.0001)


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)

num_epochs = 30
# Training loop
for epoch in range(num_epochs):
    model.train()  # Set the model to training mode
    running_loss = 0.0

    # Iterate over batches in the training set
    for batch_X, batch_y in train_loader: 
        # Move data to the appropriate device
        batch_X, batch_y = batch_X.to(device), batch_y.to(device)

        # Zero the gradients
        optimizer.zero_grad()

        # Forward pass
        outputs = model(batch_X)

        # Compute loss
        loss = criterion(outputs.squeeze(), batch_y)

        # Backward pass and optimization
        loss.backward()
        optimizer.step()

        running_loss += loss.item()

    # Print the average loss for the epoch
    print(f'Epoch [{epoch+1}/{num_epochs}], Loss: {running_loss/len(train_loader)}')

    # Validation loop (optional, to monitor the validation performance)
    model.eval()  # Set the model to evaluation mode
    with torch.no_grad():  # No gradients are needed for validation
        val_loss = 0.0
        y_pred = []
        y_true = []
        
        for batch_X, batch_y in valid_loader:
            batch_X, batch_y = batch_X.to(device), batch_y.to(device)

            # Forward pass
            outputs = model(batch_X)
            loss = criterion(outputs.squeeze(), batch_y)
            val_loss += loss.item()

            # Collect predictions and actual values for evaluation
            y_pred.append(outputs.squeeze().cpu().numpy())
            y_true.append(batch_y.cpu().numpy())

        avg_val_loss = val_loss / len(valid_loader)
        print(f'Validation Loss: {avg_val_loss}')

        # You can also calculate other metrics like MSE, RMSE, etc.
        y_pred = np.concatenate(y_pred)
        y_true = np.concatenate(y_true)
        
        #print(f'Validation RMSE: {rmse}')

# After training, test the model on the test set
model.eval()  # Set the model to evaluation mode for testing
with torch.no_grad():
    test_loss = 0.0
    y_pred = []
    y_true = []

    for batch_X, batch_y in test_loader:
        batch_X, batch_y = batch_X.to(device), batch_y.to(device)

        # Forward pass
        outputs = model(batch_X)
        loss = criterion(outputs, batch_y)
        test_loss += loss.item()

        probabilities = torch.softmax(outputs, dim=1)

        # Collect predictions and actual values
        y_pred.append(probabilities.cpu().numpy())
        y_true.append(batch_y.cpu().numpy())

    avg_test_loss = test_loss / len(test_loader)
    print(f'Test Loss: {avg_test_loss}')

    # Calculate RMSE or other metrics on the test set
    y_pred = np.concatenate(y_pred)
    y_true = np.concatenate(y_true)

    predicted_classes = np.argmax(y_pred, axis=1)
    true_classes = np.argmax(y_true, axis=1)
    
    pos = np.sum(true_classes)
    neg = len(true_classes) - pos
    print(f'p: {pos}, n: {neg}, all: {pos+neg}')

    # Accuracy
    accuracy = accuracy_score(true_classes, predicted_classes)

    # Precision (for each class or averaged)
    precision = precision_score(true_classes, predicted_classes, average='binary')  # Use 'binary' for 2 classes
    recall = recall_score(true_classes, predicted_classes, average='binary')
    f1 = f1_score(true_classes, predicted_classes, average='binary')

    # Confusion matrix
    conf_matrix = confusion_matrix(true_classes, predicted_classes)

    print(f"Accuracy: {accuracy:.2f}")
    print(f"Precision: {precision:.2f}")
    print(f"Recall: {recall:.2f}")
    print(f"F1 Score: {f1:.2f}")
    print("Confusion Matrix:\n", conf_matrix)


# Example input array
input_array = pdb.build_input('nikola-jokic', '@', 'sa')  # This is a (92,) numpy array

# Step 1: Add batch dimension to make it (1, 92)
input_array = input_array[np.newaxis, :]  # Shape becomes (1, 92)

# Step 2: Convert to PyTorch tensor
input_tensor = torch.tensor(input_array, dtype=torch.float32)

# Step 3: Move to the same device as the model
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
input_tensor = input_tensor.to(device)

# Step 4: Run the model
model.eval()  # Set the model to evaluation mode
with torch.no_grad():  # No gradients needed for inference
    output = model(input_tensor)

# Step 5: Process the output (e.g., apply softmax if logits are returned)
probabilities = torch.softmax(output, dim=1)  # Convert logits to probabilities
print(probabilities)