import torch
import torch.nn as nn
import torch.optim as optim
from dataset_builders.stat_dataset_builder import StatDatasetBuilder
from architectures import StatModel
from train import PlayerEmbeddings
import pandas as pd
import numpy as np
from sklearn.metrics import mean_squared_error, mean_absolute_error


#
embedder = PlayerEmbeddings(embedding_size=100)
embs = embedder.load()

metric='PTS'
evals = [
    ('den', '@', 'dal'),
    ('dal', 'vs', 'den')
]
max_games=100
offset=0

pdb = StatDatasetBuilder(player_embeddings=embs, metric=metric, scaler='minmax', max_games=max_games, offset=offset)
pdb.set_batches()

#d schroder reb under
#j morant reb under
#j mcdaniels reb 6.0
#!c sexton pra 26.4 line 30.5
#j jaquez pra 16.5 line 22.5
#!d hunter pra 26.5 line 23.5
#r dunn pra 11.1 line 15.5
#houston pra unders?
#!j morant pra 28.5 line 34.5 + injury
#!d wade pra 15.7 line 11.5
#g niang pra 15.9 line 12.5
#r barrett hater bet?
#g dick ra again?
#d schroder pra 18.0 line 23.5


valid_evals = []
for e in evals:
    if (e[0] not in pdb.team_names or e[2] not in pdb.team_names):
        print(e)
    else:
        st = pdb.ext.get_team_stats(e[0])
        for p in st['PLAYER']:
            if p not in pdb.player_embeddings.keys():
                #print(p)
                continue
            else:
                valid_evals.append((p, e[1], e[2]))



batch_size = 16
train_loader, valid_loader, test_loader = pdb.get_dataloaders(batch_size=batch_size, train=0.8, test=0.1, valid=0.1)

input_size, output_size = pdb.get_io_sizes()
hidden_size = 100


model = StatModel(input_size=input_size, hidden_size=hidden_size, output_size=output_size)

criterion = nn.L1Loss()
optimizer = optim.Adam(model.parameters(), lr= 0.0001)


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)

num_epochs = 20
# Training loop
for epoch in range(num_epochs):
    model.train()  # Set the model to training mode
    running_loss = 0.0

    # Iterate over batches in the training set
    for batch_X, batch_y in train_loader:

        #print(f'batchx shape: {batch_X.shape}, batchy shape: {batch_y.shape}')

        # Move data to the appropriate device
        batch_X, batch_y = batch_X.to(device), batch_y.to(device)

        # Zero the gradients
        optimizer.zero_grad()

        # Forward pass
        outputs = model(batch_X)

        # Compute loss
        loss = criterion(outputs, batch_y)

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
            loss = criterion(outputs, batch_y)
            val_loss += loss.item()

            # Collect predictions and actual values for evaluation
            y_pred.append(outputs.cpu().numpy())
            y_true.append(batch_y.cpu().numpy())

        avg_val_loss = val_loss / len(valid_loader)
        print(f'Validation Loss: {avg_val_loss}')

        # You can also calculate other metrics like MSE, RMSE, etc.
        y_pred = np.concatenate(y_pred)
        y_true = np.concatenate(y_true)
        rmse = np.sqrt(mean_squared_error(y_true, y_pred))
        print(f'Validation RMSE: {rmse}')

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

        # Collect predictions and actual values
        y_pred.append(outputs.cpu().numpy())
        y_true.append(batch_y.cpu().numpy())

    avg_test_loss = test_loss / len(test_loader)
    print(f'Test Loss: {avg_test_loss}')

    # Calculate RMSE or other metrics on the test set
    y_pred = np.concatenate(y_pred)
    y_true = np.concatenate(y_true)
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    print(f'Test RMSE: {rmse}')


    denorm_y_pred = pdb.denormalize(y_pred)
    denorm_y_true = pdb.denormalize(y_true)
    
    rmse = np.sqrt(mean_squared_error(denorm_y_true, denorm_y_pred))
    print(f'Test RMSE for {metric} (denormalized): {rmse}')

    mae = mean_absolute_error(denorm_y_true, denorm_y_pred)
    print(f'Test MAE for {metric} (denormalized): {mae}')

    comp = np.column_stack((np.round(denorm_y_pred[:10]),
                            np.round(denorm_y_true[:10])))

    print(f'Predicted vs Actual (first 10 elements)')
    print(comp)



def run(e):

    p, l, t = e
    # Example input array
    input_array = pdb.build_input(p, l, t)  # This is a (92,) numpy array

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

    denorm = pdb.denormalize(output.item())
    return (p, l, t, denorm)


es = []
for e in valid_evals:
    es.append(run(e))


es = pd.DataFrame(es, columns=['player', 'loc', 'opp', 'pred'])
print(metric)
pd.set_option('display.max_rows', None)  # Show all rows
pd.set_option('display.max_columns', None)  # Show all columns
pd.set_option('display.width', None)  # Adjust width to avoid truncation
pd.set_option('display.max_colwidth', None)  # Prevent truncation of column contents

grouped = es.groupby('opp')

for group_name, group_data in grouped:
    print(f'group: {group_name}')
    print(group_data)
    print('-' * 40)

#print(es)


#torch.save({
#    'model_state_dict': model.state_dict()
#}, 'Models/model_ou_24_5.pth')