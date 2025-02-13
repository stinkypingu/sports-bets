import torch
import torch.nn as nn
import torch.optim as optim
from dataset_builders.player_embedding_dataset_builder import PlayerEmbeddingDatasetBuilder
from architectures import EmbeddingSkipGramModel, EmbeddingCBOWModel
import pandas as pd
import numpy as np
from itertools import combinations
import logging
import os
from pathlib import Path
import json
 
class PlayerEmbeddings():
    def __init__(self, req_games=5, max_games=100, offset=0, ignore_teams=False, embedding_size=100, lr=0.001, epochs=30, emb_dir='embeddings', method='skipgram'):
        """
        Initializes the PlayerEmbeddings object. Use this by calling load(), which trains and saves if necessary, or reads from 
        an existing json file. Then, call using PlayerEmbeddings(player) to get an embedding.

        NOTE: Only embedding_size is used for json file retrieval currently.

        Args:
            req_games (int, optional): Minimum number of games required for a player to be included. Defaults to 5.
            max_games (int, optional): Maximum number of games to consider for each player. Defaults to 100.
            offset (int, optional): Offset for player data filtering. Defaults to 0.
            ignore_teams

            embedding_size (int, optional): Size of the embedding vector. Defaults to 100.
            lr (float, optional): Learning rate for the model training. Defaults to 0.001.
            epochs (int, optional): Number of epochs for model training. Defaults to 30.

            emb_dir
            method
        
        Sets up the file paths, directories, and logging configurations for training and saving player embeddings.
        """
        #set up file system for caching
        self.root_dir = Path(__file__).resolve().parent.parent
        self.emb_dir = self.root_dir / emb_dir

        #filename will be simple for now, just the embedding dimension
        self.emb_filename = f'emb_{method}_size{embedding_size}_epochs{epochs}_ignoreteams{int(ignore_teams)}.json'
        self.emb_file = self.emb_dir / self.emb_filename

        #contains the actual embeddings
        self.embeddings = None

        #databuilder params
        self.req_games = req_games
        self.max_games = max_games
        self.offset = offset
        self.ignore_teams = ignore_teams
        
        #model training params
        self.method = method
        self.embedding_size = embedding_size
        self.lr = lr
        self.epochs = epochs
        
        #logger setup
        self.logger = logging.getLogger(self.__class__.__name__)
        self.logger.setLevel(logging.INFO)
        handler = logging.StreamHandler()
        formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
        handler.setFormatter(formatter)
        self.logger.addHandler(handler)
        self.logger.propagate = False  # Prevent logging from propagating to the root logger


    def retrieve_embeddings(self, emb_dir=None, emb_filename=None):
        """
        Gets presaved embeddings from an embeddings file. Modifies existing file structure to find the file, or uses presets if
        not specified.

        Args:
            emb_dir (str, optional): Directory name of the embeddings.
            emb_filename (str, optional): Filename without .csv extension
        
        Returns:
            embeddings (dict): Dictionary mapping player names as strings to np.arrays of embeddings.
        """
        #set new directory structure, or use presets if not specified
        if emb_dir is None:
            emb_dir = self.emb_dir
        else:
            emb_dir = self.root_dir / emb_dir
        
        if emb_filename is None:
            emb_filename = self.emb_filename
        else:
            emb_filename = f'{emb_filename}.json'

        #retrieve the file data
        emb_file = emb_dir / emb_filename
        self.logger.info(f'Retrieving embeddings from file: {emb_file}')
        
        assert(emb_file.exists())
        with open(emb_file, 'r') as f:
            embeddings = json.load(f)

        #return to np from serialized lists
        for player, emb in embeddings.items():
            embeddings[player] = np.array(emb)

        return embeddings




    def load(self, update=False, save=False):
        """
        Loads player embeddings from a file if available, otherwise trains and saves them.

        Args:
            update (bool, optional): Whether to force retraining even if the embeddings file exists. Defaults to False.
            save (bool, optional): Whether to save to a file. Defaults to False

        Returns:
            self.embeddings (dict): Dictionary of embeddings
        
        This method checks if the embeddings file exists. If the file is found and the `update` flag is set to `False`, 
        it loads the embeddings from the file into the `self.embeddings` attribute. If the file does not exist or if `update` is `True`, 
        it will train the embeddings by calling the `train_and_save` method, and then save the generated embeddings to a file.
        """
        if self.emb_file.exists() and not update and not save:
            self.logger.info(f'Loading embeddings from {self.emb_file}.')
            with open(self.emb_file, 'r') as f:
                self.embeddings = json.load(f)
            
            #return to np from serialized lists
            for player, emb in self.embeddings.items():
                self.embeddings[player] = np.array(emb)

            return self.embeddings

        else:
            self.logger.info(f'Embedding file {self.emb_file} not found or explicitly forcing update. Training embeddings.')

            #which model to embed with
            self.train()

            if save:
                self.save()
            else:
                self.logger.info(f'Not saved to file.')

            return self.embeddings
        
            
    def train(self):
        if self.method == 'skipgram':
            self.logger.info(f'Training using SKIP-GRAM')
            self.train_skipgram()
        elif self.method == 'cbow':
            self.logger.info(f'Training using CBOW')
            self.train_cbow()
        else:
            self.logger.error(f'Invalid training method.')
            raise


    
    def train_skipgram(self):
        """
        Trains the player embeddings using the specified dataset and model parameters. Stores to self.embeddings.

        This method builds the dataset for training using the `PlayerEmbeddingDatasetBuilder`, sets up the model,
        loss function, and optimizer, and performs the training cycle over multiple epochs. After training, the method
        computes the embeddings for each player and stores them in the `self.embeddings` DataFrame.

        The training process includes:
            - Initializing the dataset builder and batches.
            - Setting up a neural network model.
            - Running the training loop, calculating loss, and updating model weights.
            - Computing player embeddings after training and storing them in a pandas DataFrame.

        Returns:
            None
        """
        self.logger.info(f'Building training dataset for player embeddings...')

        #build the dataset to train on
        databuilder = PlayerEmbeddingDatasetBuilder(req_games=self.req_games, max_games=self.max_games, offset=self.offset, ignore_teams=self.ignore_teams)
        databuilder.set_logger_level(logging.INFO)
        databuilder.set_batches()

        batched_data = databuilder.get_batches(batch_size=16)

        #model setup
        input_size, output_size = databuilder.get_io_sizes()
        model = EmbeddingSkipGramModel(input_size=input_size, hidden_size=self.embedding_size, output_size=output_size)

        #loss function and optimizer
        self.logger.info(f'Training with Smooth L1 Loss, learning rate {self.lr} for {self.epochs} epochs.')
        criterion = nn.SmoothL1Loss() #sensitive to small differences, but less sensitive to large differences that might be outliers (steph hitting 2pts in a game)
        optimizer = optim.Adam(model.parameters(), lr=self.lr)

        #training cycle
        model.train()
        for epoch in range(self.epochs):
            total_loss = 0
            for batch in batched_data:
                x_batch, y_batch = batch

                output = model(x_batch)
                loss = criterion(output, y_batch)

                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

                total_loss += loss.item()

            self.logger.info(f'Epoch {epoch + 1}/{self.epochs}, Loss: {total_loss / len(batched_data):.4f}')
        
        #collect the embeddings
        valid_players = databuilder.get_valid_players()
        self.logger.info(f'Computing embeddings for {len(valid_players)} players.')

        #iterate through each player and compute their embedding, adding it to a list and then converting to a dataframe
        data = {}
        model.eval()
        with torch.no_grad():
            for player in valid_players:
                onehot = databuilder.get_onehot_player(player)
                onehot_tensor = torch.tensor(onehot, dtype=torch.float32)

                emb = model.get_embedding(onehot_tensor)
                emb = emb.numpy()

                data[player] = emb
        
        #cache this
        self.embeddings = data
        return



    def train_cbow(self):
        """
        Trains the player embeddings using the specified dataset and model parameters. Stores to self.embeddings.

        This method builds the dataset for training using the `PlayerEmbeddingDatasetBuilder`, sets up the model,
        loss function, and optimizer, and performs the training cycle over multiple epochs. After training, the method
        computes the embeddings for each player and stores them in the `self.embeddings` DataFrame.

        The training process includes:
            - Initializing the dataset builder and batches.
            - Setting up a neural network model.
            - Running the training loop, calculating loss, and updating model weights.
            - Computing player embeddings after training and storing them in a pandas DataFrame.

        Returns:
            None
        """
        self.logger.info(f'Building training dataset for player embeddings...')

        #build the dataset to train on
        databuilder = PlayerEmbeddingDatasetBuilder(req_games=self.req_games, max_games=self.max_games, offset=self.offset, ignore_teams=self.ignore_teams)
        databuilder.set_logger_level(logging.INFO)
        databuilder.set_batches()

        batched_data = databuilder.get_batches(batch_size=16)

        #model setup, swapped from the skipgram version since inputs and outputs are swapped
        output_size, input_size = databuilder.get_io_sizes()
        model = EmbeddingCBOWModel(input_size=input_size, hidden_size=self.embedding_size, output_size=output_size)

        #loss function and optimizer
        self.logger.info(f'Training with Cross Entropy Loss, learning rate {self.lr} for {self.epochs} epochs.')
        criterion = nn.CrossEntropyLoss()
        optimizer = optim.Adam(model.parameters(), lr=self.lr)

        #training cycle
        model.train()
        for epoch in range(self.epochs):
            total_loss = 0
            for batch in batched_data:
                y_batch, x_batch = batch #swapped with the skipgram version, since inputs and outputs are swapped for cbow

                output = model(x_batch)
                loss = criterion(output, y_batch)

                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

                total_loss += loss.item()

            self.logger.info(f'Epoch {epoch + 1}/{self.epochs}, Loss: {total_loss / len(batched_data):.4f}')
        
        #collect the embeddings
        valid_players = databuilder.get_valid_players()
        self.logger.info(f'Computing embeddings for {len(valid_players)} players.')

        #iterate through each player and compute their embedding, adding it to a list and then converting to a dataframe
        data = {}
        model.eval()
        with torch.no_grad():
            for player in valid_players:
                onehot = databuilder.get_onehot_player(player)
                onehot_tensor = torch.tensor(onehot, dtype=torch.float32)

                emb = model.get_embedding(onehot_tensor)
                emb = emb.numpy()
                
                data[player] = emb
        
        #cache this
        self.embeddings = data
        return
    




    def save(self):
        """
        Saves the computed player embeddings to a CSV file.

        This method saves the embeddings stored in `self.embeddings` to a CSV file at the location specified by
        `self.emb_file`. The embeddings are saved in a format with columns 'PLAYER' and 'EMBEDDING'.

        Returns:
            None
        """
        assert(self.embeddings is not None)

        #ensure parent directories exists
        self.emb_file.parent.mkdir(parents=True, exist_ok=True)

        self.logger.info(f'Saving embeddings to file: {self.emb_file}')
        with open(self.emb_file, 'w') as f:
            embs = {k: v.tolist() for k,v in self.embeddings.items()}
            json.dump(embs, f) #note that this automatically parses np arrays as lists so they must be reconverted when extracting later
        return




'''
    def __call__(self, player):
        """
        Retrieves the embedding for the specified player by name.

        Args:
            player_name (str): The name of the player whose embedding is to be retrieved.
        
        Returns:
            numpy.ndarray or None: The embedding vector for the player, or None if the player is not found.
        
        This method allows you to call an instance of the PlayerEmbeddings class as if it were a function, passing the player's name
        as the argument. If the embeddings are not already loaded, it will first load them. If the player is found in the embeddings, 
        the corresponding embedding is returned as a numpy array. If the player is not found, a warning is logged and None is returned.
        """
        #ensure embeddings are loaded
        if self.embeddings is None:
            self.load()

        #retrieve the corresponding embedding
        player_row = self.embeddings[self.embeddings['PLAYER'] == player]

        if not player_row.empty:
            return player_row['EMBEDDING'].iloc[0]
        else:
            self.logger.error(f'Player ({player}) not found in embeddings.')
'''


