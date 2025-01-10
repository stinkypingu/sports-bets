from Extractors.ESPNExtractor import ESPNNBAExtractor
from abc import ABC, abstractmethod
import logging
import numpy as np
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.model_selection import train_test_split
import torch
from torch.utils.data import DataLoader, TensorDataset

class DatasetBuilder(ABC):
    def __init__(self):

        self.ext = ESPNNBAExtractor()
        self.ext.set_logger_level(logging.INFO)
        
        #placeholders for dataset inputs and outputs
        self.X = np.empty((0, 0))  # Inputs
        self.Y = np.empty((0, 0))  # Outputs

        self.standard_scaler = StandardScaler()
        self.minmax_scaler = MinMaxScaler()

        #logger setup
        self.logger = logging.getLogger(self.__class__.__name__)
        self.logger.setLevel(logging.INFO)
        handler = logging.StreamHandler()
        formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
        handler.setFormatter(formatter)
        self.logger.addHandler(handler)
        self.logger.propagate = False  # Prevent logging from propagating to the root logger




    #----------------------------------------------------------
    #change logging level
    def set_logger_level(self, level):
        """Change the logging level for the logger and its handlers."""
        if isinstance(level, int) or level in logging._nameToLevel:
            self.logger.setLevel(level)
            for handler in self.logger.handlers:
                handler.setLevel(level)
        else:
            raise ValueError(f"Invalid logging level: {level}. Use one of {list(logging._nameToLevel.keys())}.")
        return




    def log_data_shapes(self):
        """
        Logs the shapes of the dataset inputs and outputs.
        """
        self.logger.info(f"Input shape (X): {self.X.shape}")
        self.logger.info(f"Output shape (Y): {self.Y.shape}")


    @abstractmethod
    def set_batches(self):
        raise NotImplementedError(f'Subclasses must implement this method.')


    #----------------------------------------------------------
    def get_batches(self, batch_size=16):
        """
        Splits the dataset into batches and returns the batches as PyTorch tensors.

        Args:
            batch_size (int): The size of each batch. Default is 8.

        Returns:
            batches (zip): A zip object containing pairs of (x_batch, y_batch),
                        where each batch is a tuple of input features (X) 
                        and corresponding target labels (Y), both in PyTorch tensor format.
        """
        assert(self.X.shape[0] == self.Y.shape[0])
        num_data = self.X.shape[0]

        #leverage np to do the batching for us
        num_batches = np.ceil(num_data / batch_size)

        x_batches = np.array_split(self.X, num_batches)
        y_batches = np.array_split(self.Y, num_batches)

        #convert batches to torch tensors
        x_batches = [torch.tensor(x_batch, dtype=torch.float32) for x_batch in x_batches]
        y_batches = [torch.tensor(y_batch, dtype=torch.float32) for y_batch in y_batches]

        #collect the X and Y batches together to use as training data
        batches = list(zip(x_batches, y_batches))
        return batches


    def get_tensor_data(self):
        X_tensor = torch.tensor(self.X, dtype=torch.float32)
        Y_tensor = torch.tensor(self.Y, dtype=torch.float32)
        return X_tensor, Y_tensor
    

    def get_dataloaders(self, batch_size=16, train=0.7, valid=0.15, test=0.15):

        assert(train + valid + test == 1)

        #cut data into training, validation, and testing splits
        #split into training and non-training parts
        percentage_temp = (1 - train)
        X_train, X_temp, y_train, y_temp = train_test_split(self.X, self.Y, test_size=percentage_temp, random_state=42)

        #split the non-training part into fractional parts
        percentage_test = (test / (test + valid))
        X_val, X_test, y_val, y_test = train_test_split(X_temp, y_temp, test_size=percentage_test, random_state=42)

        self.logger.debug(f'Training set size: {len(X_train)}, Validation set size: {len(X_val)}, Test set size: {len(X_test)}')
        
        #convert data into tensors (if not already)
        X_train_tensor = torch.tensor(X_train, dtype=torch.float32)
        y_train_tensor = torch.tensor(y_train, dtype=torch.float32)
        X_val_tensor = torch.tensor(X_val, dtype=torch.float32)
        y_val_tensor = torch.tensor(y_val, dtype=torch.float32)
        X_test_tensor = torch.tensor(X_test, dtype=torch.float32)
        y_test_tensor = torch.tensor(y_test, dtype=torch.float32)

        #create TensorDataset for each split
        train_dataset = TensorDataset(X_train_tensor, y_train_tensor)
        val_dataset = TensorDataset(X_val_tensor, y_val_tensor)
        test_dataset = TensorDataset(X_test_tensor, y_test_tensor)

        #create DataLoader for each split
        train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
        val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
        test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

        train_msg = f'{len(train_loader)} training samples ({int(100 * train)}%)'
        val_msg = f'{len(val_loader)} training samples ({int(100 * valid)}%)'
        test_msg = f'{len(test_loader)} training samples ({int(100 * test)}%)'
        self.logger.info(f'Split data into {train_msg}, {val_msg}, {test_msg}.')

        return train_loader, val_loader, test_loader


    def get_io_sizes(self):
        return self.X.shape[1], self.Y.shape[1]
