from DatasetBuilders.DatasetBuilder import DatasetBuilder
import pandas as pd 
import numpy as np
from itertools import combinations
import logging

class PlayerEmbeddingDatasetBuilder(DatasetBuilder):
    """
    A DatasetBuilder subclass for creating player embedding datasets.

    Args:
        max_games (int): The maximum number of games to include in each player's match history.
        required_minutes (int): Minimum number of minutes played to include a player in training.
        significant_minutes (int): Minimum number of minutes played to include a player in team averages. (Recommended <=15)
    """
    def __init__(self, max_games=20, offset=0, required_minutes=20, significant_minutes=15): 
        super().__init__()

        #values for this databuilder
        self.max_games = max_games #maximum number of games to go into each player's match history
        self.offset = offset #amount of recent games to ignore before counting match history
        self.required_minutes = required_minutes #players to consider for training
        self.significant_minutes = significant_minutes #players to consider for calculating team average

        self.select_columns = ['+/-', 'OREB', 'DREB', 'REB', 'AST', 'STL', 'BLK', 'TO', 'PF', 'PTS', 'FGM', 'FGA', '3PM', '3PA', 'FTM', 'FTA']
        self.zscore_columns = [['+/-'],
                               ['PTS']]
        self.minmax_columns = [['OREB', 'DREB', 'REB'],
                               ['AST'],
                               ['STL'],
                               ['BLK'],
                               ['TO'],
                               ['PF'],
                               ['FGM', 'FGA'],
                               ['3PM', '3PA'],
                               ['FTM', 'FTA']]

        #mapping from player to team, and mapping for encodable players
        self.player_to_team = self.ext.get_player_to_team()
        self.encoded_players = {player: idx for idx, player in enumerate(self.player_to_team.keys())} #initially this is every possible player

        #numpy arrays of the inputs and corresponding outputs
        self.X = np.empty((0, len(self.encoded_players)))
        self.Y = np.empty((0, len(self.select_columns) * 3))



    #----------------------------------------------------------
    def get_valid_players(self):
        return self.encoded_players.keys()



    #----------------------------------------------------------
    def recent_games(self, player):
        """
        Extract the most recent games from the player's match history.

        Args:
            player (str): The name or identifier of the player whose match history is being retrieved.

        Returns:
            pandas.DataFrame: A DataFrame containing the most recent games of the player, 
                            limited to `self.max_games` rows.
        """
        #extract the max_games most recent games from the player's match history
        game_log = self.ext.get_player_game_log(player)
        
        recent = game_log.iloc[self.offset:].head(self.max_games)
        return recent
    


    #----------------------------------------------------------
    def filter_players(self):
        """
        Filters players based on their average minutes played and encodes the qualified players.

        This function resets `self.encoded_players` and creates a subset of players 
        who meet the `self.required_minutes` threshold for average minutes played in their 
        recent games. Players who qualify are assigned an encoding index.

        Args:
            None

        Returns:
            self.encoded_players (dict): Dictionary containing valid players with indices for onehot encoding.
        """
        #resets the encoded_players and limits it to a filtered subset based on average minutes played
        self.encoded_players = {}

        #each player gets is checked to have an average minutes played above a certain cutoff
        idx = 0
        for player in self.player_to_team.keys():
            game_log = self.recent_games(player)

            if len(game_log) > 0:
                avg_min = game_log['MIN'].mean()

                if avg_min < self.required_minutes:
                    self.logger.debug(f'Player ({player}) does not meet minimum average minutes ({avg_min} < {self.required_minutes}) ' +
                                    f'for past ({len(game_log)}) games.')
                else:
                    self.encoded_players[player] = idx
                    idx += 1

        #change the shape of X for efficient processing
        self.X = np.empty((0, len(self.encoded_players)))
        self.logger.info(f'Filtered number of players from {len(self.player_to_team)} to {len(self.encoded_players)}')

        return self.encoded_players



    #----------------------------------------------------------
    def get_onehot_player(self, player):
        """
        Generates a one-hot encoded vector for the specified player.

        Args:
            player (str): The name or identifier of the player to be one-hot encoded.

        Returns:
            numpy.ndarray: A NumPy array of length `len(self.encoded_players)` with 
                        a 1 at the index corresponding to the player's encoding 
                        and 0 elsewhere.
        """
        onehot = np.zeros(len(self.encoded_players))
        idx = self.encoded_players.get(player)
        if idx is None:
            self.logger.error(f'Player ({player}) not in valid players to onehot encode.')
            return None
        else:
            onehot[idx] = 1
            return onehot
    
    def get_player_vector(self, player):
        """
        Gets a vector embedding for the specified player.

        Args:
            player (str): The name or identifier of the player to get a vector embedding for

        Returns:
            numpy.ndarray: A NumPy array of length `embedding size`.
        """
        if self.player_embeddings is None:
            self.logger.error(f'Failed to find any player vector embeddings. Save vector embeddings with set_player_embeddings().')
            raise ValueError("Player vector embeddings have not been set.")
        
        emb = self.player_embeddings.get(player)
        if emb is None:
            self.logger.error(f'Failed to find player ({player}) in vector embeddings.')
            raise KeyError(f"Player '{player}' not found in vector embeddings.")
        
        return emb



    #----------------------------------------------------------
    #custom standard scaler that operates on the same scale for multiple columns
    def z_score_normalize_inplace(self, df, columns):
        """
        Perform Z-score normalization across multiple columns, normalizing them based on 
        the combined mean and standard deviation of all the selected columns (modifies the original dataframe).
        
        Args:
        - df (pd.DataFrame): The dataframe containing the columns to normalize.
        - columns (list): A list of column names to normalize.
        
        Returns:
        - None: The dataframe is modified in place.
        """
        # Combine all selected columns into one DataFrame to compute global mean and std
        combined_df = df[columns]
        
        # Calculate global mean and std across all the selected columns
        combined_mean = combined_df.mean().mean()  # Mean of all values in the selected columns
        combined_std = combined_df.stack().std()  # Std of all values in the selected columns
        
        # Apply the same Z-score normalization using the global mean and std
        for column in columns:
            df[column] = (df[column] - combined_mean) / combined_std

    #custom minmax scaler that operates on the same scale for multiple columns
    def min_max_normalize_inplace(self, df, columns):
        """
        Perform Min-Max normalization across multiple columns, normalizing them based on 
        the combined min and max values of all the selected columns (modifies the original dataframe).
        
        Args:
        - df (pd.DataFrame): The dataframe containing the columns to normalize.
        - columns (list): A list of column names to normalize.
        
        Returns:
        - None: The dataframe is modified in place.
        """
        # Combine all selected columns into one DataFrame to compute global min and max
        combined_df = df[columns]
        
        # Calculate global min and max across all the selected columns
        combined_min = combined_df.min().min()  # Min value across all selected columns
        combined_max = combined_df.max().max()  # Max value across all selected columns
        
        # Apply Min-Max scaling using the global min and max
        for column in columns:
            df[column] = (df[column] - combined_min) / (combined_max - combined_min)

    def normalize_outputs(self):
        """
        Normalizes the output data (`self.Y`) column-wise using appropriate scaling methods
        and returns the scaled input (`self.X`) and output (`self.Y`) arrays.

        Columns with negative values are scaled using StandardScaler (Z-score scaling),
        while columns with only non-negative values are scaled using MinMaxScaler.

        Returns:
        --------
        tuple:
            - self.X: The input data (unchanged in this method).
            - self.Y: The normalized output data.
        """
        assert not np.isnan(self.X).any(), "Input data contains NaN values before scaling."
        assert not np.isnan(self.Y).any(), "Input data contains NaN values before scaling."

        df = pd.DataFrame(self.Y, columns=(self.select_columns * 3))

        #print(df['PTS'])
        
        #some columns need to be put on standard scaling, then squished into 0-1 range with minmax. PTS and +/-
        for group_col in self.zscore_columns:
            self.z_score_normalize_inplace(df, group_col)
            self.min_max_normalize_inplace(df, group_col)

            self.logger.debug(f'Zscore normalized, then Minmax normalized columns: {group_col}')

        #all other stats are minmaxed, but some of them in groups. OREB, DREB, and REB are all minmaxed together on the same scale.
        for group_col in self.minmax_columns:
            self.min_max_normalize_inplace(df, group_col)

            self.logger.debug(f'Minmax normalized columns: {group_col}')

        #print(df['PTS'])

        Y_scaled = df.to_numpy()
        assert(self.Y.shape == Y_scaled.shape)
        self.logger.info(f'Completed normalization for outputs.')

        self.Y = Y_scaled
        return self.X, self.Y



    #----------------------------------------------------------
    def set_pre_normalized_io(self):
        """
        Generates pre-normalized input (X) and output (Y) datasets for all players.

        Iterates through each player in `self.encoded_players`, collects their game logs, 
        and processes their inputs and outputs for training.

        Returns:
        --------
        tuple:
            - self.X: A NumPy array containing input features for all players.
            - self.Y: A NumPy array containing output features for all players.
        """
        for player in self.encoded_players.keys():
            x, y = self.pre_normalized_player_io(player)

            #append player's data to the main input and output arrays
            self.X = np.vstack([self.X, x])
            self.Y = np.vstack([self.Y, y])
        
        assert(self.X.shape[0] == self.Y.shape[0])
        self.logger.info(f'Completed preparing pre-normalized inputs and outputs: {self.X.shape[0]} games/data points')
        return self.X, self.Y

    def pre_normalized_player_io(self, player):
        """
        Processes the pre-normalized input and output for a single player.

        Parameters:
        -----------
        player : str
            The player identifier.

        Returns:
        --------
        tuple:
            - x: A NumPy array containing input features for the player's recent games.
            - y: A NumPy array containing output features for the player's recent games.
        """
        #get recent games
        game_log = self.recent_games(player)
        self.logger.debug(f'Preparing pre-normalized data for player: {player}, {len(game_log)} games')
        
        #one hot encode the player and copy it to the number of games since each game is a datapoint
        player_input = self.get_onehot_player(player)
        x = np.tile(player_input, (len(game_log), 1))

        #for each game get the expected output values for training
        player_outputs = [
            self.pre_normalized_game_output(player, game.GAMEID)
            for game in game_log.itertuples(index=False)
        ]
        y = np.vstack(player_outputs)

        assert(y.shape[0] == x.shape[0] and y.shape[0] == len(game_log))

        #print(x.shape)
        #print(y.shape)
        return x, y

    def pre_normalized_game_output(self, player, game_id):
        """
        Calculates the pre-normalized output for a single game.

        Parameters:
        -----------
        player : str
            The player identifier.
        game_id : str
            The game identifier.

        Returns:
        --------
        np.ndarray:
            A NumPy array containing the concatenated stats for the player,
            the average stats of the player's team (excluding the player),
            and the average stats of the opposing team.
        """
        #retrieve game data
        game = self.ext.get_game(game_id)
        self.logger.debug(f'Getting pre-normalized game data for: {game_id}, {player}')

        #filtering conditions
        #TODO: consider weighting player contribution to team calculation by minutes played
        is_player = game['PLAYER'] == player
        played_enough = game['MIN'] > self.significant_minutes

        player_team = game.loc[is_player, 'TEAM'].iloc[0] #find the player's team at the time of the game
        on_team = game['TEAM'] == player_team

        #filter dataframe using p (player), t (player's team wihtout the player), o (opponent team)
        p_data = game[is_player][self.select_columns]
        t_data = game[~is_player & played_enough & on_team][self.select_columns]
        o_data = game[played_enough & ~on_team][self.select_columns]

        #turn these into pd.Series, and then into numpy rows for faster concatenation
        p_stats = p_data.iloc[0].to_numpy()
        t_avg = t_data.mean(axis=0).to_numpy()
        o_avg = o_data.mean(axis=0).to_numpy()

        y = np.hstack([p_stats, t_avg, o_avg])

        return y



    #----------------------------------------------------------
    def set_batches(self):
        self.filter_players()
        self.set_pre_normalized_io()
        self.normalize_outputs()
        self.logger.info(f'DatasetBuilder preparation completed for making player embeddings, use get_batches(batch_size) to get batches.')





#db = PlayerEmbeddingDatasetBuilder()

#db.set_batches()
#db.X, db.Y = db.pre_normalized_player_io('nikola-jokic')
#pd.set_option('display.max_columns', None)
#db.normalize_outputs()

#print(db.pre_normalized_game_output('nikola-jokic', 401705050))

#print(db.recent_games('nikola-jokic'))

#db.set_all()


#db = DatasetBuilder()
#db.set_logger_level(logging.INFO)
#db.filter_players()
#db.set_pre_normalized_io()
#db.normalize_outputs()

#------------------------------------------------------------------------------------------------------------------------------



class AllPlayerEmbeddingDatasetBuilder(DatasetBuilder):
    """
    A DatasetBuilder subclass for creating player embeddings but for all players.

    Args:
        max_games (int): The maximum number of games to include in each player's match history.
    """
    def __init__(self, req_games=5, max_games=20, offset=0): 
        super().__init__()

        #values for this databuilder
        self.req_games = req_games #minimum number of games player required to have to be given an embedding
        self.max_games = max_games #maximum number of games to go into each player's match history
        self.offset = offset #amount of recent games to ignore before counting match history

        #19 values
        self.select_columns = ['MIN', '+/-', 'OREB', 'DREB', 'REB', 'AST', 'STL', 'BLK', 'TO', 'PF', 'PTS', 'FGM', 'FGA', '3PM', '3PA', 'FTM', 'FTA', '2PM', '2PA']
        self.zscore_columns = [['+/-'],
                               ['PTS']]
        self.minmax_columns = [['MIN'],
                               ['OREB', 'DREB', 'REB'],
                               ['AST'],
                               ['STL'],
                               ['BLK'],
                               ['TO'],
                               ['PF'],
                               ['FGM', 'FGA'],
                               ['3PM', '3PA'],
                               ['FTM', 'FTA'],
                               ['2PM', '2PA']]

        #list of all players, and their corresponding index
        self.players = self.ext.get_player_to_team().keys()
        self.player_to_idx = {player: idx for idx, player in enumerate(self.players)} #initially this is every possible player

        #numpy arrays of the inputs and corresponding outputs
        self.X = np.empty((0, len(self.players)))

        output_dim = ((len(self.select_columns) - 1) * 3) + 1 #everything times 3, except for minutes, which only exists for the player
        self.Y = np.empty((0, output_dim))



    #----------------------------------------------------------
    def get_valid_players(self):
        return self.player_to_idx.keys()



    #----------------------------------------------------------
    def recent_games(self, player):
        """
        Extract the most recent games from the player's match history.

        Args:
            player (str): The name or identifier of the player whose match history is being retrieved.

        Returns:
            pandas.DataFrame: A DataFrame containing the most recent games of the player, 
                            limited to `self.max_games` rows.
        """
        #extract the max_games most recent games from the player's match history
        game_log = self.ext.get_player_game_log(player)
        
        recent = game_log.iloc[self.offset:].head(self.max_games)
        return recent
    


    #----------------------------------------------------------
    def filter_players(self):
        """
        Filters players based on their average minutes played and encodes the qualified players.

        This function resets `self.player_to_idx` and creates a subset of players 
        who meet the `self.req_minutes` threshold for minimum minutes played in their 
        recent games. Players who qualify are assigned an encoding index.

        Args:
            None

        Returns:
            self.player_to_idx (dict): Dictionary containing valid players with indices for onehot encoding.
        """
        if self.max_games < self.req_games:
            self.logger.error(f'In AllPlayerEmbeddingDatasetBuilder calls, max_games must be greater than or equal to req_games.')
            raise

        #resets the player_to_idx mapping and limits it to a filtered subset based on average minutes played
        self.player_to_idx = {}

        #each player gets is checked to have an average minutes played above a certain cutoff
        idx = 0
        for player in self.players:
            game_log = self.recent_games(player)

            if (len(game_log) < self.req_games):
                self.logger.debug(f'Player ({player}) has not played enough ({len(game_log)}<{self.req_games}) games.')
            else:
                self.player_to_idx[player] = idx
                idx += 1

        #change the shape of X for efficient processing
        self.X = np.empty((0, len(self.get_valid_players())))
        self.logger.info(f'Filtered number of players from {len(self.players)} to {len(self.get_valid_players())}')

        return self.player_to_idx



    #----------------------------------------------------------
    def get_onehot_player(self, player):
        """
        Generates a one-hot encoded vector for the specified player.

        Args:
            player (str): The name or identifier of the player to be one-hot encoded.

        Returns:
            numpy.ndarray: A NumPy array of length `len(self.get_valid_players())` with 
                        a 1 at the index corresponding to the player's encoding 
                        and 0 elsewhere.
        """
        onehot = np.zeros(len(self.player_to_idx))
        idx = self.player_to_idx.get(player)
        if idx is None:
            self.logger.error(f'Player ({player}) not in valid players to onehot encode.')
            return None
        else:
            onehot[idx] = 1
            return onehot
    
    def get_player_vector(self, player):
        """
        Gets a vector embedding for the specified player.

        Args:
            player (str): The name or identifier of the player to get a vector embedding for

        Returns:
            numpy.ndarray: A NumPy array of length `embedding size`.
        """
        if self.player_embeddings is None:
            self.logger.error(f'Failed to find any player vector embeddings. Save vector embeddings with set_player_embeddings().')
            raise ValueError("Player vector embeddings have not been set.")
        
        emb = self.player_embeddings.get(player)
        if emb is None:
            self.logger.error(f'Failed to find player ({player}) in vector embeddings.')
            raise KeyError(f"Player '{player}' not found in vector embeddings.")
        
        return emb



    #----------------------------------------------------------
    #custom standard scaler that operates on the same scale for multiple columns
    def z_score_normalize_inplace(self, df, columns):
        """
        Perform Z-score normalization across multiple columns, normalizing them based on 
        the combined mean and standard deviation of all the selected columns (modifies the original dataframe).
        
        Args:
        - df (pd.DataFrame): The dataframe containing the columns to normalize.
        - columns (list): A list of column names to normalize.
        
        Returns:
        - None: The dataframe is modified in place.
        """
        # Combine all selected columns into one DataFrame to compute global mean and std
        combined_df = df[columns]
        
        # Calculate global mean and std across all the selected columns
        combined_mean = combined_df.mean().mean()  # Mean of all values in the selected columns
        combined_std = combined_df.stack().std()  # Std of all values in the selected columns
        
        # Apply the same Z-score normalization using the global mean and std
        for column in columns:
            df[column] = (df[column] - combined_mean) / combined_std

    #custom minmax scaler that operates on the same scale for multiple columns
    def min_max_normalize_inplace(self, df, columns):
        """
        Perform Min-Max normalization across multiple columns, normalizing them based on 
        the combined min and max values of all the selected columns (modifies the original dataframe).
        
        Args:
        - df (pd.DataFrame): The dataframe containing the columns to normalize.
        - columns (list): A list of column names to normalize.
        
        Returns:
        - None: The dataframe is modified in place.
        """
        # Combine all selected columns into one DataFrame to compute global min and max
        combined_df = df[columns]
        
        # Calculate global min and max across all the selected columns
        combined_min = combined_df.min().min()  # Min value across all selected columns
        combined_max = combined_df.max().max()  # Max value across all selected columns
        
        # Apply Min-Max scaling using the global min and max
        for column in columns:
            df[column] = (df[column] - combined_min) / (combined_max - combined_min)

    def normalize_outputs(self):
        """
        Normalizes the output data (`self.Y`) column-wise using appropriate scaling methods
        and returns the scaled input (`self.X`) and output (`self.Y`) arrays.

        Columns with negative values are scaled using StandardScaler (Z-score scaling),
        while columns with only non-negative values are scaled using MinMaxScaler.

        Returns:
        --------
        tuple:
            - self.X: The input data (unchanged in this method).
            - self.Y: The normalized output data.
        """
        assert not np.isnan(self.X).any(), "Input data contains NaN values before scaling."
        assert not np.isnan(self.Y).any(), "Input data contains NaN values before scaling."

        column_names = self.select_columns + (2 * self.select_columns[1:])
        df = pd.DataFrame(self.Y, columns=column_names)

        #print(df['PTS'])
        
        #some columns need to be put on standard scaling, then squished into 0-1 range with minmax. PTS and +/-
        for group_col in self.zscore_columns:
            self.z_score_normalize_inplace(df, group_col)
            self.min_max_normalize_inplace(df, group_col)

            self.logger.debug(f'Zscore normalized, then Minmax normalized columns: {group_col}')

        #all other stats are minmaxed, but some of them in groups. OREB, DREB, and REB are all minmaxed together on the same scale.
        for group_col in self.minmax_columns:
            self.min_max_normalize_inplace(df, group_col)

            self.logger.debug(f'Minmax normalized columns: {group_col}')

        #print(df['PTS'])

        Y_scaled = df.to_numpy()
        assert(self.Y.shape == Y_scaled.shape)
        self.logger.info(f'Completed normalization for outputs.')

        self.Y = Y_scaled
        return self.X, self.Y



    #----------------------------------------------------------
    def set_pre_normalized_io(self):
        """
        Generates pre-normalized input (X) and output (Y) datasets for all players.

        Iterates through each player in `self.player_to_idx`, collects their game logs, 
        and processes their inputs and outputs for training.

        Returns:
        --------
        tuple:
            - self.X: A NumPy array containing input features for all players.
            - self.Y: A NumPy array containing output features for all players.
        """
        for player in self.get_valid_players():
            x, y = self.pre_normalized_player_io(player)

            #append player's data to the main input and output arrays
            self.X = np.vstack([self.X, x])
            self.Y = np.vstack([self.Y, y])
        
        assert(self.X.shape[0] == self.Y.shape[0])
        self.logger.info(f'Completed preparing pre-normalized inputs and outputs: {self.X.shape[0]} games/data points')
        return self.X, self.Y

    def pre_normalized_player_io(self, player):
        """
        Processes the pre-normalized input and output for a single player.

        Parameters:
        -----------
        player : str
            The player identifier.

        Returns:
        --------
        tuple:
            - x: A NumPy array containing input features for the player's recent games.
            - y: A NumPy array containing output features for the player's recent games.
        """
        #get recent games
        game_log = self.recent_games(player)
        self.logger.debug(f'Preparing pre-normalized data for player: {player}, {len(game_log)} games')
        
        #for each game get the expected output values for training
        player_inputs = []
        player_outputs = []
        for game in game_log.itertuples(index=False):

            input = self.get_onehot_player(player)
            output = self.pre_normalized_game_output(player, game.GAMEID)

            #check that it returns something valid
            if output is not None:
                player_outputs.append(output)
                player_inputs.append(input)

        x = np.vstack(player_inputs)
        y = np.vstack(player_outputs)

        assert(y.shape[0] == x.shape[0])
        #this next part does not work consistently due to edge cases: 
        # and y.shape[0] == len(game_log))

        #print(x.shape)
        #print(y.shape)
        return x, y

    def pre_normalized_game_output(self, player, game_id):
        """
        Calculates the pre-normalized output for a single game.

        Parameters:
        -----------
        player : str
            The player identifier.
        game_id : str b  
            The game identifier.

        Returns:
        --------
        np.ndarray:
            A NumPy array containing the concatenated stats for the player,
            the average stats of the player's team (excluding the player),
            and the average stats of the opposing team.
        """
        #retrieve game data
        game = self.ext.get_game(game_id)
        self.logger.debug(f'Getting pre-normalized game data for: {game_id}, {player}')

        #filtering conditions
        is_player = game['PLAYER'] == player

        if not is_player.any(): #break out when player didnt play in this game
            self.logger.debug(f'Edge case! {player} played 0 minutes in game {game_id}, despite a game log entry.')
            return None

        player_team = game.loc[is_player, 'TEAM'].iloc[0] #find the player's team at the time the game was played (consider trades)
        on_team = game['TEAM'] == player_team

        #filter dataframe using p (player), t (player's team wihtout the player), o (opponent team)
        p_df = game[is_player][self.select_columns]
        t_df = game[~is_player & on_team][self.select_columns]
        o_df = game[~on_team][self.select_columns]

        #turn these into pd.Series, and then into numpy rows for faster concatenation
        p_stats = p_df.iloc[0].to_numpy()
        t_avg = self.weighted_average_team(t_df).to_numpy()
        o_avg = self.weighted_average_team(o_df).to_numpy()

        y = np.hstack([p_stats, t_avg, o_avg])

        return y

    def weighted_average_team(self, df):
        """
        Given a dataframe containing a team's data of self.select_columns, compute the weighted average of the team
        as a whole, using minutes as the weighting stat. See timing test for impact_min for implementation details

        Parameters:
        -----------
        df : pd.DataFrame
            Contains all selectable stat columns to average (except for MIN)

        Returns:
        --------
        pd.Series:
            A pandas Series containing all stats except for minutes played, since it averages team stats weighting 
            with minutes played.
        """
        mins = df['MIN']
        df = df.drop(columns=['MIN'])

        df = df.multiply(mins, axis=0).sum(axis=0) / mins.sum()
        return df



    #----------------------------------------------------------
    def set_batches(self):
        self.filter_players()
        self.set_pre_normalized_io()
        self.normalize_outputs()
        self.logger.info(f'DatasetBuilder preparation completed for making all player embeddings.') 

        print(self.Y)


#db = AllPlayerEmbeddingDatasetBuilder()


