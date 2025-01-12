from DatasetBuilders.DatasetBuilder import DatasetBuilder
import pandas as pd 
import numpy as np
from itertools import combinations
import logging
import json

class OverUnderDatasetBuilder(DatasetBuilder):
    """
    A class for building a dataset for the over/under prediction task in basketball.

    This class inherits from `DatasetBuilder` and is designed to create datasets using pre-computed player embeddings 
    for predicting whether a player will score over or under a certain threshold based on a specified metric.

    Attributes:
    -----------
    max_games : int
        The maximum number of previous games to use in the dataset for each team (default is 20).
    
    embedding_size : int
        The size of the embedding vector for each player (default is 30).
    
    select_columns : list
        A list of the statistical columns used in the dataset, typically ['REB', 'AST', 'STL', 'BLK', 'PTS'].
    
    metric : str or None
        The metric used to determine the over/under outcome (e.g., 'PTS', 'AST'). Must be set before building the dataset.
    
    threshold : float or None
        The threshold value that determines whether the outcome is over or under for the given metric.
    
    player_embeddings : numpy.ndarray or None
        A 2D array storing the embeddings for each player, indexed by player ID.
    
    usable_games : list or None
        A list of the games that can be used to build the dataset.
    
    X : numpy.ndarray
        The input features for the dataset. Shape is (num_samples, (embedding_size * 3) + 2), where 3 represents 
        the number of embedded features (player, player's team, opponent team), and 2 for home/away indicators. 
        These are ordered as [player, home/away, player team, opponent team]
    
    Y : numpy.ndarray
        The target labels for the dataset. Shape is (num_samples, 2), representing the percentage for [under, over].
    """
    def __init__(self, max_games=20, embedding_size=30): 
        super().__init__()

        #values for this databuilder
        self.embedding_size = embedding_size #embedding size for a single player

        self.select_columns = ['REB', 'AST', 'STL', 'BLK', 'PTS']

        #which metric to measure over/under on, and the threshold for over/under
        self.metric = None
        self.threshold = None

        #embeddings for each player
        self.player_embeddings = None

        #games to build the dataset out of
        self.usable_games = None

        #numpy arrays of the inputs and corresponding outputs
        self.X = np.empty((0, ((self.embedding_size * 3) + 2))) #+2 for home/away neurons
        self.Y = np.empty((0, 2)) #percentages for [under, over]

        #for building a specific problem
        self.player_to_team = self.ext.get_player_to_team()
        self.team_names = self.ext.get_team_names().keys()
        



    #----------------------------------------------------------
    #sets the metric and threshold, ex: PTS over/under 25.5
    def set_metric_threshold(self, metric, threshold):
        """
        Sets the metric and threshold for classification based on the provided inputs.

        Args:
        --------
        metric : str
            The name of the metric to be used for classification. This metric should be one 
            of the columns selected in `self.select_columns`.
            
        threshold : float
            The threshold value that will be used to classify instances as either the positive 
            or negative class based on the selected metric. The classification will depend on 
            whether the metric value exceeds the threshold.
        """
        assert(metric in self.select_columns)
        self.metric = metric
        self.threshold = threshold
        self.logger.info(f'Set the metric as {self.metric} and threshold at {self.threshold}.')

    def get_expected_output(self, value):
        """
        Determines the expected output (one-hot encoded vector) based on the input value
        and the specified threshold.

        Args:
        --------
        value : float
            The input value used to determine whether the prediction should be classified 
            as 'over' or 'under'.

        Returns:
        --------
        numpy.ndarray
            A one-hot encoded vector where:
            - [0, 1] represents the positive class ("over") if the input value exceeds 
            the threshold, this is considered the positive class at index 1 when argmaxed.
            - [1, 0] represents the negative class ("under") if the input value is 
            below or equal to the threshold.

        Notes:
        ------
        The threshold is a predefined value that distinguishes between the positive and 
        negative class. In this case, if the input value exceeds the threshold, the 
        class is considered "over" (positive class, index 1). Otherwise, it is considered 
        "under" (negative class, index 0).
        """
        if value > self.threshold:
            return np.array([0, 1])
        else:
            return np.array([1, 0])



    #----------------------------------------------------------
    def set_embeddings(self, embeddings=None, embeddings_file=None):
        """
        Saves player vector embedding dictionary to this object instance.

        Args:
            embeddings (dict, optional): A dictionary of player embeddings to load.
            embedding_file (str, optional): Path to a CSV file containing player embeddings.

        Returns:
            None
        """
        if self.player_embeddings is not None:
            self.logger.info(f'Overwriting pre-existing player vector embeddings.')
        
        #load in the raw embeddings
        if embeddings is not None:
            self.player_embeddings = embeddings
            self.logger.debug(f'Reading in player vector embeddings from raw data.')

        #load in embeddings from a file
        elif embeddings_file is not None:
            df = pd.read_csv(embeddings_file)
            df = df[['PLAYER', 'EMBEDDING']] #keep only these columns
            
            #convert the 'EMBEDDING' column to actual NumPy arrays
            df['NUMPY EMBEDDING'] = df['EMBEDDING'].apply(
                lambda x: np.fromstring(x.strip('[]'), sep=' ')
            )

            #ensure correct read of numpy arrays by checking embeddings having the same length
            embedding_lengths = df['NUMPY EMBEDDING'].apply(len).unique()
            if len(embedding_lengths) == 1:

                #reset embedding size, and corresponding expected input dimensions
                self.embedding_size = embedding_lengths[0]
                self.X = np.empty((0, ((self.embedding_size * 3) + 2)))
                self.logger.info(f'Embedding size of each player is set to {self.embedding_size}.')

            else:
                self.logger.error(f'Failure in each player having same embedding size.')
                raise ValueError(f'Embeddings have inconsistent lengths.')
    
            #put it into a dictionary
            self.player_embeddings = df.groupby('PLAYER')['NUMPY EMBEDDING'].first().to_dict()
            self.logger.debug(f'Reading in player vector embeddings from embedding file.')
        
        #nothing to read the embeddings from
        else:
            self.logger.error(f'Failed to save new player vector embeddings.')
            raise

        return None

    def get_player_embedding(self, player):
        """
        Retrieves the vector embedding for the specified player.

        Args:
            player (str): The name or identifier of the player to retrieve the embedding for.

        Returns:
            numpy.ndarray: A NumPy array representing the player's vector embedding.    
        """
        emb = self.player_embeddings.get(player)
        
        #check for missing player embedding
        if emb is None:
            self.logger.error(f'Failure to find ({player}) in embeddings.')
            raise
        
        return emb
    
    def get_loc_embedding(self, loc):

        if loc == 'vs':
            return np.array([1, 0])
        #away
        elif loc == '@':
            return np.array([0, 1])

        else:
            self.logger.error(f'Failure to get location embedding for ({loc}), expected @ or vs')
            raise
        


    #----------------------------------------------------------
    def filter_game_ids(self, max_games=20, offset=0):
        """
        Filters games based on their recency by team schedule and collects unique game ids.

        This function resets `self.usable_games` and creates a subset of game ids 
        that meet the `max_games` threshold number of games for recency in team's schedules.

        Args:
            offset (int): How many games far back to ignore per team. For example, if max_games=20 and offset=2,
            the games selected would ignore the 2 most recent games up until the 22nd most recent game. This accounts
            for cutoffs in data, so if there were only 17 games in each team's schedule having been played, it would select games 1-15.

        Returns:
            self.usable_games (set): Set that has all unique game ids.
        """
        #collect the unique game_ids from each team
        game_ids_set = set()

        #go through each team's schedule and pick the max_games most recent games
        team_abbrs = self.ext.get_team_names().keys()
        for team_abbr in team_abbrs:
            sched = self.ext.get_team_sched_played(team_abbr)
            #print(sched)
            all_game_ids = sched['GAMEID'].tolist()
            
            #indices for slicing the games to take
            oldest_index = max(len(all_game_ids) - max_games - offset, 0)
            newest_index = len(all_game_ids) - offset

            if newest_index < 2:
                self.logger.error(f'Failed to find enough game data from team schedule: {team_abbr}')
                raise ValueError(f'Not enough data.')
            
            game_ids = all_game_ids[oldest_index : newest_index]

            #print(oldest_index, newest_index)
            #print(game_ids[0], game_ids[-1])

            #add to set
            game_ids_set.update(str(game_id) for game_id in game_ids)

        self.usable_games = game_ids_set
            
        return self.usable_games



    #----------------------------------------------------------
    def set_io(self):
        """
        Generates input (X) and output (Y) datasets for all games and valid players. No need for normalization, 
        since inputs are already normalized between 0 and 1, and outputs are decided.

        Iterates through each game in `self.usable_games`, collects the data for each player in the game, 
        and processes their inputs and outputs for training.

        Returns:
        --------
        tuple:
            - self.X: A NumPy array containing input features for all players.
            - self.Y: A NumPy array containing output features for all players.
        """
        for game in self.usable_games:
            x, y = self.game_io(game)

            #append game's data to the main input and output arrays
            self.X = np.vstack([self.X, x])
            self.Y = np.vstack([self.Y, y])

        assert(self.X.shape[0] == self.Y.shape[0])
        self.logger.info(f'Completed preparing inputs and outputs: {self.X.shape[0]} data points from {len(self.usable_games)} games.')

        under, over = np.sum(self.Y, axis=0)
        self.logger.info(f'Under samples: {int(under)}, Over samples: {int(over)}. Sample ratio: {np.round(under/over, 2)}')
        return self.X, self.Y
    
    def game_io(self, game_id):
        """
        Processes the inputs and outputs for each game, which consists of combining a player embedding,
        the home/away status of the player, the player's teams approximate embedding, and the opposite 
        team's approximate embedding. The outputs are the selected stats, which should be normalized later.

        Parameters:
        -----------
        game_id : str
            The game identifier.

        Returns:
        --------
        tuple:
            - x: A NumPy array containing input features for each player in the game.
            - y: A NumPy array containing output features for each player in the game.
        """
        if self.metric is None or self.threshold is None:
            self.logger.error(f'Failed to find valid metric and threshold.')
            raise

        #inputs and outputs extrapolated from this game
        x, y = [], []

        #retrieve game data
        game = self.ext.get_game(game_id)
        self.logger.debug(f'Getting game data for: {game_id}')

        #filter rows where the player is a key in the embedded players dictionary
        filtered_rows = game[game['PLAYER'].map(self.player_embeddings.__contains__)].copy() #avoid SettingWithCopyWarning from pandas
        filtered_rows['EMBEDDING'] = filtered_rows['PLAYER'].apply(lambda x: self.get_player_embedding(x))

        #print(filtered_rows)

        #conditional for splitting teams
        is_home = filtered_rows['LOC'] == 'vs'
        home_rows = filtered_rows[is_home]
        away_rows = filtered_rows[~is_home]

        #fewest dictionary accesses, bundles together all players from each team. to undo this we will reverse each player at a time
        #TODO: refactor bundling method, average seems to work slightly better
        #home_vec = np.prod(home_rows['EMBEDDING'], axis=0)
        #away_vec = np.prod(away_rows['EMBEDDING'], axis=0)
        home_vec = np.sum(home_rows['EMBEDDING'], axis=0)
        away_vec = np.sum(away_rows['EMBEDDING'], axis=0)

        #home team
        for _, row in home_rows.iterrows():
            player_emb = row['EMBEDDING']
            loc_emb = self.get_loc_embedding(row['LOC'])

            #TODO: refactor bundling method
            #team_vec = np.divide(home_vec, player_emb, where=player_emb != 0) #safe division
            #opps_vec = away_vec
            team_vec = (home_vec - player_emb) / (len(home_rows) - 1)
            opps_vec = away_vec / len(home_rows) #this should be swapped for the other team

            #collect the input info, which should have player embedding, home/away embedding, player's team, and opp team
            input_data = np.hstack([player_emb, loc_emb, team_vec, opps_vec])
            x.append(input_data)

            #expected stats for the player
            value = row[self.metric]
            output_data = self.get_expected_output(value)
            y.append(output_data)

        #away team
        for _, row in away_rows.iterrows():
            player_emb = row['EMBEDDING']
            loc_emb = self.get_loc_embedding(row['LOC'])

            #TODO: bundling method
            #team_vec = np.divide(away_vec, player_emb, where=player_emb != 0) #safe division
            #opps_vec = home_vec
            team_vec = (away_vec - player_emb) / (len(away_rows) - 1)
            opps_vec = home_vec / len(home_rows)

            #collect the input info, which should have player embedding, home/away embedding, player's team, and opp team
            input_data = np.hstack([player_emb, loc_emb, team_vec, opps_vec])
            x.append(input_data)

            #expected stat for the player
            value = row[self.metric]
            output_data = self.get_expected_output(value)
            y.append(output_data)

        x = np.vstack(x)
        y = np.vstack(y).astype(float)

        self.logger.debug(f'Processed ({len(filtered_rows)}) players for game ({game_id}).')

        #print(x.shape)
        #print(y.shape)

        #print(x.dtype)
        #print(y.dtype)
        return x, y

    def build_input(self, player, loc, opp):

        #player embedding raises exception if no valid embedding
        player_emb = self.get_player_embedding(player)

        loc_emb = self.get_loc_embedding(loc)

        #find the player's current team
        team = self.player_to_team.get(player)
        if team is None:
            self.logger.error(f'Failed to find ({player}) in player-to-team mapping.')
            raise

        if opp not in self.team_names:
            self.logger.error(f'Failed to find ({opp}) in valid team names. Please try: {self.team_names}')
            raise

        #process player team
        df = self.ext.get_team_stats(team)
        assert(player in df['PLAYER'].tolist())

        filtered_rows = df[df['PLAYER'].map(self.player_embeddings.__contains__)].copy() #avoid SettingWithCopyWarning from pandas
        filtered_rows['EMBEDDING'] = filtered_rows['PLAYER'].apply(lambda x: self.get_player_embedding(x))
        team_vec = np.sum(filtered_rows['EMBEDDING'], axis=0)
        team_vec = (team_vec - player_emb) / (len(filtered_rows) - 1)

        #process opp team
        df = self.ext.get_team_stats(opp)
        assert(player not in df['PLAYER'].tolist())

        filtered_rows = df[df['PLAYER'].map(self.player_embeddings.__contains__)].copy() #avoid SettingWithCopyWarning from pandas
        filtered_rows['EMBEDDING'] = filtered_rows['PLAYER'].apply(lambda x: self.get_player_embedding(x))
        opp_vec = np.sum(filtered_rows['EMBEDDING'], axis=0)
        opp_vec /= len(filtered_rows)

        input = np.hstack([player_emb, loc_emb, team_vec, opp_vec])

        self.logger.info(f'Built input: ({player}, {team} {loc} {opp})')
        #print(input.shape)

        return input
        


    #----------------------------------------------------------
    def set_batches(self, embeddings=None, embeddings_file=None, metric='PTS', threshold=16.5, max_games=20, offset=0):
        self.set_embeddings(embeddings=embeddings, embeddings_file=embeddings_file)
        self.set_metric_threshold(metric=metric, threshold=threshold)
        self.filter_game_ids(max_games=max_games, offset=offset)
        self.set_io()
        self.logger.info(f'DatasetBuilder preparation completed for making over/under data, use get_batches(batch_size) to get batches.')




#db = OverUnderDatasetBuilder(max_games=20)
#db.set_batches(embeddings_file='emb_player_10_1_20_15_100.csv', metric='PTS', threshold=19.5, max_games=20, offset=1)
#print(db.Y)
#print(np.sum(db.Y, axis=0))

#db.set_embeddings(embeddings_file='emb_player_30units_20min_20games.csv')
#db.filter_game_ids(offset=1)

#db.build_input('jaylen-wells', '@', 'sac')


#db.set_batches()

#print(db.X.shape)
#print(db.Y.shape)

#print(db.get_player_embedding('alperen-sengun'))

#db.pre_normalized_game_io('401737912')
