from dataset_builders.base_dataset_builder import BaseDatasetBuilder
import pandas as pd 
import numpy as np


class StatDatasetBuilder(BaseDatasetBuilder):
    """
    A class for building a dataset for the stat prediction task in basketball.

    This class inherits from `DatasetBuilder` and is designed to create datasets using pre-computed player embeddings 
    for predicting whether a player will score over or under a certain threshold based on a specified metric.

    Attributes:
    -----------
    player_embedding : dict
        Mapping dictionary of player embeddings.    

    select_columns : list
        A list of the statistical columns used in the dataset, typically ['REB', 'AST', 'STL', 'BLK', 'PTS'].
    
    metric : str or None
        The metric used to determine the over/under outcome (e.g., 'PTS', 'AST'). Must be set before building the dataset.

    scaler : str
        The scaling method for the output.
    
    max_games : int
        The maximum number of previous games to use in the dataset for each team (default is 20).
    
    usable_games : list or None
        A list of the games that can be used to build the dataset.
    
    X : numpy.ndarray
        The input features for the dataset. Shape is (num_samples, (embedding_size * 3) + 2), where 3 represents 
        the number of embedded features (player, player's team, opponent team), and 2 for home/away indicators. 
        These are ordered as [player, home/away, player team, opponent team]
    
    Y : numpy.ndarray
        The target labels for the dataset. Shape is (num_samples, 2), representing the percentage for [under, over].
    """
    def __init__(self, player_embeddings, metric='PTS', scaler='zscore', max_games=20, offset=0): 
        super().__init__()

        #embeddings for each player
        self.player_embeddings = player_embeddings

        #selectable columns from the data to make a dataset, there are others like PRA which is handled elsewhere
        self.select_columns = ['REB', 'AST', 'STL', 'BLK', 'PTS']

        #which metric to measure stat on, these will get changed by set_metric_scaler
        self.metric = metric
        self.scaler = scaler

        #filtering games to use in the dataset
        self.max_games = max_games
        self.offset = offset
        self.usable_games = None

        #numpy arrays of the inputs and corresponding outputs
        self.X = np.empty((0, 0)) #3* player embedding size +2 for home/away neurons
        self.Y = np.empty((0, 1)) #percentages for [under, over]

        #for building a specific problem
        self.player_to_team = self.ext.get_player_to_team()
        self.team_names = self.ext.get_team_names().keys()
        



    #----------------------------------------------------------
    #sets the metric, ex: PTS, PRA
    def set_metric_scaler(self):
        """
        Sets the metric for estimation based on the provided inputs. Updates the metric and scaler to be used.
        """
        special_columns = {
            'PRA': ['PTS', 'REB', 'AST'],
            'AR' : ['AST', 'REB'],
            'RA' : ['AST', 'REB'],
        }

        if self.metric in special_columns.keys():
            self.metric = special_columns[self.metric]

        else:
            assert(self.metric in self.select_columns)

        #setting the scaler to use
        valid_scales = {
            'minmax': self.minmax_scaler,
            'zscore': self.standard_scaler,
            'standard': self.standard_scaler
        }

        assert(self.scaler in valid_scales.keys())
        if self.scaler in valid_scales.keys():
            self.scaler = valid_scales[self.scaler]

        self.logger.info(f'Set the metric as {self.metric}, and scaler to {self.scaler}')

    def get_expected_output(self, value):
        """
        Determines the expected output PTS, AST, or PRA, etc.

        Args:
        --------
        value : df.Series
        """
        y = np.sum(value)
        return y

    def normalize_outputs(self):
        """
        Normalizes the output data using the specified scaling method.
        """
        self.Y = self.scaler.fit_transform(self.Y)

        self.logger.info(f'Normalized outputs using {self.scaler}.')

    def denormalize(self, value):
        """
        Denormalizes a value or array of values based on the previously used scaling method.
        
        Parameters:
            value (float or np.ndarray): The value(s) to denormalize. Can be a scalar or an array.
        
        Returns:
            np.ndarray: The denormalized value(s) in the original scale.
        """
        if np.isscalar(value):  # If it's a single value, reshape it
            value = np.array([[value]])
        return self.scaler.inverse_transform(value).flatten()
        


    #----------------------------------------------------------
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
    def filter_game_ids(self):
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
            oldest_index = max(len(all_game_ids) - self.max_games - self.offset, 0)
            newest_index = len(all_game_ids) - self.offset

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
        all_x = []
        all_y = []
        for game in self.usable_games:
            x, y = self.game_io(game)

            #append game's data to the main input and output arrays
            all_x.append(x)
            all_y.append(y)
            #self.X = np.vstack([self.X, x])
            #self.Y = np.vstack([self.Y, y])

        self.X = np.vstack(all_x)
        self.Y = np.vstack(all_y)

        assert(self.X.shape[0] == self.Y.shape[0])
        self.logger.info(f'Completed preparing inputs and outputs: {self.X.shape[0]} data points from {len(self.usable_games)} games.')
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
        if self.metric is None:
            self.logger.error(f'Failed to find valid metric.')
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
    def set_batches(self):
        self.set_metric_scaler()
        self.filter_game_ids()
        self.set_io()
        self.normalize_outputs()
        self.logger.info(f'DatasetBuilder preparation completed for making stats data, use get_batches(batch_size) to get batches.')




#db = StatDatasetBuilder()
#db.set_batches(embeddings_file='emb_player_10_1_20_15_100.csv', metric='PTS', max_games=20, offset=1)
#print(db.X.shape)
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








class StatAttentionDatasetBuilder(BaseDatasetBuilder):
    """
    A class for building a dataset for the stat prediction task in basketball.

    This class inherits from `DatasetBuilder` and is designed to create datasets using pre-computed player embeddings 
    for predicting whether a player will score over or under a certain threshold based on a specified metric.

    Attributes:
    -----------
    player_embedding : dict
        Mapping dictionary of player embeddings.    

    select_columns : list
        A list of the statistical columns used in the dataset, typically ['REB', 'AST', 'STL', 'BLK', 'PTS'].
    
    metric : str or None
        The metric used to determine the over/under outcome (e.g., 'PTS', 'AST'). Must be set before building the dataset.

    scaler : str
        The scaling method for the output.
    
    max_games : int
        The maximum number of previous games to use in the dataset for each team (default is 20).
    
    usable_games : list or None
        A list of the games that can be used to build the dataset.
    
    X : numpy.ndarray
        The input features for the dataset. Shape is (num_samples, (embedding_size * 3) + 2), where 3 represents 
        the number of embedded features (player, player's team, opponent team), and 2 for home/away indicators. 
        These are ordered as [player, home/away, player team, opponent team]
    
    Y : numpy.ndarray
        The target labels for the dataset. Shape is (num_samples, 2), representing the percentage for [under, over].
    """
    def __init__(self, player_embeddings, mppt=8, metric='PTS', scaler='zscore', max_games=20, offset=0, out_players=[]): 
        super().__init__()

        #embeddings for each player
        self.player_embeddings = player_embeddings

        #maximum number of players from each team to consider
        self.mppt = mppt

        #selectable columns from the data to make a dataset, there are others like PRA which is handled elsewhere
        self.select_columns = ['REB', 'AST', 'STL', 'BLK', 'PTS']

        #which metric to measure stat on, these will get changed by set_metric_scaler
        self.metric = metric
        self.scaler = scaler

        #filtering games to use in the dataset
        self.max_games = max_games
        self.offset = offset
        self.usable_games = None

        #numpy arrays of the inputs and corresponding outputs
        self.X = np.empty((0, 2 * mppt, 64)) #2* max players from each team, Games,Players,Embeddingsize
        self.Y = np.empty((0, 2 * mppt, 1)) #stat for each player

        #for building a specific problem
        self.player_to_team = self.ext.get_player_to_team()
        self.team_names = self.ext.get_team_names().keys()

        self.out_players = None
        



    #----------------------------------------------------------
    #sets the metric, ex: PTS, PRA
    def set_metric_scaler(self):
        """
        Sets the metric for estimation based on the provided inputs. Updates the metric and scaler to be used.
        """
        special_columns = {
            'PRA': ['PTS', 'REB', 'AST'],
            'AR' : ['AST', 'REB'],
            'RA' : ['AST', 'REB'],
            'PR' : ['PTS', 'REB'],
            'RP' : ['PTS', 'REB']
        }

        if self.metric in special_columns.keys():
            self.metric = special_columns[self.metric]

        else:
            assert(self.metric in self.select_columns)

        #setting the scaler to use
        valid_scales = {
            'minmax': self.minmax_scaler,
            'zscore': self.standard_scaler,
            'standard': self.standard_scaler
        }

        assert(self.scaler in valid_scales.keys())
        if self.scaler in valid_scales.keys():
            self.scaler = valid_scales[self.scaler]

        self.logger.info(f'Set the metric as {self.metric}, and scaler to {self.scaler}')

    def get_expected_output(self, value):
        """
        Determines the expected output PTS, AST, or PRA, etc.

        Args:
        --------
        value : df.Series
        """
        y = np.sum(value)
        return y

    def normalize_outputs(self):
        """
        Normalizes the output data using the specified scaling method.
        """
        flat_Y = self.Y.reshape(-1, 1)

        norm_Y = self.scaler.fit_transform(flat_Y)

        self.Y = norm_Y.reshape(self.Y.shape)

        self.logger.info(f'Normalized outputs using {self.scaler}.')

    def denormalize(self, value):
        """
        Denormalizes a value or array of values based on the previously used scaling method.
        
        Parameters:
            value (float or np.ndarray): The value(s) to denormalize. Can be a scalar or an array.
        
        Returns:
            np.ndarray: The denormalized value(s) in the original scale.
        """
        # Ensure `value` is at least an array
        value = np.asarray(value)
        
        # Save original shape
        original_shape = value.shape
        
        # Reshape to 2D (required by scaler)
        value_reshaped = value.reshape(-1, original_shape[-1]) if value.ndim > 1 else value.reshape(-1, 1)
        
        # Apply inverse transformation
        denormalized = self.scaler.inverse_transform(value_reshaped)
        
        # Reshape back to the original shape
        return denormalized.flatten()
        


    #----------------------------------------------------------
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
    def filter_game_ids(self):
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
            oldest_index = max(len(all_game_ids) - self.max_games - self.offset, 0)
            newest_index = len(all_game_ids) - self.offset

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
        all_x = []
        all_y = []
        for game in self.usable_games:
            x, y = self.game_io(game)

            #append game's data to the main input and output arrays
            all_x.append(x)
            all_y.append(y)
            #self.X = np.vstack([self.X, x])
            #self.Y = np.vstack([self.Y, y])

        self.X = np.stack(all_x, axis=0)
        self.Y = np.stack(all_y, axis=0)

        assert(self.X.shape[0] == self.Y.shape[0])
        self.logger.info(f'Completed preparing inputs and outputs: {self.X.shape[0]} data points from {len(self.usable_games)} games.')
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
        if self.metric is None:
            self.logger.error(f'Failed to find valid metric.')
            raise

        #inputs and outputs extrapolated from this game
        x, y = [], []

        #retrieve game data
        game = self.ext.get_game(game_id)
        self.logger.debug(f'Getting game data for: {game_id}')

        grouped = game.groupby('LOC')

        home_df = grouped.get_group('vs')
        away_df = grouped.get_group('@')

        home_df = self.prune_team_df(home_df, self.get_loc_embedding('vs'))
        away_df = self.prune_team_df(away_df, self.get_loc_embedding('@'))

        combined = pd.concat([home_df, away_df])
        #print(combined)

        #add in location
        x = []
        y = []
        for _, row in combined.iterrows():
            
            x.append(row['EMBEDDING'])

            value = row[self.metric]
            out = self.get_expected_output(value)
            y.append(out)

        x = np.vstack(x)
        y = np.vstack(y).astype(float)

        #padding if not enough players
        player_count, embedding_size = x.shape
        desired_player_count = 2 * self.mppt

        if player_count < desired_player_count:
            pad_x = np.zeros((desired_player_count, embedding_size))
            pad_x[:player_count, :] = x
            x = pad_x

            pad_y = np.zeros((desired_player_count,1))
            pad_y[:player_count] = y
            y = pad_y

        self.logger.debug(f'Processed ({len(combined)}) players for game ({game_id}).')

        #print(x.shape)
        #print(y.shape)

        #print(x.dtype)
        #print(y.dtype)
        return x, y



    def prune_team_df(self, df, loc_emb, out_players=False):
        filtered_rows = df[df['PLAYER'].map(self.player_embeddings.__contains__)].copy() #avoid SettingWithCopyWarning from pandas

        #combines pre computed embeddings with home or away status
        filtered_rows['EMBEDDING'] = filtered_rows['PLAYER'].apply(lambda x: np.hstack((self.get_player_embedding(x), loc_emb)))

        #if we want to remove players that are injured based on self.out_players
        if out_players:
            filtered_rows = filtered_rows[~filtered_rows['PLAYER'].isin(self.out_players)]
            
        pruned_df = filtered_rows.nlargest(self.mppt, 'MIN')
        #print(pruned_df['PLAYER'].tolist())
        return pruned_df




    def set_out_players(self, out_players):
        self.out_players = out_players

    def build_input(self, home_team, away_team, home_incl=[], away_incl=[], home_excl=[], away_excl=[]):

        #player embedding raises exception if no valid embedding
        home_df = self.ext.get_team_stats(home_team)
        away_df = self.ext.get_team_stats(away_team)

        #ensure they are current players and not traded players
        home_current_players = self.ext.extract_team_roster(home_team)['PLAYER'].tolist()
        away_current_players = self.ext.extract_team_roster(away_team)['PLAYER'].tolist()

        home_df = home_df[home_df['PLAYER'].isin(home_current_players)]
        away_df = away_df[away_df['PLAYER'].isin(away_current_players)]

        #remove excluded players (and included players, which prevents duplicates)
        home_df = home_df[~home_df['PLAYER'].isin(home_excl + home_incl)]
        away_df = away_df[~away_df['PLAYER'].isin(away_excl + away_incl)]

        #add included players and make maximum minutes to guarantee inclusion
        default_values = {col: 0 for col in home_df.columns if col != 'PLAYER'}
        default_values['MIN'] = np.finfo(np.float64).max  # Set 'MIN' to max float

        home_new_rows = pd.DataFrame([{**default_values, 'PLAYER': player} for player in home_incl])
        away_new_rows = pd.DataFrame([{**default_values, 'PLAYER': player} for player in away_incl])

        home_df = pd.concat([home_df, home_new_rows], ignore_index=True)
        #print(home_df)
        away_df = pd.concat([away_df, away_new_rows], ignore_index=True)

        #prune modified df
        home_df = self.prune_team_df(home_df, self.get_loc_embedding('vs'), out_players=True)
        away_df = self.prune_team_df(away_df, self.get_loc_embedding('@'), out_players=True)

        combined = pd.concat([home_df, away_df])
        #print(combined)

        #add in location
        players = combined['PLAYER'].values
        teams = ([home_team] * len(home_df)) + ([away_team] * len(away_df))
        x = np.vstack(combined['EMBEDDING'].values)

        #padding if not enough players
        player_count, embedding_size = x.shape
        desired_player_count = 2 * self.mppt

        if player_count < desired_player_count:
            pad_x = np.zeros((desired_player_count, embedding_size))
            pad_x[:player_count, :] = x
            x = pad_x
        return x, players, teams
                


    #----------------------------------------------------------
    def set_batches(self):
        self.set_metric_scaler()
        self.filter_game_ids()
        self.set_io()
        self.normalize_outputs()
        self.logger.info(f'DatasetBuilder preparation completed for making stats data, use get_batches(batch_size) to get batches.')


#db.build_input('jaylen-wells', '@', 'sac')


#db.set_batches()

#print(db.X.shape)
#print(db.Y.shape)

#print(db.get_player_embedding('alperen-sengun'))

#db.pre_normalized_game_io('401737912')
