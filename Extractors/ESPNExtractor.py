import re
import requests
import json
import itertools
import pandas as pd
import numpy as np
import logging
import os
import time
from pathlib import Path
from Extractors.Extractor import Extractor #call this from the root directory


class ESPNNBAExtractor(Extractor):
    def __init__(self):
        super().__init__()
        self.base_url = 'https://www.espn.com/nba'
        self.root_dir = Path(__file__).resolve().parent.parent
        self.data_dir = self.root_dir / 'espn_data'
        
        self.teams_dir = self.data_dir / 'teams'
        self.games_dir = self.data_dir / 'games'

        #important file paths to files that should contain consolidated information
        self.team_names_file = self.teams_dir / 'team_names.json'
        self.team_names = None

        self.player_to_team_file = self.teams_dir / 'player_to_team.json'
        self.player_to_team = None
        
        self.injury_file = self.teams_dir / 'injuries.csv'
        self.injury = None

        self.game_ids_file = self.games_dir / 'game_ids.json'
        self.game_ids = None

        #logging
        self.logger = logging.getLogger(self.__class__.__name__)
        self.logger.setLevel(logging.INFO)
        handler = logging.StreamHandler()
        formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
        handler.setFormatter(formatter)
        self.logger.addHandler(handler)
        self.logger.propagate = False  #prevent logging from propagating to the root logger



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



    #----------------------------------------------------------
    #extract team_names from webpage
    def extract_team_names(self):
        """
        Extracts NBA team names and their abbreviations from the team stats page.

        Returns:
            dict: A dictionary of the form {full_team_name: abbreviation}.
        """
        #find the link
        url = f'{self.base_url}/teams'
        self.logger.debug(f'Extracting team names from URL: {url}')

        try:
            webpage = self.fetch_webpage(url)

            #extract teamnames
            full_team_names = self.strip_tag(webpage, 'h2')[:30]
            full_team_names = [x.lower().replace(' ', '-') for x in full_team_names]
        
            #extract abbreviations
            pattern = r'/nba/team/stats/_/name/([a-zA-Z]{2,4})/'
            abbr_team_names = re.findall(pattern, webpage)[:30]
            abbr_team_names = [x.lower() for x in abbr_team_names]
    
            #validate matching lengths before zipping
            if len(full_team_names) != len(abbr_team_names):
                self.logger.error(f'Mismatch between full team names ({len(full_team_names)}) and abbreviated team names ({len(abbr_team_names)})')
                raise

            #dictionary of the form {bos: boston-celtics, ...}
            teams = {abbr: full for abbr, full in zip(abbr_team_names, full_team_names)}
            self.logger.debug(f'Successfully extracted {len(teams)} teams')
            return teams
        
        except Exception as e:
            self.logger.error(f'Error extracting team names: {e}')
            raise
    
    #get {team_name: team_abbr}
    def get_team_names(self, update=False):
        """
        First attempts to read from saved JSON data in this instance. Otherwise, retrieves team names 
        from a JSON file if available. If not available or if update is True,
        it fetches team names from the webpage and saves them to a JSON file.

        Args:
            update (bool): Whether to force a refresh of the team names data.
        
        Returns:
            dict: A dictionary of team names and abbreviations.
        """
        #attempt to read saved information
        if self.team_names and not update:
            self.logger.debug(f'Reading cached team names.')
            return self.team_names

        #make the filepath to read from
        file_path = self.team_names_file
        file_path.parent.mkdir(parents=True, exist_ok=True) #ensure directory structure exists

        #attempt reading existing file
        if file_path.exists() and not update:
            self.logger.debug(f'Reading existing team names file from: {file_path}')

            try:
                with open(file_path, 'r') as f:
                    team_names = json.load(f)

                #save to cache
                self.team_names = team_names
                self.logger.debug(f'Successfully loaded team names from: {file_path}')            
                return team_names
            
            except Exception as e:
                self.logger.error(f'Error reading existing team names file: {e}')
                raise
        
        #extract from the webpage and overwrite file
        try:
            self.logger.debug('Fetching new team names')

            #save to cache
            team_names = self.extract_team_names()
            self.team_names = team_names

            with open(file_path, 'w') as f:
                json.dump(team_names, f, indent=4)

            self.logger.debug(f'Successfully saved team names to: {file_path}')
            return team_names
        
        except Exception as e:
            self.logger.error(f'Error retrieving team names: {e}')
            raise



    #----------------------------------------------------------
    #extract team_stats (df), includes player id used by ESPN
    def extract_team_stats(self, team_abbr):
        """
        Extract team statistics from the webpage for a given team abbreviation.

        Args:
            team_abbr (str): Team abbreviation (e.g., 'lal' for Los Angeles Lakers).

        Returns:
            pd.DataFrame: Extracted team statistics as a DataFrame.
            dict: Dictionary of players on the team and IDs.
        """
        #get team names to build the link
        team_names = self.get_team_names()
        full_name = team_names[team_abbr]

        #find the link
        url = f'{self.base_url}/team/stats/_/name/{team_abbr}/{full_name}'
        self.logger.debug(f'Extracting team stats from webpage: {url}')

        try:
            webpage = self.fetch_webpage(url)
            
            #extract headers
            head_tables = self.clean_tables(self.extract_table_data(webpage, section='head'))
            unpeel = list(itertools.chain.from_iterable(head_tables)) #have to flatten twice, due to format of html
            flattened_headers = list(itertools.chain.from_iterable(unpeel))

            #extract body data
            body_tables = self.clean_tables(self.extract_table_data(webpage, section='body'))
            bodies = list(zip(*body_tables))
            flattened_bodies = [list(itertools.chain(*body)) for body in bodies]

            #make dataframe and remove totaled row, drop duplicate columns
            df = pd.DataFrame(flattened_bodies, columns=flattened_headers)
            df = df[df.iloc[:, 0] != 'Total']
            df = df.loc[:, ~df.columns.duplicated()]

            #rename Name column to PLAYER for codebase consistency
            df.rename(columns={'Name': 'PLAYER'}, inplace=True)

            #attempt fixing player name formatting
            df['PLAYER'] = df['PLAYER'].apply(lambda x: re.sub(r'[^\w\s-]', '', x.lower()).replace(' ', '-'))

            #extract the players with their ids
            pattern = r'data-player-uid.*?/_/id/(\d+)/(.*?)"'
            matches = re.findall(pattern, webpage)
            player_to_id = {name: id for id, name in matches}

            #put the player id mapping into the dataframe
            df['PLAYERID'] = df['PLAYER'].map(player_to_id)

            if df['PLAYERID'].isna().any():
                self.logger.error(f'Mismatch in player name designation')
                raise ValueError(f'Some player ids are missing in the dataframe for ({team_abbr}).')
            
            #convert to ints and floats from strings
            str_cols = ['PLAYER']
            int_cols = ['GP', 'GS', 'PLAYERID']
            float_cols = [col for col in df.columns if col not in str_cols + int_cols] #all other columns

            df[int_cols] = df[int_cols].astype('int')
            df[float_cols] = df[float_cols].astype('float')

            #print(df)
            #print(df.dtypes)

            return df
        
        except Exception as e:
            self.logger.error(f'Error extracting team stats from webpage: {e}')
            raise
    
    #get team_stats (df)
    def get_team_stats(self, team_abbr, update=False):
        """
        Retrieve team statistics from a local file or scrape them from the website.

        Args:
            team_abbr (str): Team abbreviation (e.g., 'lal' for Los Angeles Lakers).
            update (bool): If True, fetch and save new data. If False, use cached data if available.

        Returns:
            pd.DataFrame: Team statistics as a DataFrame.
            dict: Dictionary with player names as keys and player IDs.
        """
        #make the filepath to read from
        stats_file_path = self.teams_dir / team_abbr / f'stats.csv'
        stats_file_path.parent.mkdir(parents=True, exist_ok=True) #ensure directory structure exists

        #attempt reading from existing files
        if stats_file_path.exists() and not update:
            self.logger.debug(f'Reading existing team stats from: {stats_file_path}')
            try:
                df = pd.read_csv(stats_file_path)
                return df
            
            except Exception as e:
                self.logger.error(f'Error reading existing files: {e}')
                raise
        
        #extract from the webpage and overwrite file
        try:
            self.logger.debug(f'Fetching new team stats for: {team_abbr}')        
            df = self.extract_team_stats(team_abbr)

            df.to_csv(stats_file_path, index=False)

            self.logger.debug(f'Successfully saved team stats to: {stats_file_path}')
            return df
        
        except Exception as e:
            self.logger.error(f'Failure to retrieve team stats for: {team_abbr}: {e}')
            raise ValueError(f'Error retrieving team stats for: {team_abbr}')



    #----------------------------------------------------------
    #extract team_schedule (df) from webpage
    def extract_team_sched(self, team_abbr, season_type=2):
        """
        Extract team schedule from the webpage for a given team abbreviation.

        Args:
            team_abbr (str): Team abbreviation (e.g., 'lal' for Los Angeles Lakers).
            season_type (int): 1 for preseason, 2 for regular season

        Returns:
            pd.DataFrame: Extracted team schedule as a DataFrame.
        """
        #find the link
        url = f'{self.base_url}/team/schedule/_/name/{team_abbr}/seasontype/{season_type}'
        self.logger.debug(f'Extracting team schedule from webpage: {url}')

        try:
            webpage = self.fetch_webpage(url)

            #extract body data, this webpage has everything in the body for some reason
            body_tables = self.clean_tables(self.extract_table_data(webpage, section='body'), ignore_columns=[1, 2])
            rows = body_tables[0]

            season, *body = rows

            #explicit typecast headers, preseason table organization is different from regular season
            headers = ['DATE', 'OPPONENT', 'RESULT', 'W-L',	'Hi Points', 'Hi Rebounds',	'Hi Assists'] 

            #make dataframe
            df = pd.DataFrame(body, columns=headers)

            date_pattern = r'[A-Za-z]{3}, [A-Za-z]{3} \d{1,2}'
            df = df[df['DATE'].str.match(date_pattern, na=False)]

            #fix team and opponent columns
            df['TEAM'] = team_abbr
            
            #clean html from column names before accessing
            df.columns.values[1] = 'OPP'
            df.columns.values[2] = 'RESULT'

            #processing and cleaning
            df['LOC'] = df['OPP'].apply(lambda x: self.clean_string(x, select_index=0).lower())
            df['OPP'] = df['OPP'].apply(lambda x: self.select_regex(x, r'/name/(.*?)/'))
            
            df['GAMEID'] = df['RESULT'].apply(lambda x: self.select_regex(x, r'gameId[/=](.*?)[/"]')) #this is to accommodate live game links
            df['GAMEID'] = df['GAMEID'].apply(lambda x: int(x) if isinstance(x, str) and x.isdigit() else None) #postponements and cancellations
            
            df['RESULT'] = df['RESULT'].apply(lambda x: self.clean_string(x, select_index=0))

            df.loc[~df['RESULT'].isin(['W', 'L']), ['RESULT']] = 'TBD'

            #drop useless stuff
            df = df.drop(columns=['W-L', 'Hi Points', 'Hi Rebounds', 'Hi Assists'])

            #print(df)
            #print(df.dtypes)
            return df
        
        except Exception as e:
            self.logger.error(f'Error extracting team schedule from webpage: {e}')
            raise        

    #get team_schedule (df)
    def get_team_sched(self, team_abbr, update=False):
        """
        Retrieves a team's schedule data as a pandas DataFrame. The method first checks for local cached data
        and fetches new data from the web if the `update` parameter is set to True or if no local
        data exists.

        NOTE: If update is true, or reading for the first time, compiles preseason + regular season data.

        Args:
            team_abbr (str): Team abbreviation (e.g., 'lal' for Los Angeles Lakers).
            update (bool): Optional setting to force update the file from webpage.

        Returns:
            pd.DataFrame: Extracted team schedule as a DataFrame.
        """
        #make the filepath to read from
        sched_file_path = self.teams_dir / team_abbr / f'schedule.csv'
        sched_file_path.parent.mkdir(parents=True, exist_ok=True) #ensure directory structure exists

        #attempt reading from existing files
        if sched_file_path.exists() and not update:
            self.logger.debug(f'Reading existing team schedule file: {team_abbr}')
            try:
                df = pd.read_csv(sched_file_path)
                return df
            
            except Exception as e:
                self.logger.error(f'Error reading existing file: {e}')
                raise
        
        #extract from the webpage and overwrite file
        try:
            self.logger.debug(f'Fetching new team schedule: {team_abbr}')        
            preseason = self.extract_team_sched(team_abbr, season_type=1)
            regseason = self.extract_team_sched(team_abbr, season_type=2)

            df = pd.concat([preseason, regseason], axis=0).reset_index(drop=True)

            df.to_csv(sched_file_path, index=False)

            self.logger.debug(f'Successfully saved team schedule to: {sched_file_path}')
            return df
        
        except Exception as e:
            self.logger.error(f'Error retrieving team schedule for {team_abbr}: {e}')
            raise

    #get team_schedule but only the games that have been played (df)
    def get_team_sched_played(self, team_abbr):
        """
        Retrieves a team's schedule data as a pandas DataFrame. Only returns the games that
        have been played and thus have an result.

        REQUIRES schedules already being in place (will work anyway though).

        Args:
            team_abbr (str): Team abbreviation (e.g., 'lal' for Los Angeles Lakers).
            update (bool): Optional setting to force update the file from webpage.

        Returns:
            pd.DataFrame: Extracted team schedule as a DataFrame.
        """
        df = self.get_team_sched(team_abbr)
        df = df[df['RESULT'] != 'TBD']
        return df



    #----------------------------------------------------------
    #compile {player_name: team_abbr} in /teams/player_to_team.json
    def get_player_to_team(self, update=False):
        """
        Sets or updates the mapping of all players to their respective teams and saves it to a JSON file.

        REQUIRES all team stats to be set already.

        If the player-to-team mapping file already exists and the update parameter is False, 
        the method will skip processing and just return the file. If the file doesn't exist or if 
        update is True, the method will read player data from each team's players.json file and map player names 
        to their corresponding team names. The resulting dictionary is saved in a file called 
        'player_to_team.json' under the 'data' directory and returned.

        Parameters:
            update (bool): If True, forces an update of the player-to-team mapping, even if 
                        the file already exists. Defaults to False.

        Returns:
            player_to_team (dict): Mapping of each player to their team.
        """
        if self.player_to_team and not update:
            self.logger.debug(f'Reading cached player-to-team mapping.')
            return self.player_to_team

        #file that contains a dictionary with player names with corresponding team
        file_path = self.player_to_team_file
        file_path.parent.mkdir(parents=True, exist_ok=True) #ensure parent directory exists

        #dont update
        if file_path.exists() and not update:
            self.logger.debug(f'Reading existing player-to-team file from: {file_path}')

            try:
                with open(file_path, 'r') as f:
                    player_to_team = json.load(f)
            
                #cache
                self.player_to_team = player_to_team
                self.logger.debug(f'Successfully loaded player-to-team file from: {file_path}')            
                return player_to_team

            except Exception as e:
                self.logger.error(f'Error reading existing player-to-team file: {e}')
                raise

        #iterate through folders to build the file data
        self.logger.debug(f'Consolidating player-to-team data')

        player_to_team = {}
        for team_abbr in self.get_team_names().keys():
            df = self.get_team_stats(team_abbr)
            players = df['PLAYER'].tolist()

            for player in players:
                player_to_team[player] = team_abbr

        #save newly compiled data to file
        with open(file_path, 'w') as f:
            json.dump(player_to_team, f, indent=4)
        
        #cache
        self.player_to_team = player_to_team
        self.logger.debug(f'Player to team mapping updated and saved to: {file_path}')
        return player_to_team

    #returns team_abbr
    def find_player_team(self, player):
        """
        Finds the team abbreviation for a given player by loading the player-to-team mapping.

        Parameters:
            player_name (str): The name of the player to find the corresponding team for.

        Returns:
            str: The abbreviation of the team the player belongs to.
        """
        player_to_team = self.get_player_to_team()
        
        team_abbr = player_to_team.get(player)
        
        #check that it is in the mapping
        if not team_abbr:
            self.logger.error(f'Player to team mapping is blank for ({player})')
            raise

        return team_abbr

    #compile {game_id: 0} in /games/game_ids.json
    def get_game_ids(self, update=False):
        """
        Sets or updates the file containing all game_ids for games that have been played and saves it to a JSON file.

        REQUIRES all player game logs to be set already.

        NOTE: DOES NOT INCLUDE PRESEASON GAMES (based off team regular season schedule)

        If the file already exists and the update parameter is False, 
        the method will skip processing and return the file data. If the file doesn't exist or 
        if update is True, the method will compile the game_ids from each team and return it.

        Parameters:
            update (bool): If True, forces an update of the game_ids. Default is False.

        Returns:
            game_ids (dict): All game_ids.
        """
        if self.game_ids and not update:
            self.logger.debug(f'Reading cached game_ids.')
            return self.game_ids
        
        #file that contains a dictionary with player names with corresponding team
        file_path = self.game_ids_file
        file_path.parent.mkdir(parents=True, exist_ok=True) #ensure parent directory exists

        #dont update
        if file_path.exists() and not update:
            self.logger.debug(f'Reading existing game_ids file from: {file_path}')
            
            try:
                with open(file_path, 'r') as f:
                    game_ids = json.load(f)
            
                #cache
                self.game_ids = game_ids
                self.logger.debug(f'Successfully loaded game_ids file from: {file_path}') 
                return game_ids

            except Exception as e:
                self.logger.error(f'Error reading existing game_ids file: {e}')
                raise

        #iterate through every team, and collect all unique game_ids that have been played by team during regular season. 
        #this updates faster on espn than individual game logs.
        game_ids = {}
        for team_abbr in self.get_team_names().keys():
            sched = self.get_team_sched_played(team_abbr)
            ids = sched['GAMEID']

            for id in ids:
                game_ids[id] = 0

        #save newly compiled data
        with open(file_path, 'w') as f:
            json.dump(game_ids, f, indent=4)
        
        #cache
        self.game_ids = game_ids
        self.logger.debug(f'Game id mapping updated and saved to: {file_path}')
        return game_ids




    #----------------------------------------------------------
    #extract player game_log (df) from webpage
    def extract_player_game_log(self, team_abbr, player):
        """
        Extracts the game log statistics for a specific player from a team's game log webpage.

        Parameters:
            team_abbr (str): The abbreviation of the team (e.g., 'LAL' for Los Angeles Lakers).
            player_name (str): The name of the player (e.g., 'LeBron James').

        Returns:
            pd.DataFrame: A DataFrame containing the player's game log statistics.
        """
        #get a list of players from the team and find their id
        team_stats = self.get_team_stats(team_abbr)
        player_id = team_stats.loc[team_stats['PLAYER'] == player, 'PLAYERID'].iloc[0]

        #find link
        url = f'{self.base_url}/player/gamelog/_/id/{player_id}/{player}'
        self.logger.debug(f'Extracting player game log stats from URL: {url}')

        try:        
            webpage = self.fetch_webpage(url)

            #extract headers
            head_tables = self.clean_tables(self.extract_table_data(webpage, section='head'))
            flattened_headers = head_tables[0][0]

            #extract body data
            body_tables = self.extract_table_data(webpage, section='body')
            flattened_bodies = list(itertools.chain(*body_tables))

            #make dataframe, clean and preprocess, and drop duplicate columns
            df = pd.DataFrame(flattened_bodies, columns=flattened_headers)

            #clean html
            df.columns = df.columns.str.upper()

            #remove bad rows based on date
            valid_date_pattern = r'^[A-Za-z]{3} \d{1,2}/\d{1,2}$'
            df = df[df['DATE'].str.match(valid_date_pattern)]
            
            #separate location home/away and opponent team abbr
            df['LOC'] = df['OPP'].apply(lambda x: self.strip_tag(x, 'span', 'class="pr2"')[0])
            df['OPP'] = df['OPP'].apply(lambda x: self.clean_string(x, select_index=2).lower())

            #separate the win/loss and the final score
            df['SCORE'] = df['RESULT'].apply(lambda x: self.clean_string(x))
            df['GAMEID'] = df['RESULT'].apply(lambda x: self.select_regex(x, r'/gameId/(.*?)/'))
            df['RESULT'] = df['RESULT'].apply(lambda x: self.strip_tag(x, 'div', 'class="ResultCell')[0])

            #separate made-attempts into two separate columns
            df[['FGM', 'FGA']] = df['FG'].str.split('-', expand=True)
            df[['3PM', '3PA']] = df['3PT'].str.split('-', expand=True)
            df[['FTM', 'FTA']] = df['FT'].str.split('-', expand=True)
            df = df.drop(columns=['FG', '3PT', 'FT'])

            #convert to ints and floats from strings
            str_cols = ['DATE', 'OPP', 'RESULT', 'LOC', 'SCORE']
            float_cols = ['FG%', '3P%', 'FT%']
            int_cols = [col for col in df.columns if col not in str_cols + float_cols] #all other columns

            df[int_cols] = df[int_cols].astype('int')
            df[float_cols] = df[float_cols].astype('float')
            
            #print(df)
            #print(df.dtypes)
            
            return df
        
        except Exception as e:
            self.logger.error(f'Error extracting player game log stats: {e}')
            raise

    #get player game_log (df)
    def get_player_game_log(self, player_name, update=False):
        """
        Retrieves the game log for a specific player. The method first attempts to locate the player's team, checks 
        for existing data files, and either reads the data from a local CSV file or fetches updated stats from the web.

        Parameters:
        -----------
        player_name : str
            The name of the player whose game log is being retrieved.
        update : bool, optional
            If True, forces fetching new data from the web even if a local file exists (default is False).

        Returns:
        --------
        pd.DataFrame
            A pandas DataFrame containing the player's game log data.
        """
        #find the team the player is on
        team_abbr = self.find_player_team(player_name)
        
        #make the filepath to read from
        player_file_path = self.teams_dir / team_abbr / 'players' / f'{player_name}.csv'
        player_file_path.parent.mkdir(parents=True, exist_ok=True) #ensure directory structure exists

        #attempt reading from existing files
        if player_file_path.exists() and not update:
            self.logger.debug(f'Reading existing player game_log: {team_abbr}, {player_name}')
            try:
                df = pd.read_csv(player_file_path)    
                return df
            
            except Exception as e:
                self.logger.error(f'Error reading existing player game_log file: {e}')
                raise
        
        #extract from the webpage and overwrite file
        try:
            self.logger.debug(f'Fetching new player game_log: {team_abbr}, {player_name}')        
            df = self.extract_player_game_log(team_abbr, player_name)

            df.to_csv(player_file_path, index=False)

            self.logger.debug(f'Successfully saved player game_log to: {player_file_path}')
            return df
        
        except Exception as e:
            self.logger.error(f'Error retrieving player game_log for {team_abbr}, {player_name}: {e}')
            raise



    #----------------------------------------------------------
    #extract injuries (df) from webpage
    def extract_injuries(self):
        """
        Extracts injury data from a webpage and returns it as a pandas DataFrame.

        Returns:
        --------
        pd.DataFrame
            A DataFrame containing the injury data, with cleaned and flattened headers and rows.
        """
        #find link
        url = f'{self.base_url}/injuries'
        self.logger.debug(f'Extracting injury data from URL: {url}')

        try:
            webpage = self.fetch_webpage(url)

            #extract headers
            head_tables = self.clean_tables(self.extract_table_data(webpage, section='head'))
            flattened_headers = head_tables[0][0]

            #extract body data
            body_tables = self.clean_tables(self.extract_table_data(webpage, section='body'))
            flattened_bodies = list(itertools.chain(*body_tables))

            #make dataframe, clean and preprocess, and drop duplicate columns
            df = pd.DataFrame(flattened_bodies, columns=flattened_headers)
            return df
        
        except Exception as e:
            self.logger.error(f'Error extracting injury data: {e}')
            raise
    
    #get injuries (df)
    def get_injuries(self, update=False):
        """
        Retrieves injury data as a pandas DataFrame. The method first checks for local cached data
        and fetches new data from the web if the `update` parameter is set to True or if no local
        data exists.

        Parameters:
        -----------
        update : bool, optional
            If True, fetches the latest injury data from the web (default is True).

        Returns:
        --------
        pd.DataFrame
            A pandas DataFrame containing the injury data.
        """
        if self.injury and not update:
            self.logger.debug(f'Reading from cached injury file.')
            return self.injury

        #check for cached injury file
        injury_file_path = self.injury_file
        injury_file_path.parent.mkdir(parents=True, exist_ok=True)  #ensure directory structure exists

        #check for existing file
        if injury_file_path.exists() and not update:
            self.logger.debug(f'Reading existing injury data from {injury_file_path}')

            try:
                self.logger.warning(f'Might want to check when the injury file was last updated.')
                df = pd.read_csv(injury_file_path)

                self.injury = df
                return df
            
            except Exception as e:
                self.logger.error(f'Error reading existing injury data file: {e}')
                raise

        #fetch new data and overwrite file
        try:
            self.logger.debug('Fetching new injury data from the web')
            df = self.extract_injuries()
            df.to_csv(injury_file_path, index=False)

            #cache
            self.injury = df

            self.logger.debug(f'Successfully saved injury data to {injury_file_path}')
            return df
        
        except Exception as e:
            self.logger.error(f'Error retrieving or saving injury data: {e}')
            raise



    #----------------------------------------------------------
    #extract game_data (df) from webpage
    def extract_game(self, game_id):
        """
        Extracts game boxscore data from a webpage and returns it as a pandas DataFrame.

        Returns:
        --------
        pd.DataFrame
            A DataFrame containing the game data.
        """
        #find link
        game_id = int(game_id)
        url = f'{self.base_url}/boxscore/_/gameId/{game_id}'
        self.logger.debug(f'Extracting game boxscore data from URL: {url}')

        try:
            webpage = self.fetch_webpage(url)

            #extract body data, for some reason everything is in the body on these pages
            body_tables = self.extract_table_data(webpage, section='body')

            #scoreline has the teams with scores by quarter
            #t1 and t2 are team1 and team2 respectively
            #standings contains all the other tables which are not directly useful
            scoreline, t1_players, t1_stats, t2_players, t2_stats, *standings = body_tables

            away = scoreline[0][0].lower()
            home = scoreline[1][0].lower()

            #process each team separately and then combine into a single dataframe
            def process_team(player_table, stats_table, team_abbr=None, away=None):
                rows = [player + stats for player, stats in zip(player_table, stats_table)]

                headers, *data = rows
                
                #fix headers
                df = pd.DataFrame(data, columns=headers)
                df.columns = df.columns.map(lambda x: self.clean_string(x, strip_tags=['div']))
                df.columns.values[0] = 'PLAYER'

                #clean player column
                pattern = r'data-player-uid.*?/_/id/\d+/(.*?)"'
                df['PLAYER'] = df['PLAYER'].apply(lambda x: self.select_regex(x, pattern))
                
                #remove bad rows (players that didnt get playtime, or extraneous row information)
                df['MIN'] = pd.to_numeric(df['MIN'], errors='coerce')
                df = df[df['MIN'] > 0]

                #return +/- as dtype int64 instead of strings
                df['+/-'] = pd.to_numeric(df['+/-'], errors='coerce')

                #fill in team name information and home or away information
                df['TEAM'] = team_abbr
                df['LOC'] = '@' if away else 'vs'

                #separate made-attempts into two separate columns
                df[['FGM', 'FGA']] = df['FG'].str.split('-', expand=True)
                df[['3PM', '3PA']] = df['3PT'].str.split('-', expand=True)
                df[['FTM', 'FTA']] = df['FT'].str.split('-', expand=True)
                df = df.drop(columns=['FG', '3PT', 'FT'])

                #convert to ints and floats from strings
                str_cols = ['PLAYER', 'TEAM', 'LOC']
                int_cols = [col for col in df.columns if col not in str_cols] #all other columns

                df[int_cols] = df[int_cols].astype('int')

                #add in the 2P stats
                df['2PM'] = df['FGM'] - df['3PM']
                df['2PA'] = df['FGA'] - df['3PA']

                #print(df)
                return df

            #combine data
            t1 = process_team(t1_players, t1_stats, team_abbr=away, away=True)
            t2 = process_team(t2_players, t2_stats, team_abbr=home, away=False)

            df = pd.concat([t1, t2], axis=0)
            return df
        
        except Exception as e:
            self.logger.error(f'Error extracting game data: {e}')
            raise        
        
    #get game_data (df)
    def get_game(self, game_id, update=False):
        """
        Retrieves game boxscore data as a pandas DataFrame. The method first checks for local cached data
        and fetches new data from the web if the `update` parameter is set to True or if no local
        data exists, and writes it to a file.

        Parameters:
        -----------
        update : bool, optional
            If True, fetches the latest game data from the web (default is False).

        Returns:
        --------
        pd.DataFrame
            A pandas DataFrame containing the game data.
        """
        #check for saved game file
        file_path = self.games_dir / f'{game_id}.csv'
        file_path.parent.mkdir(parents=True, exist_ok=True)  #ensure directory structure exists

        #check for existing file
        if file_path.exists() and not update:
            self.logger.debug(f'Reading existing game data from {file_path}')

            try:
                df = pd.read_csv(file_path)
                return df
            
            except Exception as e:
                self.logger.error(f'Error reading existing game data file: {e}')
                raise

        #fetch new data and overwrite file
        try:
            self.logger.debug('Fetching new game data from the web')
            df = self.extract_game(game_id)
            df.to_csv(file_path, index=False)

            self.logger.debug(f'Successfully saved game data to: {file_path}')
            return df
        
        except Exception as e:
            self.logger.error(f'Error retrieving or saving game data: {e}')
            raise



    #----------------------------------------------------------
    #compile team_stats, player_ids, player_teams, game_ids
    def set_all_team_data(self, update=True, sleep=0.1):
        """
        Processes and updates the statistics and schedules for all teams and updates several mapping.

        This method first retrieves the list of all teams and their abbreviations, and then for each team,
        collects the team's broader stats and schedule.

        Parameters:
            update (bool): If True, forces an update of every team stats and schedule. Defaults to True.

        Returns:
            None
        """
        #get all team names and update if necessary
        team_names = self.get_team_names(update=update)
        self.logger.info(f'Completed setting team_names.')

        #write estimated time to complete
        #doubled because sleep is called twice, and +0.5 estimated processing time
        eta = (2 * sleep * len(team_names)) + (0.5 * len(team_names))
        self.logger.info(f'Estimated time to complete setting all team data: ~{int(eta // 60)}m{int(eta % 60)}s')

        #set all each teams stats
        for team_abbr, team_name in team_names.items():
            self.logger.debug(f"Processing stats for: {team_abbr} -- {team_name}")
            
            _ = self.get_team_stats(team_abbr, update=update)
            time.sleep(sleep)

            _ = self.get_team_sched(team_abbr, update=update)
            time.sleep(sleep)
        
        self.logger.info(f'Completed setting all stats and schedules.')

        print()
        return
    
    #compile game_logs for all players
    def set_all_player_game_logs(self, update=True, sleep=0.1):
        """
        Fetches and stores game logs for all players in the player-to-team mapping.

        Parameters:
            update (bool): Whether to force fetching new stats even if the player stats already exist. Default is True.
            sleep (float): How long to wait between each request, to prevent overwhelming server. Default is 0.2
        """
        #set player to team mapping
        player_to_team = self.get_player_to_team(update=update)
        self.logger.info(f'Completed setting player-to-team mapping.')

        #write out estimated time to finish
        #sleep is called once +0.5 estimated processing time
        eta = (sleep + 0.5) * len(player_to_team)
        self.logger.info(f'Estimated time to complete setting all player game logs: ~{int(eta // 60)}m{int(eta % 60)}s')

        #extractions or reads
        for player in player_to_team.keys():
            self.logger.debug(f'Fetching game log for player: {player}')

            _ = self.get_player_game_log(player, update=update)
            time.sleep(sleep) #no ddos
        
        self.logger.info(f'Completed setting player game_logs for {len(player_to_team)} players.')

        print()
        return

    #compile game_data for all games
    def set_all_game_data(self, update=False, sleep=0.1):
        """
        Processes and updates the game data for all games played so far.

        This method first retrieves the list of all game ids. For each game,
        it retrieves and processes the game boxscore data.

        Parameters:
            update (bool): If True, forces an update of ALL games from the webpages, 
                        even if they already exist. Defaults to False, which updates
                        only missing game data.

        Returns:
            None
        """
        #update the game ids file
        game_ids = self.get_game_ids(update=update)
        self.logger.info(f'Completed setting game_ids.')

        #select only missing games
        existing_game_ids = set(self.select_regex(f, r'(\d+)\.') for f in os.listdir(self.games_dir) if f.endswith('.csv')) 
        all_game_ids = set(game_ids.keys())

        missing_game_ids = all_game_ids - existing_game_ids
        
        #update all or only the missing games depending on argument
        if update:
            update_game_ids = all_game_ids
            self.logger.info(f'Force updating every game file.')
        else:
            update_game_ids = missing_game_ids
            self.logger.info(f'Updating only missing game files.')

        #write estimated time to complete
        #sleep is called once per, and +0.5 estimated processing time
        eta = (sleep + 0.5) * len(update_game_ids)
        self.logger.info(f'Estimated time to complete setting boxscore data for all games: ~{int(eta // 60)}m{int(eta % 60)}s')

        for game_id in update_game_ids:
            _ = self.get_game(game_id, update=update)

        self.logger.info(f'Completed game boxscore data for {len(update_game_ids)} games.')

        print()
        return
        
    #compile everything
    def set_all(self, sleep=0.1):
        """
        Updates everything. Hard forces webpage reads.
        """
        self.set_all_team_data(update=True, sleep=sleep)
        self.set_all_player_game_logs(update=True, sleep=sleep)
        self.set_all_game_data(update=True, sleep=sleep)
        return
    
    #compile everything, and only the new individual games
    def set_new(self, sleep=0.1):
        """
        Updates team rosters, and only fetches new games.
        """
        self.set_all_team_data(update=True, sleep=sleep)
        self.set_all_player_game_logs(update=True, sleep=sleep)
        self.set_all_game_data(update=False, sleep=sleep)
        return


    #----------------------------------------------------------
    #TODO: validate everything... this is kind of cooked because players switch teams and other issues
    def validate(self):
        
        #read team_names file
        try:
            with open(self.team_names_file, 'r') as f:
                team_names = json.load(f)
        except Exception as e:
            self.logger.error(f'Failure to read team_names file: {e}')
            raise

        #read player_to_team mapping file
        try: 
            with open(self.player_to_team_file, 'r') as f:
                player_to_team = json.load(f)
        except Exception as e:
            self.logger.error(f'Failure to read player_to_team file: {e}')
            raise

        #read game_ids file
        try:
            with open(self.game_ids_file, 'r') as f:
                game_ids = json.load(f)
        except Exception as e:
            self.logger.error(f'Failure to read game_ids file: {e}')
            raise

        #write out estimated time to finish
        #
        eta = 1
        self.logger.info(f'Estimated time to complete validation: ~{int(eta // 60)}m{int(eta % 60)}s')

        for team in team_names.keys():
            sched_file = self.teams_dir / team / 'schedule.csv'
            
            #attempt to read the schedule file for each team
            try:
                sched = pd.read_csv(sched_file)
            except Exception as e:
                self.logger.error(f'Failed to access schedule ({team}): {e}')
                raise
            
            #each game from the team's schedule
            for sched_entry in sched.itertuples(index=False):
                game_id = sched_entry.GAMEID

                game_file = self.games_dir / f'{game_id}.csv'

                #attempt to check the game data and validate it
                try: 
                    game = pd.read_csv(game_file)
                except Exception as e:
                    self.logger.error(f'Failed to read game ({game_id}) from schedule for team ({team}): {e}')
                    raise
                
                #each data entry from the game in question, check the player is on the right team
                for game_entry in game.itertuples(index=False):
                    player, player_team = game_entry.PLAYER, game_entry.TEAM

                    if player_to_team[player] != player_team:
                        self.logger.error(f'Failed to validate player ({player}) on team ({player_to_team[player]}) for game ({game_id})')                        
        return



#nbaext = ESPNNBAExtractor()
#nbaext.set_logger_level(logging.INFO)
#nbaext.set_all()

#nbaext.set_all_team_data(update=True)
#nbaext.set_all_player_game_logs(update=True)
#nbaext.set_all_game_data(update=True)

#nbaext.get_game_ids(update=True)

#print(nbaext.get_game(401705051, update=False))
#nbaext.get_team_sched('bos')




#for t in nbaext.get_team_names().keys():
#    nbaext.extract_team_stats(t)
#    nbaext.get_team_stats(t, update=True)

#a = (nbaext.get_game('401716976', update=False))
#print(a)

#nbaext.set_all_team_data(update=True)
#nbaext.set_all_player_game_logs(update=True)
#nbaext.set_all_game_data(update=True)
#nbaext.logger.info(f'finished.')


#print(nbaext.get_injuries())

#print(nbaext.get_player_game_log('bogdan-bogdanovic'))
#nbaext.extract_team_sched('cha')

#nbaext.get_team_sched('atl')




#nbaext.extract_teams('https://www.espn.com/nba/teams')

#nbaext.extract_team_stats('https://www.espn.com/nba/team/stats/_/name/lal')
#nbaext.extract_team_stats('https://www.espn.com/nba/team/stats/_/name/bos')
#print(nbaext.df.shape)

#nbaext.extract_game_log('https://www.espn.com/nba/player/gamelog/_/id/4065648/jayson-tatum')

#nbaext.df.to_csv('tatum.csv', index=False)


