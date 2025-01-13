import logging
import pandas as pd
from extractors import ESPNNBAExtractor
from dataset_builders import StatDatasetBuilder
from train import PlayerEmbeddings

nbaext = ESPNNBAExtractor()
#print(nbaext.get_team_sched('atl') )
#print(nbaext.get_team_sched_played('utah'))
#nbaext.get_game_ids(update=True)
nbaext.set_new()
#print(nbaext.extract_team_sched('atl'))
#nbaext.set_all_team_data()
#nbaext.set_all()

#embedder = PlayerEmbeddings(req_games=5, max_games=100, offset=0, embedding_size=100, lr=0.001, epochs=30, emb_dir='embeddings')
#x = embedder.load(update=False)
#print(len(x))



#db = AllPlayerEmbeddingDatasetBuilder(req_games=8, max_games=10)
#db.set_logger_level(logging.DEBUG)
#db.set_batches()
#print(db.Y)

pd.set_option('display.max_rows', None)  # Show all rows
pd.set_option('display.max_columns', None)  # Show all columns

#nbaext = ESPNNBAExtractor()
#print(nbaext.get_team_sched('bos', update=False))
#nbaext.set_all()
#gamelog = nbaext.extract_player_game_log('ny', 'og-anunoby')
#print(gamelog)

'''game = nbaext.extract_game(401716994)
print(game)
print(game.dtypes)

#filtering conditions
select_columns = ['MIN', '+/-', 'OREB', 'DREB', 'REB', 'AST', 'STL', 'BLK', 'TO', 'PF', 'PTS', 'FGM', 'FGA', '3PM', '3PA', 'FTM', 'FTA', '2PM', '2PA']
player='jd-davison'
is_player = game['PLAYER'] == player

if not is_player.any(): #break out when player didnt play in this game
    print(f'fail')
player_team = game.loc[is_player, 'TEAM'].iloc[0] #find the player's team at the time the game was played (consider trades)
on_team = game['TEAM'] == player_team

#filter dataframe using p (player), t (player's team wihtout the player), o (opponent team)
p_df = game[is_player][select_columns]
t_df = game[~is_player & on_team][select_columns]
o_df = game[~on_team][select_columns]

print(p_df)
print(t_df)
print(o_df)
'''

#nbaext = ESPNNBAExtractor()
#nbaext.set_logger_level(logging.DEBUG)
#nbaext.set_all_game_data(update=True)

#print(nbaext.extract_team_sched('lal'))



'''player = 'jordan-walsh'
player = 'jayson-tatum'
is_player = game['PLAYER'] == player
if is_player.any():
    print(True)
else:
    print(False)
print(is_player)

#if len(is_player) > 0:
#    print(is_player)
#else:
#    print('not here')    
player_team = game.loc[is_player, 'TEAM'].iloc[0] #find the player's team at the time of the game
on_team = game['TEAM'] == player_team
'''
#nbaext.set_all()