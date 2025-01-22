# Sports betting woohoo!!!

This repo will support sports betting on the NBA. Data is extracted from ESPN, and then used to estimate expected stats for a player against some team.

## Methodology

1. Extract all the data from ESPN. Luckily no security on the website. The folder **/Extractors/** contains all the stuff used to extract data from the website. It uses mostly just get-requests and scans the webpage for data, compiling it into a csv file for most cases. ***This will require an extractor for betting lines for comparison in the future.***

2. Make player embeddings. This follows the Word2Vec skip-gram strategy, but instead of context words, uses expected stats as output. Note that adding more layers without adding non linearity is mathematically the same as just doing a single layer, and adding non linearity would ruin the linear relationships between embeddings.
    - Inputs are one-hot encoded players, so like [0 1 0 0 0 ...] for the current player.
    - Outputs are a combination of many things:
        - The current player's stats for a game: [PTS, REB, 3PM, ...] 
        - Team average stats, excluding the current player. **This only averages players who usually play at least a certain number of minutes, which is biased and unweighted.**
        - Opps team average stats. **This also contains bias.**
        - Normalize the outputs by stat to the range 0 to 1. For +/- and PTS, use z-score and then minmax. For others, just use minmax. **Some stats will be normalized on the same scale, such as 3PA and 3PM.**
    - Hidden layer after training is the embedding, which should hopefully represent the current player's stats.
    - Instead of CrossEntropyLoss like Word2Vec, use MSELoss to minimize stat differences.

3. Make stat predictions. Right now it's a simple feed forward network. ***Use attention to capture player to player interactions?***
    - Inputs are a combination of:
        - The current player's embedding.
        - Home or away one-hot: [0 1] or [1 0]
        - Team average embedding, which is currently only calculated by taking the average of other influential players on the team **(who play on average a certain number of minutes, this is biased and ideally fixed)**.
        - Opps team average embedding, which is also calculated by the high-minutes players on each team **(also contains bias)**.
    - Outputs are the player's expected stat. It can compute a combination of them (like PRA or RA). These are also normalized using a scale of choice.
    - After training, denormalize output estimates using the same scale from the DatasetBuilder to compare true differences with expected values.


## Big changes that have been made since project inception

* **Embeddings and Embedding Architecture:**
    - Add in 2P stats, which is FG - 3P. This might help explicitly identify players who are predominantly 2P shooters.
        - This can be modified in the ESPNExtractor to setup the dataset so it doesn't have to be manually calculated each time, and just read from the .csv file instead.
    - Additionally, include the current player's MIN, since their minutes played each game is extremely important. 
        - The current player's stats for a game should become: [MIN, PTS, REB, 3PM, ... 2PM, 2PA]
    - I want the outputs to better reflect each player's contribution to a game outcome. Instead of evenly weighting each player's stats, weight them by minutes played, so those who get more playtime will have greater influence on overall team statline. 
        - Team average stats weighted by each player's minutes played, excluding the current player's stats: [PTS, REB, 3PM, ... 2PM, 2PA]
        - Opps team average stats, weighted by each player's minutes played: [PTS, REB, 3PM, ... 2PM, 2PA]
    - Hopefully this will better capture player's performance overall by including MIN and 2P stats.

    - Add CBOW in addition to Skip-Gram style embeddings.
        - For Skip-Gram, use one hot player as input, and MSELoss for stats as outputs. This could potentially be skewed due to outliers, consider a different loss function like L1 or Robust L1/L2.
        - For CBOW, use stats as input and one hot player as output using CrossEntropyLoss. Potential advantages are faster convergence and accounting for outliers. 
    - Consider ignoring team data.
        - No longer includes data about how a player performs against or with certain types of teams/players.



## Big changes that I will be making

* **Model Architecture:**
    - The biggest issues with the current simple feed forward is that it does not do a good job of capturing the other players in the game, besides the current player. It takes a simple average of 'influential' players, which can be biased and does not include other players who might normally play a lot less but might play more in certain kinds of games. - Consider an attention based architecture:
        - For each player, compute attention scores for each player. This might be able to capture player to player interactions in each game, and how impactful each player's performance is in a game.
        - Inputs can then be more specific and include each player's embedding, which is more informative than averaging embeddings.
        - For games where star players are injured or out, it would do a better job of finding players who have played in that game overall but generally don't get very many minutes.

* **Injury report:**
    - Must implement an injury report system. This is extremely important for making an up-to-date prediction. 
        - With an attention based model, injured players could be removed from the game and it could be far more impactful than just averaging embeddings.
    - Ideally also output a message or log indicating that important players are injured, which could mean other players get more time.

