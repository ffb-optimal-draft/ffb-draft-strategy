# "Winning Draft Strategy" : Fantasy Football and Cheat sheet for PPR Leagues

- Canva Link: https://www.canva.com/design/DAFdXirVmBc/OQrB5TWPEVO8BS--8ldN6A/view?utm_content=DAFdXirVmBc&utm_campaign=designshare&utm_medium=link&utm_source=publishsharelink

- Ultimate Draft Guidebook PDF above. 

 ## Project Description
 Data Scientists using; pro-football-reference.com, webscraping data from football.fantasysports.yahoo.com, and yahoofantasy API, datasets to develop a machine learning model that will help to predict the players total season projections. 

 ## Project Goals
 - Create a fantasy football drafting strategy cheat sheet based off of historical NFL and fantasy league data.
 - Create a machine learning model that predicts the upcoming NFL FFL 2023 total season PPR points for Quarter Backs, Running Backs, Wide Recievers, Tight Ends.
 - Gather findings, draw conclusions and recommended next steps for Fantasy Football team owners.

 ## Questions to answer
 - What position should I draft this round?
 - Is it better to draft a RB, WR, TE or should draft a QB next?

 ## Initial Thoughts and Hypothesis
 Initially, the data records the past 7 years of Real-Word NFL Statistics and captured over 200 fantasy league. This large dataset will help draw accurate predictions for season projections.


 ## Planning
 - Perform Data acquisition and Preparation
 - Analyze Data features and analyze sports trends and patterns. 
 - Establish baseline of model using baseline methods. 
   * We took the mean of the number of ppr pts scored for a baseline of 113. 
 - Create Machine Learning model to predict and forecast season projections.
   * We used XGBRegressor and RidgeRegressor functions for our two primary models.
   * We split the data frame up into 4 different dataframes. One for each position.
   * We added new columns such as reception percentage, interception percentage, etc.
   * Afterwards, we trained our models on how many ppr points a player scored did from 2018-2020.
   * We validated our model on the years 2021 and 2022.
   * Finally our test/projection set is projections for the upcoming 2023 season.
 - Draw and record conclusions
   * Our conclusions are the model is predictings as expected.
   * The number of ppr points scored are within reason for each position.
   * The players in the top 25-50 for each position make sense and are within reason. 

 ## Data Dictionary

 |Target Variable | Definition|
 |-----------------|-----------|
 | Target | The number of ppr points a player scored in the following year. |

 | Feature  | Definition |
 |----------|------------|
 | Player | The name of each player in the dataset. |
 | Rk	| The rank of each player. |
 | Team	| The NFL Team Each player is rostered on. |
 | Pos	| The position each player is listed as. |
 | Age | The age number of each player. |
 | G  | The amount of games each player has played in the (regular) season. |
 | GS | The amount of games started each player has played in the (regular) season. |
 | Cmp	| The number of completed passes for each player. |
 | Pass_att	| The number of attempted passes for each player. |
 | Pass_yds	| The number of passing yards for each player. |
 | Pass_tds	| The number of passing touchdowns for each player. |
 | Int	| The number of interceptions each player. |
 | Rush_att	| The number of rushing attempts for each player. |
 | Rush_yard	| The number of rushing yards for each player. |
 | Y/A	| The average number of yards per attempts for each player. |
 | Rush_tds	| The number of rushing touchdowns for each player. |
 | Rec	| The number of completed recieving catches for each player. |
 | Y/R	| The average number of recieving yards per reception catches for each player. |
 | Rec_tds	| The number of recieving touchdowns for each player. |
 | Fmb	| The number of fumbles recorded for each player. |
 | Fl | The number of fumbles lost to opposing teams recorded for each player. |
 | Rush_rec_tds	| The total TDs(Rushing and Recieving) recorded for each player. |
 | Ppr_pts	| The total number of PPR season points for each player. |
 | Vbd	| The variance between each player's total fantasy point and the point total of a "baseline" player at the same position gives us a relative number, which can then be compared across positions. |
 | Pos_rank	| The number of seasonal rank standings for each player in their position. |
 | Year	| The year of the regular season for each player. |
 | ADP	| The number of the average draft position for each player in Fantasy Football. |
 | Adp_by_pos	| The number of the average draft position for each player in Fantasy Football where the player falls in their respecitive position. |
 | Success	| Whether each player was a sucessfull pick in the regular season based on actual season statistics. |
 | Round	| The number of the average draft position for each player in Fantasy Football Leagues by Team Owners. |


 ## Conclusions and Recommendations
 - Acquired 13 years of data including Average Draft Position.
 - Explored position by position and round by round for all 13 years.
 - Modeled player performance using RidgeRegressor and XGBRegressor.
 - Created Ultimate Draft Guidebook using the information discovered in exploration.
 - Created Cheatsheet of players predicted PPR points scored next season. 

 ## Next Steps
 - Draft Simulation for all 15 rounds with a 12 man league.
 - Test how fantasy teams do with players predicted points.
 - Test predictions against the upcoming season to determine how accurate our model is at predicting.
