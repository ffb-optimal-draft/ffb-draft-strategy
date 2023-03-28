#!/usr/bin/env python
# coding: utf-8

# In[7]:


import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

import warnings
warnings.filterwarnings('ignore')

pd.set_option('display.max_columns', 100)

from prepare import x_y_split, rmse
import xgboost

from sklearn.preprocessing import StandardScaler, RobustScaler


# In[3]:


def acquire_seasons():
    df = pd.read_csv('season.csv',index_col=0)

    seasons = range(2016,2023)
    positions = ['QB','RB','WR','TE']
    add = []
    
    df = df[df['g'] > 5]

    for s in seasons:
        for p in positions:
            test = df[(df.year==s)&(df.pos==p)].sort_values('avg_draft_pos_ppr').reset_index(drop=True)
            test['adp_by_pos'] = test.index+1

            test = test[(test.year==s)&(test.pos==p)].sort_values('ppr_pts',ascending=False).reset_index(drop=True)
            test['pos_rank'] = test.index+1

            add.append(test)

    df = pd.concat(add).reset_index(drop=True)

    for i in range(0,len(df.index)):
        if df.loc[i,'pos'] =='QB' or df.loc[i,'pos'] =='TE':

            if df.loc[i, 'adp_by_pos'] <= 3:
                if df.loc[i,'pos_rank'] <= 3:
                    df.loc[i,'success'] = 1
                else:
                    df.loc[i, 'success'] = 0
            else:
                if df.loc[i,'pos_rank']<=12:
                    df.loc[i,'success'] = 1
                else:
                    df.loc[i, 'success'] = 0

        else:
            if df.loc[i, 'adp_by_pos'] <= 6:
                if df.loc[i,'pos_rank'] <= 6:
                    df.loc[i,'success'] = 1
                else:
                    df.loc[i, 'success'] = 0

            elif df.loc[i, 'adp_by_pos'] > 36:
                if df.loc[i,'pos_rank'] < 36:
                    df.loc[i,'success'] = 1
                else:
                    df.loc[i, 'success'] = 0

            else:
                if df.loc[i,'pos_rank'] <= df.loc[i,'adp_by_pos']:
                    df.loc[i, 'success'] = 1
                else:
                    df.loc[i, 'success'] = 0


    for i in range(0,len(df.index)):
        if df.loc[i,'avg_draft_pos_ppr'] <= 12:
            df.loc[i, 'round'] = 1
        elif df.loc[i,'avg_draft_pos_ppr'] <= 24:
            df.loc[i, 'round'] = 2
        elif df.loc[i,'avg_draft_pos_ppr'] <= 36:
            df.loc[i, 'round'] = 3
        elif df.loc[i,'avg_draft_pos_ppr'] <= 48:
            df.loc[i, 'round'] = 4
        elif df.loc[i,'avg_draft_pos_ppr'] <= 60:
            df.loc[i, 'round'] = 5
        elif df.loc[i,'avg_draft_pos_ppr'] <= 72:
            df.loc[i, 'round'] = 6
        elif df.loc[i,'avg_draft_pos_ppr'] <= 84:
            df.loc[i, 'round'] = 7
        elif df.loc[i,'avg_draft_pos_ppr'] <= 96:
            df.loc[i, 'round'] = 8
        elif df.loc[i,'avg_draft_pos_ppr'] <= 108:
            df.loc[i, 'round'] = 9
        elif df.loc[i,'avg_draft_pos_ppr'] <= 120:
            df.loc[i, 'round'] = 10
        elif df.loc[i,'avg_draft_pos_ppr'] <= 132:
            df.loc[i, 'round'] = 11
        elif df.loc[i,'avg_draft_pos_ppr'] <= 144:
            df.loc[i, 'round'] = 12
        elif df.loc[i,'avg_draft_pos_ppr'] <= 156:
            df.loc[i, 'round'] = 13
        elif df.loc[i,'avg_draft_pos_ppr'] <= 168:
            df.loc[i, 'round'] = 14
        else:
            df.loc[i, 'round'] = 15
        
    return df

def find_pts_averages(player):
    pts_var_avg = player.pts_var.abs().mean()
    return pts_var_avg

def pass_rec_rush():
    
    passing = pd.read_csv('passing.csv', index_col=0)
    rec = pd.read_csv('receive.csv', index_col=0)
    rush = pd.read_csv('rush.csv', index_col=0)

    passing['date'] = pd.to_datetime(passing['date'])
    rush['date'] = pd.to_datetime(rush['date'])
    rec['date'] = pd.to_datetime(rec['date'])
    
    passing['year'] = passing['date'].dt.year
    rec['year'] = rec['date'].dt.year
    rush['year'] = rush['date'].dt.year
    
    return passing, rec, rush

# In[4]:


def add_target(group):
    group['target'] = group['ppr_pts'].shift(-1)
    group = group.fillna(0)
    return group


def var_avg_df(passing, rec, rush, df):
    
    s_2018 = df[df['year']>2017]
    
    weekly = pd.DataFrame(columns=['player', 'pts_var'])
    pts_var_avg = pd.DataFrame(columns=['player', 'pts_var', 'year'])

    for year in range(2022, 2017, -1):
        res = rec[rec['year']==year].groupby('player')['tgt'].mean()> 5
        res = res[res==True].index.to_list()

        rec_temp = rec[(rec['year']==year) & (rec['player'].isin(res))]
        weekly = weekly.append(rec_temp[['player'] + ['pts_var']])

        qbs = passing[passing['year']==year].groupby('player')['att'].mean()> 5
        qbs = qbs[qbs==True].index.to_list()

        pass_temp = passing[(passing['year']==year) & (passing['player'].isin(qbs))]
        weekly = weekly.append(pass_temp[['player'] + ['pts_var']])

        rbs = rush[rush['year']==year].groupby('player')['att'].mean()> 5
        rbs = rbs[rbs==True].index.to_list()

        rush_temp = rush[(rush['year']==year) & (rush['player'].isin(rbs))]
        weekly = weekly.append(rush_temp[['player'] + ['pts_var']])

        pts_avg = weekly.groupby('player', group_keys=False).apply(find_pts_averages)
        weekly_temp = pd.DataFrame(pts_avg, columns=['pts_var_avg'])

        weekly_temp['year'] = year

        pts_var_avg = pts_var_avg.append(weekly_temp)
    
    pts_var_avg = pts_var_avg.reset_index()
    
    pts_var_avg.drop(columns=['player','pts_var'], inplace=True)
    pts_var_avg.rename(columns={'index':'player'}, inplace=True)
    
    df1 = pd.merge(left=s_2018, right=pts_var_avg, on=['player','year'], how='left')
    df1.rename(columns={'pts_var':'pts_var_avg'}, inplace=True)
    
    return df1
# In[13]:


def split_pos(df):
    
    qb_df = df[df['pos']=='QB']
    rb_df = df[df['pos']=='RB']
    wr_df = df[df['pos']=='WR']
    te_df = df[df['pos']=='TE']
    
    qb_df = qb_df[(qb_df['player']!= 'Tom Brady') & 
                  (qb_df['player']!='Marcus Mariota')]
    
    wr_df = wr_df[wr_df['player']!= 'KaVontae Turpin']
    
    te_df = te_df[(te_df['player']!='Richard Rodgers') & 
                  (te_df['player']!='Feleipe Franks')]
    
    qb_df['rating'] = round(((((((qb_df['cmp']/qb_df['pass_att'])-.3)*5) + 
                      ((qb_df['pass_yds']/qb_df['pass_att']-3)*.25) +
                      ((qb_df['pass_tds']/qb_df['pass_att'])*20) +
                      (2.375-((qb_df['int']/qb_df['pass_att'])*25)))/6)*100),2)
    
    qb_df['comp%'] = round((qb_df['cmp'] / qb_df['pass_att']) * 100, 2)
    qb_df['int%'] = round((qb_df['int'] / qb_df['pass_att']) * 100, 2)
    
    qb_df.drop(columns=['rk','pos','tgt','rec','rec_yards','y/r','rec_tds','vbd', 'team',
                        'cmp','pass_att','int'], inplace=True)
    rb_df.drop(columns=['rk','team','pos','cmp','pass_att','pass_yds','pass_tds','int','vbd'], inplace=True)
    wr_df.drop(columns=['rk','team','pos','cmp','pass_tds','pass_att','pass_yds','int','rush_att','rush_yard','y/a','rush_tds',
                    'vbd'],inplace=True)
    te_df.drop(columns=['rk','team','pos','cmp','pass_att','pass_yds','pass_tds','int','rush_att','rush_yard','y/a',
                    'rush_tds','vbd'], inplace=True)
    
    qb_df = add_target(qb_df)
    rb_df = add_target(rb_df)
    wr_df = add_target(wr_df)
    te_df = add_target(te_df)    

                
    return qb_df, rb_df, wr_df, te_df


# In[6]:


def qb_xgb_modeling(df, cols):
    
    X_train = df[df['year']<2020]
    X_val = df[df['year']<2022]
    X_test = df[df['year']==2022]
    
    y_train = X_train['target']
    X_train.drop(columns=['target'], inplace = True)
    
    y_val = X_val['target']
    X_val.drop(columns=['target'], inplace = True)

    X_test.drop(columns=['target'], inplace = True)
    
    ss = StandardScaler()
    
    X_train[cols] = ss.fit_transform(X_train[cols])
    X_val[cols] = ss.transform(X_val[cols])
    X_test[cols] = ss.transform(X_test[cols])
    
    cols.append('success')
    
    xgb = xgboost.XGBRegressor(colsample_bytree=.85, gamma=10, eta=.2, max_depth=4,
                                  min_child_weight=5,n_estimators=250,subsample=.8)
    
    xgb.fit(X_train[cols], y_train, eval_set=[(X_train[cols], y_train), (X_val[cols], y_val)],
           early_stopping_rounds=25, verbose=False)
    xgb_preds = xgb.predict(X_train[cols])
    
    preds = pd.DataFrame({'actual':y_train,
                          'baseline':y_train.mean(),
                          'xgb_preds':xgb_preds})
    
    xgb_val_preds = xgb.predict(X_val[cols])
    
    val_preds = pd.DataFrame({'actual':y_val,
                              'baseline':y_train.mean(),
                              'xgb_val_preds':xgb_val_preds})
    
    
    pos_2023 = pd.DataFrame({'player':X_test['player'],
                             'preds':xgb.predict(X_test[cols])})
    
    val_2022 = X_val[X_val['year']==2021]
    pos_2022 = pd.DataFrame({'player':val_2022['player'],
                             'actual':val_2022['ppr_pts'],
                             'preds':xgb.predict(val_2022[cols])})
    
    return preds, val_preds


# In[14]:


def rb_xgb_modeling(df, cols):
    
    X_train = df[df['year']<2020]
    X_val = df[df['year']<2022]
    X_test = df[df['year']==2022]
    
    y_train = X_train['target']
    X_train.drop(columns=['target'], inplace = True)
    
    y_val = X_val['target']
    X_val.drop(columns=['target'], inplace = True)

    X_test.drop(columns=['target'], inplace = True)
    
    ss = StandardScaler()
    
    X_train[cols] = ss.fit_transform(X_train[cols])
    X_val[cols] = ss.transform(X_val[cols])
    X_test[cols] = ss.transform(X_test[cols])
    
    cols.append('success')
    
    xgb = xgboost.XGBRegressor(colsample_bytree=.7, gamma=1, eta=.03, max_depth=6,
                                min_child_weight=1,n_estimators=250,subsample=.7)
    
    xgb.fit(X_train[cols], y_train, eval_set=[(X_train[cols], y_train), (X_val[cols], y_val)],
           early_stopping_rounds=25, verbose=False)
    xgb_preds = xgb.predict(X_train[cols])
    
    preds = pd.DataFrame({'actual':y_train,
                          'baseline':y_train.mean(),
                          'xgb_preds':xgb_preds})
    
    xgb_val_preds = xgb.predict(X_val[cols])
    
    val_preds = pd.DataFrame({'actual':y_val,
                              'baseline':y_train.mean(),
                              'xgb_val_preds':xgb_val_preds})
    
    
    pos_2023 = pd.DataFrame({'player':X_test['player'],
                             'preds':xgb.predict(X_test[cols])})
    
    val_2022 = X_val[X_val['year']==2021]
    pos_2022 = pd.DataFrame({'player':val_2022['player'],
                             'actual':val_2022['ppr_pts'],
                             'preds':xgb.predict(val_2022[cols])})
    
    return preds, val_preds


# In[15]:


def wrte_xgb_modeling(df, cols):
    
    X_train = df[df['year']<2020]
    X_val = df[df['year']<2022]
    X_test = df[df['year']==2022]
    
    y_train = X_train['target']
    X_train.drop(columns=['target'], inplace = True)
    
    y_val = X_val['target']
    X_val.drop(columns=['target'], inplace = True)

    X_test.drop(columns=['target'], inplace = True)
    
    ss = StandardScaler()
    
    X_train[cols] = ss.fit_transform(X_train[cols])
    X_val[cols] = ss.transform(X_val[cols])
    X_test[cols] = ss.transform(X_test[cols])
    
    cols.append('success')
    
    xgb = xgboost.XGBRegressor(colsample_bytree=.85, gamma=0, eta=.07, max_depth=5,
                               min_child_weight=1,n_estimators=500,subsample=.85)
    
    xgb.fit(X_train[cols], y_train, eval_set=[(X_train[cols], y_train), (X_val[cols], y_val)],
           early_stopping_rounds=5, verbose=False)
    xgb_preds = xgb.predict(X_train[cols])
    
    preds = pd.DataFrame({'actual':y_train,
                          'baseline':y_train.mean(),
                          'xgb_preds':xgb_preds})
    
    xgb_val_preds = xgb.predict(X_val[cols])
    
    val_preds = pd.DataFrame({'actual':y_val,
                              'baseline':y_train.mean(),
                              'xgb_val_preds':xgb_val_preds})
    
    
    pos_2023 = pd.DataFrame({'player':X_test['player'],
                             'preds':xgb.predict(X_test[cols])})
    
    val_2022 = X_val[X_val['year']==2021]
    pos_2022 = pd.DataFrame({'player':val_2022['player'],
                             'actual':val_2022['ppr_pts'],
                             'preds':xgb.predict(val_2022[cols])})
    
    return preds, val_preds


# In[10]:
def round_preds(train_rmse, val_rmse):
    train_rmse = round(train_rmse, 2)
    val_rmse = round(val_rmse, 2)
    
    return train_rmse, val_rmse

def avg_rmse(qb_rmse, qb_val_rmse, rb_rmse, rb_val_rmse, wr_rmse, wr_val_rmse, te_rmse, te_val_rmse):
    
    train_avg = round(((qb_rmse + rb_rmse + wr_rmse + te_rmse)/4),2)
    val_avg = round(((qb_val_rmse + rb_val_rmse + wr_val_rmse + te_val_rmse)/4),2)
    
    print(f'The average RMSE for the train dataset is: {train_avg}')
    print(f'The average RMSE for the validate dataset is: {val_avg}')

qb_cols = ['age','g','gs','pass_yds','pass_tds','rush_att','rush_yard','y/a','rush_tds',
           'fmb','fl','rush_rec_tds','pos_rank','avg_draft_pos','avg_draft_pos_ppr','adp_by_pos','round','ppr_pts',
           'comp%', 'int%','rating','pts_var_avg']


# In[9]:


rb_cols = ['age','g','gs','rush_att','rush_yard','y/a','rush_tds','tgt','rec','rec_yards','y/r','rec_tds','fmb',
           'fl','rush_rec_tds','ppr_pts','pos_rank','avg_draft_pos','avg_draft_pos_ppr','adp_by_pos','round','pts_var_avg']


# In[11]:


wr_cols = ['age','g','gs','tgt','rec','rec_yards','y/r','rec_tds','fmb','fl','rush_rec_tds','ppr_pts','pos_rank',
           'avg_draft_pos','avg_draft_pos_ppr','adp_by_pos','round','pts_var_avg']


# In[12]:


te_cols = ['age','g','gs','tgt','rec','rec_yards','y/r','rec_tds','fmb','fl','rush_rec_tds','ppr_pts','pos_rank',
           'avg_draft_pos','avg_draft_pos_ppr','adp_by_pos','round','pts_var_avg']


# # Adding functions to explore 

# In[1]:


def plot_explore_distribution(explore_df):
    """
    This function takes a dataframe and plots the distribution of the highest performing players for each position (QB, RB, WR, TE) 
    based on their PPR points. The function also prints the mean and median PPR points for each position.
    
    Parameters:
    -----------
    explore_df : pandas DataFrame
        The DataFrame containing data for player performances
    
    Returns:
    --------
    None
    """
    
    # Lambda function to round numbers to 2 decimal places
    r2 = lambda x: round(x,2)

    # List of years to loop through
    years = [2016, 2017, 2018, 2019, 2020, 2021, 2022]

    # DataFrames for each position
    qb_df = pd.DataFrame()
    rb_df = pd.DataFrame()
    wr_df = pd.DataFrame()
    te_df = pd.DataFrame()

    # Loop through years to find the highest performing player for each position
    for year in years:
        # Quarterbacks
        qb_year_df = explore_df.loc[(explore_df["pos"] == "QB") & (explore_df['year']== year), ['player', 'pos', 'ppr_pts']]
        qb_top_player = qb_year_df.iloc[0]
        qb_df = pd.concat([qb_df, qb_top_player.to_frame().T])

        # Running backs
        rb_year_df = explore_df.loc[(explore_df["pos"] == "RB") & (explore_df['year']== year), ['player', 'pos', 'ppr_pts']]
        rb_top_player = rb_year_df.iloc[0]
        rb_df = pd.concat([rb_df, rb_top_player.to_frame().T])

        # Wide receivers
        wr_year_df = explore_df.loc[(explore_df["pos"] == "WR") & (explore_df['year']== year), ['player', 'pos', 'ppr_pts']]
        wr_top_player = wr_year_df.iloc[0]
        wr_df = pd.concat([wr_df, wr_top_player.to_frame().T])

        # Tight ends
        te_year_df = explore_df.loc[(explore_df["pos"] == "TE") & (explore_df['year']== year), ['player', 'pos', 'ppr_pts']]
        te_top_player = te_year_df.iloc[0]
        te_df = pd.concat([te_df, te_top_player.to_frame().T])

    # Plot distributions for Quarter Backs
    fig, ax = plt.subplots(figsize=(8,6))
    sns.histplot(x=qb_df['ppr_pts'])
    ax2 = ax.twinx()
    sns.kdeplot(x=qb_df['ppr_pts'],color="purple",ax=ax2)
    plt.title("Distribution of Highest Performing QB PPR points")
    plt.xlabel("PPR Points")
    plt.show()
    print(
            "Mean QB PPR Points: {}".format(r2(np.mean(qb_df['ppr_pts'])))
          + "\n"
          + "Median QB PPR Points: {}".format(r2(np.median(qb_df['ppr_pts'])))
          + "\n"
        )
    # Plot distributions for Running Backs
    fig, ax = plt.subplots(figsize=(8,6))
    sns.histplot(x=rb_df['ppr_pts'])
    ax2 = ax.twinx()
    sns.kdeplot(x=rb_df['ppr_pts'],color="purple",ax=ax2)
    plt.title("Distribution of Highest Performing RB PPR points")
    plt.xlabel("PPR Points")
    plt.show()
    # Print mean and median PPR points for Runningbacks
    print(
        "Mean RB PPR Points: {}".format(r2(np.mean(rb_df['ppr_pts'])))
        + "\n"
        + "Median RB PPR Pointss: {}".format(r2(np.median(rb_df['ppr_pts'])))
    )
    # Plot distributions for Wide Recievers
    fig, ax = plt.subplots(figsize=(8,6))
    sns.histplot(x=wr_df['ppr_pts'])
    ax2 = ax.twinx()
    sns.kdeplot(x=wr_df['ppr_pts'],color="purple",ax=ax2)
    plt.title("Distribution of Highest Performing WR PPR points")
    plt.xlabel("PPR Points")
    plt.show()
    # Print mean and median PPR points for Wide Recievers
    print(
        "Mean WR PPR Points: {}".format(r2(np.mean(wr_df['ppr_pts'])))
        + "\n"
        + "Median WR PPR Points: {}".format(r2(np.median(wr_df['ppr_pts']))))

    # Plot distributions for Tight Ends
    fig, ax = plt.subplots(figsize=(8,6))
    sns.histplot(x=te_df['ppr_pts'])
    ax2 = ax.twinx()
    sns.kdeplot(x=te_df['ppr_pts'],color="purple",ax=ax2)
    plt.title("Distribution of Highest Performing WR PPR points")
    plt.xlabel("PPR Points")
    plt.show()
    # Print mean and median PPR points for Tight Ends
    print(
        "Mean TE PPR Points: {}".format(r2(np.mean(te_df['ppr_pts'])))
        + "\n"
        + "Median TE PPR Points: {}".format(r2(np.median(te_df['ppr_pts']))))


# In[2]:


def ranking_df(explore_df, year):
    """
    Takes in a DataFrame and a year, and returns separate DataFrames for each position containing players' ranking, name,
    and position for that year.

    Parameters:
    explore_df (pd.DataFrame): A DataFrame containing player data including name, position, and ranking.
    year (int): The year for which the rankings should be returned.

    Returns:
    tuple: Four DataFrames containing players' ranking, name, and position for the positions of QB, RB, WR, and TE.
    """

    # Get QB DataFrame
    qb = explore_df.loc[(explore_df['pos'] =='QB') & (explore_df["year"] == year)].sort_values('rk')['player']
    qb_pos = explore_df.loc[(explore_df['pos'] =='QB') & (explore_df["year"] == year)].sort_values('rk')['pos']
    rank = explore_df.loc[(explore_df['pos'] =='QB') & (explore_df["year"] == year)].sort_values('rk')['rk']
    qb_df = pd.DataFrame({"Rank": rank, "Quarterbacks": qb, "Position": qb_pos}, index=None)
    qb_df.reset_index(drop=True, inplace=True)

    # Get RB DataFrame
    rb = explore_df.loc[(explore_df['pos'] =='RB') & (explore_df["year"] == year)].sort_values('rk')['player']
    rb_pos = explore_df.loc[(explore_df['pos'] =='RB') & (explore_df["year"] == year)].sort_values('rk')['pos']
    rank = explore_df.loc[(explore_df['pos'] =='RB') & (explore_df["year"] == year)].sort_values('rk')['rk']
    rb_df = pd.DataFrame({"Rank": rank, "Runningbacks": rb, "Position": rb_pos}, index=None)
    rb_df.reset_index(drop=True, inplace=True)

    # Get WR DataFrame
    wr = explore_df.loc[(explore_df['pos'] =='WR') & (explore_df["year"] == year)].sort_values('rk')['player']
    wr_pos = explore_df.loc[(explore_df['pos'] =='WR') & (explore_df["year"] == year)].sort_values('rk')['pos']
    rank = explore_df.loc[(explore_df['pos'] =='WR') & (explore_df["year"] == year)].sort_values('rk')['rk']
    wr_df = pd.DataFrame({"Rank": rank, "WideReceiver": wr, "Position": wr_pos}, index=None)
    wr_df.reset_index(drop=True, inplace=True)

    # Get TE DataFrame
    te = explore_df.loc[(explore_df['pos'] =='TE') & (explore_df["year"] == year)].sort_values('rk')['player']
    te_pos = explore_df.loc[(explore_df['pos'] =='TE') & (explore_df["year"] == year)].sort_values('rk')['pos']
    rank = explore_df.loc[(explore_df['pos'] =='TE') & (explore_df["year"] == year)].sort_values('rk')['rk']
    te_df = pd.DataFrame({"Rank": rank, "TightEnd": te, "Position": te_pos}, index=None)
    te_df.reset_index(drop=True, inplace=True)

    return qb_df, rb_df, wr_df, te_df


# In[17]:


def pred_comb(pos):
    df = pd.read_csv(f'{pos}_pred_comb.csv',index_col=0)
    df = df[['player','prediction','preds']]
    df = df.fillna(method = 'bfill',axis=1).fillna(method = 'ffill',axis=1)
    df['combined'] = (df.prediction+df.preds)/2
    df = df.sort_values('combined',ascending=False).reset_index(drop=True)
    df.index+=1
    df['prediction'] = df['prediction'].astype(int)
    df['preds'] = df['preds'].astype(int)
    df['combined'] = df['combined'].astype(int)
    if (pos == 'qb') or (pos == 'te'):    
        df = df.head(25)
    elif (pos == 'wr') or (pos == 'rb'):
        df = df.head(50)
    return df


# In[ ]:




