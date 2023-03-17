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


# In[4]:


def add_target(group):
    group['target'] = group['ppr_pts'].shift(-1)
    group = group.fillna(0)
    return group


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
    
    qb_df.drop(columns=['rk','pos','tgt','rec','rec_yards','y/r','rec_tds','standard_pts','vbd', 'team',
                        'cmp','pass_att','int'], inplace=True)
    rb_df.drop(columns=['rk','team','pos','cmp','pass_att','pass_yds','pass_tds','int','standard_pts','vbd'], inplace=True)
    wr_df.drop(columns=['rk','team','pos','cmp','pass_tds','pass_att','pass_yds','int','rush_att','rush_yard','y/a','rush_tds',
                    'standard_pts','vbd'],inplace=True)
    te_df.drop(columns=['rk','team','pos','cmp','pass_att','pass_yds','pass_tds','int','rush_att','rush_yard','y/a',
                    'rush_tds','standard_pts','vbd'], inplace=True)
    
    qb_df = add_target(qb_df)
    rb_df = add_target(rb_df)
    wr_df = add_target(wr_df)
    te_df = add_target(te_df)    
    
    qb_inverse_cols = ['int','fmb','fl','pos_rank','avg_draft_pos','avg_draft_pos_ppr','adp_by_pos','round']
    for col in qb_inverse_cols:
        qb_df[col] *= -1
        
    rb_inverse_cols = ['fmb','fl','pos_rank','avg_draft_pos','avg_draft_pos_ppr','adp_by_pos','round']
    for col in rb_inverse_cols:
        rb_df[col] *= -1
        
    wr_inverse_cols = ['fmb','fl','pos_rank','avg_draft_pos','avg_draft_pos_ppr','adp_by_pos','round']
    for col in wr_inverse_cols:
        wr_df[col] *= -1
        
    te_inverse_cols = ['fmb','fl','pos_rank','avg_draft_pos','avg_draft_pos_ppr','adp_by_pos','round']
    for col in te_inverse_cols:
        te_df[col] *= -1
                
    return qb_df, rb_df, wr_df, te_df


# In[6]:


def xgb_modeling(df, cols):
    
    X_train = df[df['year']<2021]
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
    
    xgb = xgboost.XGBRegressor(eval_metric='rmse',n_estimators=100, max_depth=4, 
                               subsample=.65, colsample_bytree=.8, seed=42,
                               eta=.15, gamma=100)
    
    xgb.fit(X_train[cols], y_train)
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
    
    return preds, val_preds, pos_2023, pos_2022


# In[10]:


qb_cols = ['age','g','gs','pass_yds','pass_tds','rush_att','rush_yard','y/a','rush_tds',
           'fmb','fl','rush_rec_tds','pos_rank','avg_draft_pos','avg_draft_pos_ppr','adp_by_pos','round','ppr_pts',
           'comp%', 'int%','rating']


# In[9]:


rb_cols = ['age','g','gs','rush_att','rush_yard','y/a','rush_tds','tgt','rec','rec_yards','y/r','rec_tds','fmb',
           'fl','rush_rec_tds','ppr_pts','pos_rank','avg_draft_pos','avg_draft_pos_ppr','adp_by_pos','round']


# In[11]:


wr_cols = ['age','g','gs','tgt','rec','rec_yards','y/r','rec_tds','fmb','fl','rush_rec_tds','ppr_pts','pos_rank',
           'avg_draft_pos','avg_draft_pos_ppr','adp_by_pos','round']


# In[12]:


te_cols = ['age','g','gs','tgt','rec','rec_yards','y/r','rec_tds','fmb','fl','rush_rec_tds','ppr_pts','pos_rank',
           'avg_draft_pos','avg_draft_pos_ppr','adp_by_pos','round']


# In[ ]:




