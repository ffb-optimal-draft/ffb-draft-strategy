import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

#---------------------------------------------------------------

def avg_pts_by_adp(df):
    fig, axs = plt.subplots(2, 2, figsize=(10, 10))
    fig.suptitle('Comparison of Average PPR Points by Position and Position Rank', fontsize=16, fontweight='bold', y=0.92)

    axs[0, 0].set_title('Quarterbacks (QB)')
    df[df.pos=='QB'].groupby('adp_by_pos')['ppr_pts'].mean()[0:24].plot(ax=axs[0, 0], color='royalblue')
    axs[0, 0].set_xlabel('Avg Draft Position')
    axs[0, 0].set_ylabel('Average Points Per Season')
    axs[0, 0].tick_params(axis='x', labelsize=8)

    axs[0, 1].set_title('Running Backs (RB)')
    df[df.pos=='RB'].groupby('adp_by_pos')['ppr_pts'].mean()[0:75].plot(ax=axs[0, 1], color='seagreen')
    axs[0, 1].set_xlabel('Avg Draft Position')
    axs[0, 1].set_ylabel('Average Points Per Season')
    axs[0, 1].tick_params(axis='x', labelsize=8)

    axs[1, 0].set_title('Wide Receivers (WR)')
    df[df.pos=='WR'].groupby('adp_by_pos')['ppr_pts'].mean()[0:75].plot(ax=axs[1, 0], color='orangered')
    axs[1, 0].set_xlabel('Avg Draft Position')
    axs[1, 0].set_ylabel('Average Points Per Season')
    axs[1, 0].tick_params(axis='x', labelsize=8)

    axs[1, 1].set_title('Tight Ends (TE)')
    df[df.pos=='TE'].groupby('adp_by_pos')['ppr_pts'].mean()[0:24].plot(ax=axs[1, 1], color='gold')
    axs[1, 1].set_xlabel('Avg Draft Position')
    axs[1, 1].set_ylabel('Average Points Per Season')
    axs[1, 1].tick_params(axis='x', labelsize=8)

    plt.show()

#---------------------------------------------------------------

def pnt_avg_by_pos_rank(df):
    fig, axs = plt.subplots(2, 2, figsize=(10, 10))
    fig.suptitle('Comparison of Average PPR Points by Position and Position Rank', fontsize=16, fontweight='bold', y=0.92)

    axs[0, 0].set_title('Quarterbacks (QB)')
    df[df.pos=='QB'].groupby('pos_rank')['ppr_pts'].mean()[0:24].plot(ax=axs[0, 0], color='royalblue')
    axs[0, 0].set_xlabel('Position Rank')
    axs[0, 0].set_ylabel('Average Points Per Season')
    axs[0, 0].tick_params(axis='x', labelsize=8)

    axs[0, 1].set_title('Running Backs (RB)')
    df[df.pos=='RB'].groupby('pos_rank')['ppr_pts'].mean()[0:75].plot(ax=axs[0, 1], color='seagreen')
    axs[0, 1].set_xlabel('Position Rank')
    axs[0, 1].set_ylabel('Average Points Per Season')
    axs[0, 1].tick_params(axis='x', labelsize=8)

    axs[1, 0].set_title('Wide Receivers (WR)')
    df[df.pos=='WR'].groupby('pos_rank')['ppr_pts'].mean()[0:75].plot(ax=axs[1, 0], color='orangered')
    axs[1, 0].set_xlabel('Position Rank')
    axs[1, 0].set_ylabel('Average Points Per Season')
    axs[1, 0].tick_params(axis='x', labelsize=8)

    axs[1, 1].set_title('Tight Ends (TE)')
    df[df.pos=='TE'].groupby('pos_rank')['ppr_pts'].mean()[0:36].plot(ax=axs[1, 1], color='gold')
    axs[1, 1].set_xlabel('Position Rank')
    axs[1, 1].set_ylabel('Average Points Per Season')
    axs[1, 1].tick_params(axis='x', labelsize=8)

    plt.show()

#---------------------------------------------------------------

def success_rate_by_season(df):
    df_by_season = pd.DataFrame(df.groupby('season').mean()[['qb','rb','wr','te']])
    plt.plot(df_by_season, linewidth = 4)
    plt.legend(['qb', 'rb', 'wr', 'te'])
    plt.title('Average Success Rate of Picks by Season', fontdict = { 'fontsize': 15})
    plt.ylabel('Successful Pick Rate', fontdict = { 'fontsize': 15})
    plt.xlabel('Seasons', fontdict = { 'fontsize': 15})
    plt.show()

#---------------------------------------------------------------

def qb_draft_strat(df):
    fig, axs = plt.subplots(1, 2, figsize=(10, 4))
    fig.suptitle('Quarterback Draft Strategy', fontsize=24, fontweight='bold', y=1.1)

    df[df.pos=='QB'].groupby('pos_rank')['ppr_pts'].mean()[0:24].plot(ax=axs[0], color='black').grid(axis = 'y')
    axs[0].set_xlabel('Avg Position Rank')
    axs[0].set_ylabel('Points Scored')
    axs[0].axvline(x=6, color = 'red', linestyle = '--')
    axs[0].axvline(x=12, color = 'green', linestyle = ':')
    axs[0].tick_params(axis='x', labelsize=10)
    
    df[df.pos=='QB'].groupby('round')[['ppr_pts']].mean().plot(ax=axs[1], color='black').grid(axis = 'y')
    axs[1].set_xlabel('Round Drafted')
    axs[1].set_ylabel(' ')
    axs[1].axvline(x=5, color = 'red', linestyle = '--')
    axs[1].axvline(x=6, color = 'red', linestyle = '--')
    axs[1].axvline(x=8, color = 'green', linestyle = 'dashdot')
    axs[1].axvline(x=10, color = 'green', linestyle = 'dashdot')
    axs[1].tick_params(axis='x', labelsize=10)
    plt.show()

#---------------------------------------------------------------

def rb_draft_strat(df):
    fig, axs = plt.subplots(1, 2, figsize=(10, 4))
    fig.suptitle('Running Back Draft Strategy', fontsize=24, fontweight='bold', y=1.1)

    df[df.pos=='RB'].groupby('pos_rank')['ppr_pts'].mean()[0:45].plot(ax=axs[0], color='black').grid(axis = 'y')
    axs[0].set_xlabel('Avg Position Rank')
    axs[0].set_ylabel('Points Scored')
    axs[0].axvline(x=7, color = 'red', linestyle = '--')
    axs[0].axvline(x=30, color = 'green', linestyle = ':')
    axs[0].tick_params(axis='x', labelsize=10)
    
    df[df.pos=='RB'].groupby('round')[['ppr_pts']].mean().plot(ax=axs[1], color='black').grid(axis = 'y')
    axs[1].set_xlabel('Round Drafted')
    axs[1].set_ylabel(' ')
    axs[1].axvline(x=1, color = 'red', linestyle = '--')
    axs[1].axvline(x=4, color = 'red', linestyle = '--')
    axs[1].axvline(x=11, color = 'green', linestyle = 'dashdot')
    axs[1].axvline(x=14, color = 'green', linestyle = 'dashdot')
    axs[1].tick_params(axis='x', labelsize=10)
    plt.show()

#---------------------------------------------------------------

def wr_draft_strat(df):
    fig, axs = plt.subplots(1, 2, figsize=(10, 4))
    fig.suptitle('Wide Receiver Draft Strategy', fontsize=24, fontweight='bold', y=1.1)

    df[df.pos=='WR'].groupby('pos_rank')['ppr_pts'].mean()[0:45].plot(ax=axs[0], color='black').grid(axis = 'y')
    axs[0].set_xlabel('Avg Position Rank')
    axs[0].set_ylabel('Points Scored')
    axs[0].axvline(x=10, color = 'red', linestyle = '--')
    axs[0].axvline(x=30, color = 'green', linestyle = ':')
    axs[0].tick_params(axis='x', labelsize=10)
    
    df[df.pos=='WR'].groupby('round')[['ppr_pts']].mean().plot(ax=axs[1], color='black').grid(axis = 'y')
    axs[1].set_xlabel('Round Drafted')
    axs[1].set_ylabel(' ')
    axs[1].axvline(x=1, color = 'red', linestyle = '--')
    axs[1].axvline(x=6, color = 'red', linestyle = '--')
    axs[1].axvline(x=7, color = 'green', linestyle = 'dashdot')
    axs[1].axvline(x=13, color = 'green', linestyle = 'dashdot')
    axs[1].tick_params(axis='x', labelsize=10)

#---------------------------------------------------------------

def te_draft_strat(df):
    fig, axs = plt.subplots(1, 2, figsize=(10, 4))
    fig.suptitle('Tight Ends Draft Strategy', fontsize=24, fontweight='bold', y=1.1)

    df[df.pos=='TE'].groupby('pos_rank')['ppr_pts'].mean()[0:24].plot(ax=axs[0], color='black').grid(axis = 'y')
    axs[0].set_xlabel('Avg Position Rank')
    axs[0].set_ylabel('Points Scored')
    axs[0].axvline(x=3, color = 'red', linestyle = '--')
    axs[0].axvline(x=12, color = 'green', linestyle = ':')
    axs[0].tick_params(axis='x', labelsize=10)
    
    df[df.pos=='TE'].groupby('round')[['ppr_pts']].mean().plot(ax=axs[1], color='black').grid(axis = 'y')
    axs[1].set_xlabel('Round Drafted')
    axs[1].set_ylabel(' ')
    axs[1].axvline(x=2, color = 'red', linestyle = '--')
    axs[1].axvline(x=4, color = 'red', linestyle = '--')
    axs[1].axvline(x=10, color = 'green', linestyle = 'dashdot')
    axs[1].axvline(x=14, color = 'green', linestyle = 'dashdot')
    axs[1].tick_params(axis='x', labelsize=10)

#---------------------------------------------------------------

def rb_wr_comparison(df):
    seasons = range(2010,2023)
    new = []

    for s in seasons:
        num = [3,6,12,18,24,30,36,42,48,54,60,66,72]
        wr = []
        rb = []
        li = [rb,wr]
        pos = ['RB','WR']

        for n in range(3,72,3):
            d = df[(df.year==s) & ((df.pos=='WR') | (df.pos== 'RB'))][['pos','ppr_pts']].\
            sort_values(by='ppr_pts',ascending=False).head(n).groupby('pos').count()/n
        
            for p,l in zip(pos,li):
                try:
                    l.append(d.loc[p,'ppr_pts'])
                except:
                    l.append(0)
                
        new.append(pd.DataFrame({'n':range(3,72,3),'rb':rb,'wr':wr}))

    plt.figure(figsize=(18,9))
    prop_rb_wr = pd.concat(new)
    prop_rb_wr.groupby('n').mean().plot()
    plt.title('Do RBs or WRs Score More Points?')
    plt.ylabel('Percent')
    plt.xlabel('Top "n" Players')
    plt.savefig('rb_wr_pts.png')
    plt.show()

#---------------------------------------------------------------

def top_positions(df):
    seasons = range(2010,2023)
    new = []

    for s in seasons:
        num = [3,6,12,18,24,30,36,42,48,54,60,66,72]
        qb = []
        wr = []
        rb = []
        te = []
        li = [qb,rb,wr,te]
        pos = ['QB','RB','WR','TE']

        for n in range(3,72,3):
            d = df[df.year==s][['pos','ppr_pts']].sort_values(by='ppr_pts',ascending=False).head(n).\
            groupby('pos').count()/n
            for p,l in zip(pos,li):
                try:
                    l.append(d.loc[p,'ppr_pts'])
                except:
                    l.append(0)
                
        new.append(pd.DataFrame({'n':range(3,72,3),'qb':qb,'rb':rb,'wr':wr,'te':te}))

    prop_all_positions = pd.concat(new)
    prop_all_positions.groupby('n').mean().plot()
    plt.title('What Position Scores the Most Points?')
    plt.ylabel('Percent')
    plt.xlabel('Top "n" Players')
    plt.savefig('top_n_players.png')
    plt.show()