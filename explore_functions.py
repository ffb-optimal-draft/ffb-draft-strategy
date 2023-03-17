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
    df[df.pos=='TE'].groupby('adp_by_pos')['ppr_pts'].mean()[0:36].plot(ax=axs[1, 1], color='gold')
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



#---------------------------------------------------------------


#---------------------------------------------------------------


#---------------------------------------------------------------



#---------------------------------------------------------------



#---------------------------------------------------------------

