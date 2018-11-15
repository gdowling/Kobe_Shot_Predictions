# -*- coding: utf-8 -*-
"""
Created on Tue Oct 16 21:27:03 2018

@author: George
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import math
from sklearn.cross_validation import KFold
from sklearn.feature_extraction.text import CountVectorizer
import xgboost
from sklearn.preprocessing import LabelEncoder
import sklearn.metrics as metrics

kobe_data = pd.read_csv("C:/Users/George/Documents/Kaggle/Kobe Shots/data.csv")

train = kobe_data[kobe_data["shot_made_flag"].notnull()].reset_index()
test = kobe_data[kobe_data["shot_made_flag"].isnull()].reset_index()

#Plots 

# Made/Missed by Location
plt.figure(figsize=(12,12))
plt.subplot(121)
plt.scatter(train.loc[train.shot_made_flag==1, 'loc_x'],train.loc[train.shot_made_flag==1, 'loc_y'], alpha = .05, color = 'green')
plt.ylim(-100, 900)
plt.title('Kobe Made Shots')
plt.subplot(122)
plt.scatter(train.loc[train.shot_made_flag==0, 'loc_x'],train.loc[train.shot_made_flag==0, 'loc_y'], alpha = .05, color = 'red')
plt.ylim(-100, 900)
plt.title('Kobe Missed Shots')
plt.show()

#Shot percentage by time
time_left_avg = train.groupby('minutes_remaining')['shot_made_flag'].mean().reset_index()
plt.plot(time_left_avg.minutes_remaining, time_left_avg.shot_made_flag)
plt.xlabel('Time Left in Period')
plt.ylabel('Shot Avg.')
plt.title('Avg by Time Left')
plt.show()

seconds_left_avg = train.groupby('seconds_remaining')['shot_made_flag'].mean().reset_index()
plt.plot(seconds_left_avg.seconds_remaining, seconds_left_avg.shot_made_flag)
plt.xlabel('Seconds Left in Period')
plt.ylabel('Shot Avg.')
plt.title('Avg by Seconds Left')
plt.show()

#Shot percentage by distance 
distance_avg = train.groupby('shot_distance')['shot_made_flag'].mean().reset_index()
plt.plot(distance_avg.shot_distance, distance_avg.shot_made_flag)
plt.xlabel('Shot Distance')
plt.ylabel('Shot Avg.')
plt.title('Avg. by Distance')
plt.show()

#Plotting shots by zone

#Random Color Generator helper function
def color_generator(num_colors):
    colors = []
    for i in range(num_colors):
        colors.append((np.random.rand(), np.random.rand(), np.random.rand()))
    return colors

colors = color_generator(100)

def plot_zones(zone):
    zone_mean = train.groupby(zone)['shot_made_flag'].mean()
    plt.figure(figsize=(15,15))
    for i, area in enumerate(train[zone].unique()):
        plt.subplot(121)
        zone_area = train.loc[(train[zone]==area)]
        plt.scatter(zone_area.loc_x, zone_area.loc_y, alpha = 0.5, color = colors[i])
        plt.text(zone_area.loc_x.mean(),zone_area.loc_y.quantile(.8),'%0.3f'%(zone_mean[area]), size=15, bbox=dict(facecolor='red', alpha=0.5))
        plt.ylim(-100, 900)
    plt.legend(train[zone].unique())
    plt.title(zone)
    plt.show()

#Shot Percentage by Period
Period_avg = train.groupby('period')['shot_made_flag'].mean().reset_index()
plt.plot(Period_avg.period,Period_avg.shot_made_flag)
plt.xlabel('Period')
plt.ylabel('Average')
plt.title('Shot Average by Period')
plt.show()

#Shot Percentage by Season
sns.barplot('season', 'shot_made_flag', data = train)
plt.xticks(rotation='vertical')
plt.title('Shot Average by Season')
plt.show()

#Shot Percentage by Shot_Types
sns.barplot('shot_type', 'shot_made_flag', data = train)
plt.title('Shot Average by Shot Type')
plt.show()

sns.barplot('combined_shot_type', 'shot_made_flag', data = train)
plt.title('Shot Average by Shot Type')
plt.show()

plt.figure(figsize=(15,6))
sns.barplot('action_type', 'shot_made_flag', data=train)
plt.xticks(rotation='vertical')
plt.show()

#Prepping Data
kobe_data_2 = pd.read_csv("C:/Users/George/Documents/Kaggle/Kobe Shots/data.csv")

#Shot Angle
kobe_data_2['shot_angle'] = kobe_data_2.apply(lambda row: 90 if row['loc_y']==0 else math.degrees(math.atan(row['loc_x']/abs(row['loc_y']))), axis=1)

kobe_data_2['shot_angle_bin'] = pd.cut(kobe_data_2.shot_angle, bins = 7, labels = range(7))
kobe_data_2['shot_angle_bin'] = kobe_data_2.shot_angle_bin.astype(int)

#Home Away
kobe_data_2['Home_Away'] = kobe_data_2.matchup.apply(lambda x: 0 if (x.split(' ')[1])=='@' else 1)

#Shot Type Prepping
kobe_data_2['action_type'] = kobe_data_2.action_type.apply(lambda x: x.replace('-', ''))
kobe_data_2['action_type'] = kobe_data_2.action_type.apply(lambda x: x.replace('Follow Up', 'followup'))
kobe_data_2['action_type'] = kobe_data_2.action_type.apply(lambda x: x.replace('Finger Roll', 'fingerroll'))

cv = CountVectorizer(max_features = 50, stop_words=['shot'])

shot_features = cv.fit_transform(kobe_data_2['action_type']).toarray()
shot_features = pd.DataFrame(shot_features, columns=cv.get_feature_names())

kobe_data_2 = pd.concat([kobe_data_2, shot_features], axis = 1)

#Time Features
kobe_data_2['game_date'] = pd.to_datetime(kobe_data_2.game_date)

kobe_data_2['game_date_month'] = kobe_data_2.game_date.dt.month

kobe_data_2['game_date_quarter'] = kobe_data_2.game_date.dt.quarter

kobe_data_2['time_remaining'] = kobe_data_2.apply(lambda row: row['minutes_remaining']*60+row['seconds_remaining'], axis=1)

kobe_data_2['timeUnder5'] = kobe_data_2.time_remaining.apply(lambda x: 1 if x<5 else 0)

#Shot Distance 
kobe_data_2['distance_bin'] = pd.cut(kobe_data_2.shot_distance, bins=10, labels=range(10))

angle_distance = kobe_data_2.groupby(['shot_angle_bin', 'distance_bin'])['shot_made_flag'].agg([np.mean],as_index= False).reset_index()
angle_distance['group'] = range(len(angle_distance))
angle_distance.drop('mean', inplace=True, axis=1)
kobe_data_2 = kobe_data_2.merge(angle_distance, 'left', ['shot_angle_bin', 'distance_bin'])

#Dropping Irrlevent data
kobe_data_2.columns.drop(['game_event_id'
                              , 'shot_made_flag'
                              , 'game_id' 
                              , 'shot_id' 
                              , 'game_date'
                              , 'minutes_remaining'
                              , 'seconds_remaining'
                              ,'lat', 'lon' 
                              , 'playoffs' 
                              , 'team_id', 'team_name'
                              , 'matchup'
                             ])

predictors = ['action_type', 'combined_shot_type', 'shot_distance',
       'shot_zone_basic', 'opponent',
       'bank', 'driving', 'dunk', 'fadeaway', 'jump', 'pullup', 'running',
       'slam', 'turnaround','timeUnder5', 'shot_angle_bin','loc_x', 'loc_y','period', 'season']    

#Label encoding
le = LabelEncoder()
for col in kobe_data_2:
    if kobe_data_2[col].dtype=='object':
        kobe_data_2[col] = le.fit_transform(kobe_data_2[col])

kobe_test = kobe_data_2.loc[kobe_data_2.shot_made_flag.isnull(), :]
kobe_test.index = range(len(kobe_test))

kobe_data_2.dropna(inplace=True)
kobe_data_2.index =  range(len(kobe_data_2))
#models
xgb = xgboost.XGBClassifier(seed=1, learning_rate=0.01, n_estimators=500,silent=False, max_depth=7, subsample=0.6, colsample_bytree=0.6)

kf = KFold(n=len(kobe_data_2), shuffle=True,n_folds=5, random_state=50)

def run_test(predictors):
    all_score = []
    for train_index, test_index in kf:
        xgb.fit(kobe_data_2.loc[train_index, predictors], kobe_data_2.loc[train_index, 'shot_made_flag'])
        score = metrics.log_loss(kobe_data_2.loc[test_index, 'shot_made_flag'], xgb.predict_proba(kobe_data_2.loc[test_index, predictors])[:,1])
        all_score.append(score)
        print(score)
    print('Mean score =', np.array(all_score).mean())
    
#Fit
xgb.fit(kobe_data_2[predictors], kobe_data_2['shot_made_flag'])

#Test
xgb.predict_proba(kobe_test[predictors])[:,1]

preds = xgb.predict_proba(kobe_test[predictors])

kobe_test['shot_made_flag'] = xgb.predict_proba(kobe_test[predictors])[:,1]
