import numpy as np
import pandas as pd
import os
import matplotlib.pyplot as plt
import seaborn as sns
dataset = pd.read_csv("master.csv")

#this simply gives us list of countries in our dataset
#we can do this for generations and sex too
unique_country = dataset['country'].unique()
print(unique_country)

#this tells us how much data of each country is available in our dataset
#we can do it for sex and generations too
alpha = 1
plt.figure(figsize=(20,45))
sns.countplot(y='country', data=dataset, alpha=alpha)
plt.title('Data by country')

#this shows the no. of male and female so we know if our dataset is good if it have almost equal numbers
plt.figure(figsize=(16,7))
#Plot the graph
sex = sns.countplot(x='sex',data = dataset)
#plt.show()

#this gives the correlation matrix heatmap between each 2 features
#correlation is strong where color is light
plt.figure(figsize=(16,7))
cor = sns.heatmap(dataset.corr(), annot = True)
#plt.show()

#age groups tht commit most suicides grouped on bases of gender
plt.figure(figsize=(16,7))
###this gives us barplot
bar_age = sns.barplot(x = 'sex', y = 'suicides_no', hue = 'age',data = dataset)
plt.show()

#this is to get mean std count etc of our dataset
dataset.describe()

dataset.info()

#Now create a varible that stores a 'male' & 'female' data so we can easily plot a Lineplot
male_population = dataset.loc[dataset.loc[:, 'sex']=='male',:]
female_population = dataset.loc[dataset.loc[:, 'sex']=='female',:]
#Set figure size
plt.figure(figsize=(16,7))
##Plot the Lineplot
lp_male = sns.lineplot(x = 'year' , y = 'suicides_no' , data = male_population)
lp_female = sns.lineplot(x = 'year' , y = 'suicides_no' , data = female_population)
leg1 = plt.legend(['Males','Females'])

#similar to male female implement the line plot for age groups
age_5 = dataset.loc[dataset.loc[:, 'age']=='5-14 years',:]
age_15 = dataset.loc[dataset.loc[:, 'age']=='15-24 years',:]
age_25 = dataset.loc[dataset.loc[:, 'age']=='25-34 years',:]
age_35 = dataset.loc[dataset.loc[:, 'age']=='35-54 years',:]
age_55 = dataset.loc[dataset.loc[:, 'age']=='55-74 years',:]
age_75 = dataset.loc[dataset.loc[:, 'age']=='75+ years',:]
#Set figure size
plt.figure(figsize=(16,7))
#Now lets plot a line plot
age_5_lp = sns.lineplot(x='year', y='suicides_no', data=age_5)
age_15_lp = sns.lineplot(x='year', y='suicides_no', data=age_15)
age_25_lp = sns.lineplot(x='year', y='suicides_no', data=age_25)
age_35_lp = sns.lineplot(x='year', y='suicides_no', data=age_35)
age_55_lp = sns.lineplot(x='year', y='suicides_no', data=age_55)
age_75_lp = sns.lineplot(x='year', y='suicides_no', data=age_75)
leg = plt.legend(['5-14 years', '15-24 years', '25-34 years', '35-54 years', '55-74 years', '75+ years'])
plt.show()
