#!/usr/bin/env python
# coding: utf-8

# In[400]:


#####################################################################################################
######################### LOAN DATA SET  #######################################################
#####################################################################################################


# In[401]:


#################################################################
############ Part I - Importing
#################################################################

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
sns.set_style('whitegrid')
get_ipython().run_line_magic('matplotlib', 'inline')


# In[402]:


#### getting the data

df = pd.read_csv('loan_data.csv')


# In[403]:


df.head()


# In[404]:


df.info()


# In[405]:


#####################################################################
########################### Part II - Duplicates
#####################################################################


# In[406]:


df[df.duplicated()]                   #### no duplicates found


# In[407]:


####################################################################
############## Part III - Missing Values
####################################################################


# In[408]:


from matplotlib.colors import LinearSegmentedColormap

Amelia = LinearSegmentedColormap.from_list('black_yellow', ['black', 'yellow'])


# In[409]:


fig, ax = plt.subplots(figsize=(20,8))

sns.heatmap(df.isnull(),yticklabels=False,cbar=False,cmap=Amelia,ax=ax)

ax.set_xlabel('Columns')
ax.set_ylabel('Rows')
ax.set_title('Missing Data Heatmap')

#### why Amelia, if you coming from R then you might have used Amelia package which detects the missing value 
#### On July 2, 1937, Amelia disappeared over the Pacific Ocean while attempting to become the first female pilot to circumnavigate the world


# In[410]:


df.isnull().any()                         #### no null data


# In[411]:


####################################################################
############## Part IV - Feature Engineering
####################################################################


# In[412]:


df.head()


# In[413]:


df.purpose.unique()


# In[414]:


df['pub.rec'].unique()


# In[415]:


df['credit.policy'].unique()


# In[416]:


df.rename(columns={'credit.policy':'credit_policy',
                   'int.rate':'int_rate',
                   'log.annual.inc':'annual_income',
                   'dti':'debt_to_income',
                   'days.with.cr.line':'days_credit_line',
                   'revol.bal':'revol_balance',
                   'revol.util':'revol_utilization_rate',
                   'inq.last.6mths':'inq_6_months',
                   'delinq.2yrs':'past_due',
                   'pub.rec':'public_records',
                   'not.fully.paid':'paid'},inplace=True)


# In[417]:


df.head()


# In[418]:


######################################################################
############## Part V - EDA
######################################################################


# In[419]:


df['revol_utilization_rate'].plot(legend=True,figsize=(20,7),marker='o',markerfacecolor='red',color='black')

plt.title('Loan Revol Utilization Graph')

plt.xlabel('Number of people')

plt.ylabel('Density')


#### revol_utilization_rate is amount of credit line used compared to what they have in total
#### it seems the mean is falling around 40-50 ratio which honestly is not bad
#### we will have to see their revol_balance to know if they pay it all back 


# In[420]:


df.revol_utilization_rate.mean()


# In[421]:


df.revol_utilization_rate.std()


# In[422]:


df['revol_balance'].plot(legend=True,figsize=(20,7),marker='o',markerfacecolor='red',color='black')

plt.title('Loan Revol Balance Graph')

plt.xlabel('Number of people')

plt.ylabel('Density')


#### from the plot it seems theres some massive outliers here but the mean seems to be in 0.2 density range, meaning around 1500-1800 range


# In[423]:


df.revol_balance.mean()                    #### seems like we were right about the mean


# In[424]:


df.revol_balance.min()


# In[425]:


df.revol_balance.max()                      #### this is throwing it off


# In[426]:


corr = df.corr()                            #### lets see the corr quickly


# In[427]:


fig, ax = plt.subplots(figsize=(20,8))

sns.heatmap(corr,ax=ax,linewidths=0.5,annot=True,cmap='viridis')


#### from this its very crystal which feature columns we will pay more attention to


# In[428]:


df['inq_6_months'].plot(legend=True,figsize=(20,7),marker='o',markerfacecolor='red',color='black')

plt.title('Loan Inquiry Graph')

plt.xlabel('Number of people')

plt.ylabel('Density')

#### seems like the mean should be between 0-5


# In[429]:


df.inq_6_months.unique()


# In[430]:


df.inq_6_months.mean()


# In[431]:


df.head()


# In[432]:


custom = {0:'purple',
         1:'red'}

g = sns.jointplot(x=df.inq_6_months,y=df.int_rate,data=df,hue='paid',palette=custom)

g.fig.set_size_inches(17,9)


#### its hard to derive any conclusion from this but the sweat spot is people who get more then 10-15 inqueries majority dont pay
#### but again deriving anything from this is not a good idea as people who got more then 30 inqueries did pay up the loan


# In[433]:


g = sns.jointplot(x='installment',y='annual_income',data=df,kind='reg',color='black',joint_kws={'line_kws':{'color':'red'}})

g.fig.set_size_inches(17,9)


#### seems like as the scale for income increases the installment for the payment also increases which suggests people with more income get more credit balance hence the increase which makes sense


# In[434]:


from scipy.stats import pearsonr                  #### lets see this with pearsonr


# In[435]:


co_eff, p_value = pearsonr(df.annual_income,df.installment)


# In[436]:


co_eff                              #### this is amazing correlation


# In[437]:


p_value                             #### p_value is less then 0.05 hence we accept correlation here


# In[438]:


df.head()                          #### we will go ahead now and convert the categorical column purpose to numerical to see if they have any correlation


# In[439]:


df.purpose.unique()


# In[440]:


df['reasons'] = df.purpose.map({'debt_consolidation':0,
                                'credit_card':1,
                                'all_other':2,
                                'home_improvement':3,
                                'small_business':4,
                                'major_purchase':5,
                                'educational':6})


# In[441]:


df.head(1)


# In[442]:


corr = df.corr()


# In[443]:


fig, ax = plt.subplots(figsize=(20,8))

sns.heatmap(corr,ax=ax,linewidths=0.5,annot=True,cmap='viridis')


#### seems like reasons is a factor to the end result, interesting


# In[444]:


sns.catplot(x='paid',y='installment',data=df,kind='box',height=7,aspect=2,legend=True,hue='purpose',palette='Set2')


#### pretty interesting, its a very tight between who pays and who dont pay
#### majority of people who don't pay are from small businesses but on the other hand majority of people who do pay are from the same category
#### the only major point we can derive from here is that people who take the loan based on major purchases majority of them pay us back


# In[445]:


new_df = df.groupby(['purpose','fico'])['paid'].sum().unstack()

new_df                                 #### we did something very interesting here, this is how we can see who will pay and who will not


# In[446]:


new_df.fillna(0,inplace=True)


# In[447]:


new_df


# In[448]:


fig, ax = plt.subplots(figsize=(30,25)) 

sns.heatmap(new_df,linewidths=0.1,ax=ax,cmap='viridis')


#### this is quite interesting, from here we see that fico score doesn't matter to depict if they will pay or not
#### but we can derive something from here like people who take loans based on other reasons majority of them pay back when their fico scores are 640-730
#### seems like the safest bet is debt_consolidation and who has fico scores of 662-700 which is ironic if you ask me as someone will think better fico scores meaning they pay back quicker but not here


# In[449]:


new_df.loc['debt_consolidation'].sort_values(ascending=False).head()            #### the best ones who has paid us back from this section


# In[450]:


df_2 = new_df.loc['all_other'].sort_values(ascending=False).head().plot(legend=True,figsize=(20,7),marker='o',markerfacecolor='black',markersize=15,linestyle='dashed',linewidth=4,color='red')


#### this is top fico scores and people who paid back from all other reasons 


# In[451]:


sns.catplot(x='purpose',y='days_credit_line',data=df,kind='strip',height=7,aspect=2,palette=custom,legend=True,hue='paid',jitter=True)


#### seems like majority of people with higher number of days they had the credit ended up paying it except some few in each of the purpose categories


# In[452]:


new_df = df.groupby(['purpose','inq_6_months'])['paid'].sum().unstack()

new_df


# In[453]:


new_df.fillna(0,inplace=True)


# In[454]:


fig, ax = plt.subplots(figsize=(30,25)) 

sns.heatmap(new_df,linewidths=0.1,ax=ax,cmap='viridis')


#### this is pretty good way to see that people who had less inquery in last 6 month ended up paying their loans regardless of their reasons


# In[455]:


df.head()


# In[456]:


custom = {0:'red',
          1:'green'}

pl = sns.FacetGrid(df,hue='paid',aspect=4,height=4,palette=custom)

pl.map(sns.kdeplot,'int_rate',fill=True)

pl.add_legend()


#### seems like people don't tend to pay when the interest rate is low but when it goes to higher level people tends to be paying more


# In[457]:


pl = sns.catplot(y='paid',x='public_records',data=df,kind='point',hue='purpose',height=10,aspect=2,palette='Set2')

plt.xticks([0,1,2,3,4,5])


#### we can see that people who had less then 3 public records are more likely to pay back especially if its credit card as the feature


# In[458]:


custom = {0:'black',
          1:'green'}

sns.lmplot(x='revol_balance',y='annual_income',data=df,hue='paid',palette=custom,height=6,aspect=2)


#### we have some major outliers here but definately we see the linear relationship between annual income and revol balance


# In[459]:


custom = {0:'red',
          1:'purple'}


sns.catplot(x='purpose',data=df,kind='count',hue='paid',palette=custom,height=7,aspect=2)

#### this is quite revealing and telling, seems like debt_consolidation is the biggest one here who hasn't paid fully


# In[460]:


df.head()


# In[461]:


custom = {0:'black',
          1:'green'}

sns.lmplot(x='days_credit_line',y='annual_income',data=df,hue='paid',palette=custom,height=6,aspect=2)


#### we clearly see the linear relationship here, also its hard but we can see as credit line days increase the probability of paying it goes down


# In[462]:


mean_df = df.int_rate.mean()

mean_df


# In[463]:


std_df = df.int_rate.std()

std_df


# In[464]:


#### lets dive deeper into interest rate in the loans 

from scipy.stats import norm

x = np.linspace(mean_df - 4*std_df, mean_df + 4*std_df, 1000)
y = norm.pdf(x, mean_df, std_df)

#### plot
plt.figure(figsize=(20, 7))

#### normal distribution curve
plt.plot(x, y, label='Normal Distribution')

#### areas under the curve
plt.fill_between(x, y, where=(x >= mean_df - std_df) & (x <= mean_df + std_df), color='green', alpha=0.2, label='68%')
plt.fill_between(x, y, where=(x >= mean_df - 2*std_df) & (x <= mean_df + 2*std_df), color='purple', alpha=0.2, label='95%')
plt.fill_between(x, y, where=(x >= mean_df - 3*std_df) & (x <= mean_df + 3*std_df), color='red', alpha=0.2, label='99.7%')

#### mean and standard deviations
plt.axvline(mean_df, color='black', linestyle='dashed', linewidth=1)
plt.axvline(mean_df - std_df, color='red', linestyle='dashed', linewidth=1)
plt.axvline(mean_df + std_df, color='red', linestyle='dashed', linewidth=1)
plt.axvline(mean_df - 2*std_df, color='orange', linestyle='dashed', linewidth=1)
plt.axvline(mean_df + 2*std_df, color='orange', linestyle='dashed', linewidth=1)
plt.axvline(mean_df - 3*std_df, color='yellow', linestyle='dashed', linewidth=1)
plt.axvline(mean_df + 3*std_df, color='yellow', linestyle='dashed', linewidth=1)

plt.text(mean_df, plt.gca().get_ylim()[1]*0.9, f'Mean: {mean_df:.2f}', horizontalalignment='center', fontsize=12, verticalalignment='bottom', color='black')
plt.text(mean_df + std_df, plt.gca().get_ylim()[1]*0.05, f'z=1    {mean_df + std_df:.2f}', horizontalalignment='center', fontsize=12, verticalalignment='bottom', color='black')
plt.text(mean_df - std_df, plt.gca().get_ylim()[1]*0.05, f'z=-1   {mean_df - std_df:.2f}', horizontalalignment='center', fontsize=12, verticalalignment='bottom', color='black')
plt.text(mean_df + 2*std_df, plt.gca().get_ylim()[1]*0.05, f'z=2  {mean_df + 2*std_df:.2f}', horizontalalignment='center', fontsize=12, verticalalignment='bottom', color='black')
plt.text(mean_df - 2*std_df, plt.gca().get_ylim()[1]*0.05, f'z=-2 {mean_df - 2*std_df:.2f}', horizontalalignment='center', fontsize=12, verticalalignment='bottom', color='black')
plt.text(mean_df + 3*std_df, plt.gca().get_ylim()[1]*0.05, f'z=3  {mean_df + 3*std_df:.2f}', horizontalalignment='center', fontsize=12, verticalalignment='bottom', color='black')
plt.text(mean_df - 3*std_df, plt.gca().get_ylim()[1]*0.05, f'z=-3 {mean_df - 3*std_df:.2f}', horizontalalignment='center', fontsize=12, verticalalignment='bottom', color='black')


#### annotate the plot
plt.text(mean_df, max(y), 'Mean', horizontalalignment='center', fontsize=12, verticalalignment='bottom', color='black')
plt.text(mean_df - std_df, max(y), '-1σ', horizontalalignment='center', fontsize=12, verticalalignment='bottom', color='black')
plt.text(mean_df + std_df, max(y), '+1σ', horizontalalignment='center', fontsize=12, verticalalignment='bottom', color='black')
plt.text(mean_df - 2*std_df, max(y), '-2σ', horizontalalignment='center', fontsize=12, verticalalignment='bottom', color='black')
plt.text(mean_df + 2*std_df, max(y), '+2σ', horizontalalignment='center', fontsize=12, verticalalignment='bottom', color='black')
plt.text(mean_df - 3*std_df, max(y), '-3σ', horizontalalignment='center', fontsize=12, verticalalignment='bottom', color='black')
plt.text(mean_df + 3*std_df, max(y), '+3σ', horizontalalignment='center', fontsize=12, verticalalignment='bottom', color='black')

#### labels
plt.title('Fare distribution inside the Titanic Dataset')
plt.xlabel('Fare')
plt.ylabel('Probability Density')

plt.legend()


#### seems like the mean deviates + or - 0.02 on either side


# In[465]:


#### lets find out the interest rate of customers with confidence level of 95% then increase it to 99%

standard_error = std_df/np.sqrt(df.shape[0])


# In[466]:


from scipy import stats

stats.norm.interval(alpha=0.95,loc=mean_df,scale=standard_error)               


# In[467]:


#### 99% confidence level that the interest rate falls between these for most customers

stats.norm.interval(alpha=0.99,loc=mean_df,scale=standard_error)


# In[468]:


df.int_rate.max()


# In[469]:


sns.lmplot(x='revol_utilization_rate',y='paid',data=df,hue='past_due',height=7,aspect=2,x_bins=[0,1,3,5,10,15,20,30,40,50,60,70,80,100,120],palette='Set2')


#### the most obvious and clear linear relationship I see is people who are past due 4 are most likely to pay back


# In[470]:


sns.lmplot(x='inq_6_months',y='paid',data=df,hue='past_due',height=7,aspect=2,x_bins=[0,1,3,5,10,15,20,30,40,50,60,70,80,100,120],palette='Dark2')


#### clearly we do see linear correlation here too


# In[471]:


custom = {0:'black',
          1:'green'}

g = sns.jointplot(x=df.inq_6_months,y=df.past_due,data=df,hue='paid',kind='kde',fill=True,palette=custom)

g.fig.set_size_inches(17,9)

#### clearly we see something interesting here, people who are past due in the range of 0-1 and inqueries in the range of 0-10 are majority of good candidates


# In[472]:


custom = {0:'black',
          1:'red'}

g = sns.jointplot(x=df.days_credit_line,y=df.revol_utilization_rate,data=df,hue='paid',kind='kde',fill=True,palette=custom)

g.fig.set_size_inches(17,9)


#### seems like evol_util_rate 60-80 and credit day line between 2500 and 5000 are the heavily densed paid customers


# In[473]:


df.head()


# In[474]:


heat = df.groupby(['past_due','inq_6_months'])['paid'].sum().unstack().fillna(0)

heat.head()

#### seems like people who had 0 inquery in last 6 months paid the most loans


# In[475]:


fig, ax = plt.subplots(figsize=(30,25)) 

sns.heatmap(heat,linewidths=0.1,ax=ax,cmap='viridis',annot=True)


#### its obvious that people who had inqueries in the range of 0-7 and past due of range 0-3 had the best stats in paying the loans
#### we now moving with PCA


# In[476]:


######################################################################
############## Part VI - PCA
######################################################################


# In[477]:


X = df.drop(columns=['purpose','paid'])

X.head()


# In[478]:


y = df['paid']

y.head()


# In[479]:


from sklearn.preprocessing import StandardScaler


# In[480]:


scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)


# In[481]:


from sklearn.decomposition import PCA


# In[482]:


pca = PCA(n_components=2)
principal_components = pca.fit_transform(X_scaled)
principal_df = pd.DataFrame(data=principal_components, columns=['principal_component_1', 'principal_component_2'])
final_df = pd.concat([principal_df, y], axis=1)


# In[483]:


final_df.head()                      #### beauty of PCA, amazing


# In[484]:


colors = {0: 'black', 1: 'green'}

plt.figure(figsize=(15, 6))

for i in final_df['paid'].unique():
    subset = final_df[final_df['paid'] == i]
    plt.scatter(subset['principal_component_1'], subset['principal_component_2'], 
                color=colors[i], label=f'paid = {i}')

plt.xlabel('Principal Component 1')
plt.ylabel('Principal Component 2')
plt.title('2D PCA of Titanic Dataset')
plt.legend()
plt.grid(True)


#### PCA makes it so much easier to make distinction and classify them


# In[485]:


pca.n_features_                   #### how many features cols it has


# In[486]:


pca.components_


# In[487]:


X.columns


# In[488]:


df_comp = pd.DataFrame(pca.components_,columns=['credit_policy', 'int_rate', 'installment', 'annual_income','debt_to_income', 'fico', 'days_credit_line', 'revol_balance',
                                                'revol_utilization_rate', 'inq_6_months', 'past_due', 'public_records','reasons'])


# In[489]:


df_comp.head()


# In[490]:


fig, ax = plt.subplots(figsize=(20,8)) 

sns.heatmap(df_comp,linewidths=0.1,ax=ax,cmap='viridis',annot=True)


#### this is just an amazing way to see what each of clusters correlation, if you were not provided with the target data then this is the way you can make clusters


# In[491]:


#######################################################################
######################## Part VII - PCA Model
#######################################################################


# In[492]:


final_df.head()


# In[493]:


X = final_df.drop(columns='paid')

X.head()


# In[494]:


y = final_df['paid']

y.head()


# In[495]:


from sklearn.model_selection import train_test_split


# In[496]:


X_train, X_test, y_train, y_test = train_test_split(X,y,test_size=0.3,random_state=42)


# In[497]:


from sklearn.linear_model import LogisticRegression


# In[498]:


model = LogisticRegression()


# In[499]:


model.fit(X_train,y_train)


# In[500]:


y_predict = model.predict(X_test)


# In[501]:


from sklearn import metrics


# In[502]:


print(metrics.classification_report(y_test,y_predict))               #### problem here is recall, precision and f1 for 1 because the support is so unbalanced


# In[503]:


y_test.value_counts()


# In[504]:


y.value_counts()                     #### this is the problem


# In[505]:


from sklearn.utils import resample


# In[506]:


df_majority = final_df[final_df['paid'] == 0]
df_minority = final_df[final_df['paid'] == 1]

df_minority


# In[507]:


df_majority_downsampled = resample(df_majority, 
                                   replace=False,    
                                   n_samples=2000,  
                                   random_state=123)
df_majority_downsampled


# In[508]:


final_df = pd.concat([df_majority_downsampled, df_minority])

final_df                #### we had to somehow balance target to not throw off our other metrics when we do the modelling part


# In[509]:


X = final_df.drop(columns='paid')

X.head()


# In[510]:


y = final_df['paid']

y.head()


# In[511]:


X_train, X_test, y_train, y_test = train_test_split(X,y,test_size=0.3,random_state=42)

y_test.value_counts()


# In[512]:


y_train.value_counts()


# In[513]:


model = LogisticRegression()


# In[514]:


model.fit(X_train,y_train)


# In[515]:


y_predict = model.predict(X_test)


# In[516]:


print(metrics.classification_report(y_test,y_predict))                #### not the best model but its ok for what we have as data set


# In[517]:


cm = metrics.confusion_matrix(y_test,y_predict)

labels = ['Not Paid','Paid']

disp = metrics.ConfusionMatrixDisplay(confusion_matrix=cm,display_labels=labels)

fig, ax = plt.subplots(figsize=(25,12))

disp.plot(ax=ax)


# In[518]:


#############################################################################
################# PART VIII - Classification
#############################################################################


# In[519]:


df.head()


# In[520]:


X = df.drop(columns=['paid','reasons'])

X.head()


# In[521]:


y = df['paid']

y.head()


# In[522]:


from imblearn.over_sampling import SMOTE


# In[523]:


from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler


# In[524]:


preprocessor = ColumnTransformer(
    transformers=[
        ('cat', OneHotEncoder(), ['purpose']),
        ('num', StandardScaler(), ['credit_policy', 'int_rate', 'installment', 
                                   'annual_income', 'debt_to_income', 'fico', 
                                   'days_credit_line', 'revol_balance', 
                                   'revol_utilization_rate', 'inq_6_months', 
                                   'past_due', 'public_records'])
    ])


# In[525]:


from imblearn.pipeline import Pipeline as ImbPipeline


# In[526]:


from xgboost import XGBClassifier


# In[527]:


model = ImbPipeline(steps=[
    ('preprocessor', preprocessor),
    ('smote', SMOTE(random_state=42)),
    ('classifier', XGBClassifier(use_label_encoder=False, eval_metric='logloss'))
])


# In[528]:


X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42, stratify=y)


# In[529]:


param_grid = {
    'classifier__n_estimators': [100, 200],
    'classifier__learning_rate': [0.01, 0.1],
    'classifier__max_depth': [3, 5]
}


# In[530]:


from sklearn.model_selection import train_test_split, RandomizedSearchCV


# In[531]:


X.count()


# In[532]:


random_search = RandomizedSearchCV(model, param_distributions=param_grid, n_iter=8, scoring='accuracy', cv=3, verbose=2, random_state=42)


# In[533]:


random_search.fit(X_train, y_train)


# In[534]:


best_model = random_search.best_estimator_
y_pred = best_model.predict(X_test)


# In[535]:


from sklearn.metrics import classification_report


# In[536]:


print(classification_report(y_test, y_pred))


# In[537]:


X = df.drop(columns=['past_due','credit_policy','annual_income','fico','days_credit_line','reasons'])

X.head()


# In[538]:


corr = X.corr()


# In[539]:


fig, ax = plt.subplots(figsize=(20,8))

sns.heatmap(corr,ax=ax,linewidths=0.5,annot=True,cmap='viridis')


# In[540]:


y = df['paid']

y.head()


# In[541]:


preprocessor = ColumnTransformer(
    transformers=[
        ('cat', OneHotEncoder(), ['purpose']),
        ('num', StandardScaler(), ['int_rate', 'installment', 
                                   'debt_to_income', 
                                   'revol_balance', 
                                   'revol_utilization_rate', 'inq_6_months', 
                                   'public_records'])
    ])


# In[542]:


X.drop(columns='paid',inplace=True)

X.head()


# In[543]:


model = ImbPipeline(steps=[
    ('preprocessor', preprocessor),
    ('smote', SMOTE(random_state=42)),
    ('classifier', XGBClassifier(use_label_encoder=False, eval_metric='logloss'))
])


# In[544]:


X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42, stratify=y)


# In[545]:


param_grid = {
    'classifier__n_estimators': [100, 200],
    'classifier__learning_rate': [0.01, 0.1],
    'classifier__max_depth': [3, 5]
}


# In[546]:


random_search = RandomizedSearchCV(model, param_distributions=param_grid, n_iter=8, scoring='accuracy', cv=5, verbose=2, random_state=42)


# In[547]:


get_ipython().run_cell_magic('time', '', '\nrandom_search.fit(X_train, y_train)')


# In[548]:


best_model = random_search.best_estimator_

best_model


# In[549]:


y_predict = best_model.predict(X_test)


# In[550]:


print(classification_report(y_test, y_predict))                        #### definately some improvement


# In[551]:


from imblearn.combine import SMOTEENN              #### more advanced resampling


# In[552]:


model = ImbPipeline(steps=[
    ('preprocessor', preprocessor),
    ('resample', SMOTEENN(random_state=42)),
    ('classifier', XGBClassifier(use_label_encoder=False, eval_metric='logloss'))
])


# In[553]:


param_grid = {
    'classifier__n_estimators': [100, 200, 300],
    'classifier__learning_rate': [0.01, 0.1, 0.2],
    'classifier__max_depth': [3, 5, 7],
    'classifier__colsample_bytree': [0.3, 0.7],
    'classifier__subsample': [0.5, 0.8]
}


# In[554]:


get_ipython().run_cell_magic('time', '', "\nrandom_search = RandomizedSearchCV(model, param_distributions=param_grid, n_iter=50, scoring='accuracy', cv=5, verbose=2, random_state=42)\nrandom_search.fit(X_train, y_train)")


# In[555]:


best_model = random_search.best_estimator_


# In[556]:


y_predict = best_model.predict(X_test)


# In[557]:


print(classification_report(y_test, y_predict))              #### we did bring up the recall and other metrics but our accuracy has gone down


# In[558]:


cm = metrics.confusion_matrix(y_test,y_predict)

labels = ['Not Paid','Paid']

disp = metrics.ConfusionMatrixDisplay(confusion_matrix=cm,display_labels=labels)

fig, ax = plt.subplots(figsize=(25,12))

disp.plot(ax=ax)

#### the support is throwing our model off even with SMOTEENN, if we had larger data for to train model then it would have been sorted


# In[559]:


from sklearn.ensemble import RandomForestClassifier, VotingClassifier


# In[560]:


xgb_clf = XGBClassifier(use_label_encoder=False, eval_metric='logloss')
log_clf = LogisticRegression(max_iter=1000)
rf_clf = RandomForestClassifier()


# In[561]:


voting_clf = VotingClassifier(estimators=[
    ('xgb', xgb_clf),
    ('lr', log_clf),
    ('rf', rf_clf)
], voting='soft')


# In[562]:


model = ImbPipeline(steps=[
    ('preprocessor', preprocessor),
    ('resample', SMOTEENN(random_state=42)),
    ('classifier', voting_clf)
])


# In[563]:


param_grid = {
    'classifier__xgb__n_estimators': [100, 200],
    'classifier__xgb__learning_rate': [0.01, 0.1],
    'classifier__xgb__max_depth': [3, 5],
    'classifier__rf__n_estimators': [100, 200],
    'classifier__rf__max_depth': [3, 5]
}


# In[564]:


get_ipython().run_cell_magic('time', '', "\nrandom_search = RandomizedSearchCV(model, param_distributions=param_grid, n_iter=32, scoring='accuracy', cv=5, verbose=2, random_state=42)\nrandom_search.fit(X_train, y_train)")


# In[565]:


best_model = random_search.best_estimator_


# In[566]:


y_predict = best_model.predict(X_test)


# In[567]:


print(classification_report(y_test, y_predict))


# In[568]:


from sklearn.ensemble import StackingClassifier
import xgboost as xgb


# In[569]:


base_models = [
    ('logreg', LogisticRegression(max_iter=1000, class_weight='balanced')),
    ('rf', RandomForestClassifier(class_weight='balanced', random_state=42)),
    ('xgb', xgb.XGBClassifier(use_label_encoder=False, eval_metric='logloss'))
]

meta_model = LogisticRegression()

stacking_clf = StackingClassifier(estimators=base_models, final_estimator=meta_model, cv=5)


# In[570]:


get_ipython().run_cell_magic('time', '', "\nmodel = ImbPipeline(steps=[\n    ('preprocessor', preprocessor),\n    ('resample', SMOTEENN(random_state=42)),\n    ('classifier', stacking_clf)\n])\n\nmodel.fit(X_train, y_train)")


# In[571]:


y_predict = model.predict(X_test)


# In[572]:


print(metrics.classification_report(y_test,y_predict))                      #### not the best to be honest but ok model


# In[576]:


from imblearn.over_sampling import ADASYN                        #### lets bring ADASYN


# In[573]:


model = ImbPipeline(steps=[
    ('preprocessor', preprocessor),
    ('resample', ADASYN(random_state=42)),
    ('classifier', XGBClassifier(use_label_encoder=False, eval_metric='logloss'))
])


# In[574]:


param_grid = {
    'classifier__n_estimators': [100, 200, 300],
    'classifier__learning_rate': [0.01, 0.1, 0.2],
    'classifier__max_depth': [3, 5, 7],
    'classifier__colsample_bytree': [0.3, 0.7],
    'classifier__subsample': [0.5, 0.8]
}


# In[575]:


get_ipython().run_cell_magic('time', '', "\nrandom_search = RandomizedSearchCV(model, param_distributions=param_grid, n_iter=50, scoring='accuracy', cv=5, verbose=2, random_state=42)\nrandom_search.fit(X_train, y_train)")


# In[577]:


best_model = random_search.best_estimator_


# In[578]:


y_predict = best_model.predict(X_test)


# In[579]:


print(classification_report(y_test, y_predict))              #### this is the best we have yet thanks to ADASYN


# In[580]:


xgb_clf = XGBClassifier(use_label_encoder=False, eval_metric='logloss')
log_clf = LogisticRegression(max_iter=1000)
rf_clf = RandomForestClassifier()


# In[581]:


voting_clf = VotingClassifier(estimators=[
    ('xgb', xgb_clf),
    ('lr', log_clf),
    ('rf', rf_clf)
], voting='soft')


# In[583]:


model = ImbPipeline(steps=[
    ('preprocessor', preprocessor),
    ('resample', ADASYN(random_state=42)),
    ('classifier', voting_clf)
])


# In[584]:


param_grid = {
    'classifier__xgb__n_estimators': [100, 200],
    'classifier__xgb__learning_rate': [0.01, 0.1],
    'classifier__xgb__max_depth': [3, 5],
    'classifier__rf__n_estimators': [100, 200],
    'classifier__rf__max_depth': [3, 5]
}


# In[585]:


random_search = RandomizedSearchCV(model, param_distributions=param_grid, n_iter=50, scoring='accuracy', cv=5, verbose=2, random_state=42)
random_search.fit(X_train, y_train)


# In[586]:


best_model = random_search.best_estimator_


# In[587]:


y_predict = best_model.predict(X_test)


# In[588]:


print(classification_report(y_test, y_predict))


# In[ ]:


#### one last try by manually splitting and making the target evenly balanced, if we had bigger data then it would have been much easier and accurate


# In[371]:


df_majority = df[df['paid'] == 0]
df_minority = df[df['paid'] == 1]

df_minority


# In[383]:


df_majority_downsampled = resample(df_majority, 
                                   replace=False,    
                                   n_samples=1600,  
                                   random_state=123)
df_majority_downsampled


# In[384]:


new_df = pd.concat([df_majority_downsampled, df_minority])

new_df                #### we had to somehow balance target to not throw off our other metrics when we do the modelling part


# In[385]:


X = new_df.drop(columns=['past_due','credit_policy','annual_income','fico','days_credit_line','reasons','paid'])

X.head()


# In[386]:


y = new_df['paid']

y.head()


# In[387]:


X_train, X_test, y_train, y_test = train_test_split(X,y,test_size=0.3,random_state=42)

y_test.value_counts()


# In[388]:


y_train.value_counts()


# In[389]:


preprocessor = ColumnTransformer(
    transformers=[
        ('cat', OneHotEncoder(), ['purpose']),
        ('num', StandardScaler(), ['int_rate', 'installment', 
                                   'debt_to_income', 
                                   'revol_balance', 
                                   'revol_utilization_rate', 'inq_6_months', 
                                   'public_records'])
    ])


# In[390]:


base_models = [
    ('logreg', LogisticRegression(max_iter=1000, class_weight='balanced')),
    ('rf', RandomForestClassifier(class_weight='balanced', random_state=42)),
    ('xgb', xgb.XGBClassifier(use_label_encoder=False, eval_metric='logloss'))
]

meta_model = LogisticRegression()

stacking_clf = StackingClassifier(estimators=base_models, final_estimator=meta_model, cv=5)


# In[367]:


from sklearn.pipeline import Pipeline


# In[391]:


get_ipython().run_cell_magic('time', '', "\nmodel = Pipeline(steps=[\n    ('preprocessor', preprocessor),\n    ('classifier', stacking_clf)\n])\n\nmodel.fit(X_train, y_train)")


# In[392]:


y_predict = model.predict(X_test)


# In[393]:


print(metrics.classification_report(y_test,y_predict))                    #### this is more balanced and the metrics are more evenly split


# In[589]:


##########################################################################################################################
#### After thorough experimentation, we have decided to halt further model optimization as we are observing  #############
#### diminishing returns in performance improvements. When dealing with imbalanced datasets, it is crucial to  ###########
#### prioritize specific evaluation metrics that align with our desired outcomes. Due to the significant imbalance #######
#### in our target variable, maintaining uniformity across all metrics has proven challenging. Despite employing a #######
#### range of advanced techniques, the most satisfactory results were achieved using the ADASYN resampling method.######
##########################################################################################################################

