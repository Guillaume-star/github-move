#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import plotly
import plotly.express as px
import plotly.graph_objects as go
import seaborn as sns

get_ipython().run_line_magic('matplotlib', 'inline')
plotly.offline.init_notebook_mode(connected=True)

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier

from sklearn.metrics import accuracy_score, classification_report,confusion_matrix

import statsmodels.formula.api as smf

import warnings
warnings.filterwarnings('ignore')

from imblearn.under_sampling import NearMiss, RandomUnderSampler
from imblearn.over_sampling import SMOTE, RandomOverSampler

from mlxtend.feature_selection import SequentialFeatureSelector as SFS


# # Exploratory Data Analysis

# In[2]:


recidivism_data = pd.read_excel("Documents/3-Year_Recidivism_for_Offenders_Released_from_Prison_in_Iowa.xls")
recidivism_data.head()


# In[3]:


# Shape of the data
recidivism_data.shape


# In[4]:


recidivism_data['release_type'].count()


# In[5]:


# Check for missing values
recidivism_data.isnull().sum()


# In[6]:


# Data cleaning
recidivism_data.drop(['main_supervising_district','new_offense_sub_type','new_offense_classification'], axis=1, inplace=True)
recidivism_data.head()


# In[7]:


# Computing yearly recidivism rate
recidivism_trend = recidivism_data.groupby(['reporting_year','returned_to_prison'])['sex'].count().reset_index()
recidivism_trend.columns = ['reporting_year','Returned_to_prison','count']
yearly_rate = pd.DataFrame({"reporting_year":[2013, 2014, 2015,2016,2017, 2018],'rate(%)':[30,30,32,34,35,37]})

# Plotting annual recidivism rate
fig = go.Figure()
fig.add_trace(go.Scatter(x = yearly_rate['reporting_year'],y= yearly_rate['rate(%)'],mode='lines+markers'))
fig.update_layout(template='presentation')
fig.update_layout(title ='Annual Recidivism Rate', xaxis_title = 'Year', yaxis_title = 'rate(%)')
fig.show()


# In[8]:


# Total recidivism rate
recidivism_rate = recidivism_data.groupby('returned_to_prison')['sex'].count().reset_index()
recidivism_rate.columns =['returned_to_prison','count']
recidivism_rate


# In[9]:


#Plotting Total recidivism rate
fig = px.pie(recidivism_rate,names='returned_to_prison',values='count',hole=0.3, template='presentation')
fig.update_layout(title='Total Recidivism Rate', title_font_size=18)
fig.show()


# In[10]:


# Analyzing recidivism per release date
release_types = recidivism_data.groupby(['release_type','returned_to_prison'])['sex'].count().reset_index()
release_types.columns = ['release_type','returned_to_prison','count']
release_types


# In[11]:


# plotting recidivism per release date
y= release_types['count']
fig = px.bar(release_types, y='release_type', x='count',barmode='group', color='returned_to_prison', text=y,
           template='seaborn', labels={'release_type':'Release Type', 'returned_to_prison':'Returned to Prison'})
fig.update_layout(title ='Number of recidivists per release type')
fig.update_layout(paper_bgcolor="white")
fig.show()


# In[12]:


# Recidivism per race/ethnicity
ethnicity = recidivism_data.groupby(['race_ethnicity','returned_to_prison'])['sex'].count().reset_index()
ethnicity.columns = ['race_ethnicity','returned_to_prison','count']
#Plotting
fig=px.bar(ethnicity, y='race_ethnicity', x='count',barmode='group', color='returned_to_prison',
           template='plotly_white')
fig.update_layout(title='Number of Recidivists per Race/Ethnicity')
fig.show()


# In[13]:


#Computing number of recidivists per age group
age_group =recidivism_data.groupby(['age_at_release','returned_to_prison'])['sex'].count().reset_index()
age_group.columns = ['age_at_release','returned_to_prison','count']
#number of recidivists per age group
# plotting number of recidivists per age group
y= age_group['count']
fig=px.bar(age_group, x='age_at_release', y='count',barmode='group', color='returned_to_prison', text=y,
           category_orders={"age_at_release":['Under 25','25-34', '35-44', '45-54','55 and Older']},
           labels={'age_at_release':'Age at Release', 'returned_to_prison':'Returned to Prison'},
           template='seaborn')
fig.update_layout(title='Number of Recidivists per age at release')
#age_group


# In[14]:


#Computing number of recidivists per sex
sex = recidivism_data.groupby(['sex','returned_to_prison'])['offense_type'].count().reset_index()
sex.columns = ['sex','returned_to_prison','count']
#Plotting
y=sex['count']
fig=px.bar(sex, x='sex', y='count',barmode='group', color='returned_to_prison',
           template='seaborn', labels={'sex':''}, text=y)
fig.update_layout(title='Number of Recidivists per sex', showlegend=False)
fig.show()
sex


# In[15]:


#Computing number of recidivists per offense type
offense_types = recidivism_data.groupby(['offense_type','returned_to_prison'])['sex'].count().reset_index()
offense_types.columns = ['offense_type','returned_to_prison','count']
#plotting
y = offense_types['count']
fig=px.bar(offense_types, x='offense_type', y='count',barmode='group', color='returned_to_prison',
           text =y,labels={'offense_type':'Offense Type', 'returned_to_prison':'Returned to Prison'},
           template='seaborn')
fig.update_layout(title='Number of Recidivists per offense type')
fig.show()


# In[16]:


#Extracting recidivists data
recidivist = recidivism_data[recidivism_data['returned_to_prison']=='Yes']


# In[17]:


#Number of days to return per age at release
fig=px.box(recidivist, y='age_at_release', x='days_to_return',
           category_orders={"age_at_release":['Under 25','25-34', '35-44', '45-54','55 and Older']},
           labels={'days_to_return':'Days to Return', 'age_at_release':'Age at Release'},
           template='seaborn')
fig.update_layout(title='Number of days to return per age at release')
fig.show()


# In[18]:


#Number of days to return per per offense type
fig=px.box(recidivist, x='offense_type', y='days_to_return',template='seaborn',
           labels={'offense_type':'Offense Type', 'days_to_return':'Days to Return'})
fig.update_layout(title='Number of days to return per offense type')
fig.show()


# In[19]:


#Number od days to return by sex
fig=px.box(recidivist, y='sex', x='days_to_return',color='sex', template='seaborn',
           labels={'sex':'','days_to_return':'Days to Return'})
fig.update_layout(title='Number of days to return by sex', showlegend=False)
fig.show()


# In[20]:


# Total average days to return
recidivist['days_to_return'].mean()


# In[21]:


# Average days to return per release_type
average_days_to_return1= recidivist.groupby('release_type')['days_to_return'].mean().reset_index()
average_days_to_return1


# In[22]:


# Average days to return per race_ethnicity
average_days_to_return2= recidivist.groupby('race_ethnicity')['days_to_return'].mean().reset_index()
average_days_to_return2


# In[23]:


# Average days to return age at release
average_days_to_return3= recidivist.groupby('age_at_release')['days_to_return'].mean().reset_index()
average_days_to_return3


# In[24]:


# Average days to return sex
average_days_to_return4= recidivist.groupby('sex')['days_to_return'].mean().reset_index()
average_days_to_return4


# In[25]:


# Average days to return offense_type
average_days_to_return5= recidivist.groupby('offense_type')['days_to_return'].mean().reset_index()
average_days_to_return5


# In[26]:


# Counting the number of recidivists per age group
recidivism_by_age=recidivist.groupby('age_at_release').count()['sex'].reset_index()
recidivism_by_age.columns = ['age_at_release', 'count']
#Plotting
fig = px.pie(recidivism_by_age,names='age_at_release',values='count',hole=0.3, template='presentation')
fig.update_layout(title='Recidivism per age at release', title_font_size=18, legend_font_size=15)


# In[27]:


# Dropping NAs from release type column
recidivist = recidivism_data[recidivism_data['returned_to_prison']=='Yes']
recidivist.dropna(subset=['release_type'],inplace=True)


# In[28]:


#Number of days to return per release type
fig=px.box(recidivist, y='release_type', x='days_to_return', template='seaborn',
           labels={'days_to_return':'Days to Return', 'release_type':'Release Type'})
fig.update_layout(title='Number of days to return per release type')
fig.show()


# In[29]:


#Dropping NAs from days to return column
days=recidivism_data.dropna(subset=['days_to_return']).sort_values(by=['days_to_return'])
days


# In[30]:


returning= days[days['days_to_return']<418]['returned_to_prison']


# In[31]:


returning.value_counts()


# In[32]:


#Distribution of the number of days to return by sex
fig=px.histogram(days, 'days_to_return', color='sex', template='presentation',labels={'days_to_return': "Days to Return"})
fig.add_vline(x=457, line_width=3, line_dash="dash", line_color="orange")
fig.add_vline(x=471, line_width=3, line_dash="dash", line_color="blue")


# In[33]:


# Distribution of number of days to return
fig=px.histogram(days, 'days_to_return', template='presentation', labels={'days_to_return': "Days to Return"})
fig.add_vline(x=469, line_width=3, line_dash="dash", line_color="black", annotation_text=" Total average of days to return = 470")
fig.show()


# In[34]:


fig=px.box(days, 'days_to_return', template='presentation', labels={'days_to_return': "Days to Return"})
fig.show()


# # Statistical Modeling

# In[35]:


# Extracting variables of interest from the original dataset
data=recidivism_data[['release_type','age_at_release','sex','offense_type',
                      'returned_to_prison']].dropna()


# Dumyfying categorical variables

# In[36]:


release = {'Paroled':0, 'Discharged ':1, 'Released to Special Sentence':2,'Paroled w/Immediate Discharge':3,
           'Paroled to Detainer ':4}
data['release_type']=data['release_type'].map(release)


# In[37]:


age ={'25-34':0, '35-44':1, '45-54':2, 'Under 25':3, '55 and Older':4}
data["age_at_release"]=data["age_at_release"].map(age)


# In[38]:


sex = {'Male':0, 'Female':1}
data["sex"]=data["sex"].map(sex)


# In[39]:


offense = {'Violent':0, 'Property':1, 'Drug':2, 'Other':3, 'Public Order':4}
data["offense_type"]=data["offense_type"].map(offense)


# In[40]:


returned ={'No':0, 'Yes':1}
data['returned_to_prison']=data['returned_to_prison'].map(returned)


# In[41]:


# Defining X and y
X=data.iloc[:,:-1]
y =data['returned_to_prison']


# In[42]:


# The accuracy scaore of the baseline
y.value_counts(normalize=True)


# In[43]:


# Building logistic regression from statsmodel
formula = 'returned_to_prison~age_at_release+release_type+sex+offense_type'
logreg = smf.logit(formula,data).fit()
print(logreg.summary())


# In[44]:


marginal_freq = logreg.get_margeff()
print(marginal_freq.summary())


# In[45]:


logreg.pred_table(0.398)


# Modeling using Sklearn

# In[46]:


#Recalling X and y
X=data.iloc[:,:-1]
y =data['returned_to_prison']


# In[47]:


# Splitting our variables into training and testing
X_train, X_test, y_train, y_test = train_test_split(X,y, train_size=0.75, stratify=y)


# In[48]:


# Identyfiying the most important features for our model using Sequential features selection
sfs= SFS(RandomForestClassifier(n_estimators=100, random_state=0, n_jobs=-1), k_features=4,
        forward=True, floating=False, verbose=2, scoring='accuracy', cv=4, n_jobs=-1).fit(X_train, y_train)


# In[ ]:





# In[49]:


# Names of the selected features
sfs.k_feature_names_


# In[50]:


# Modeling using RandomForest
rfcl = RandomForestClassifier(n_estimators=10000, random_state=42)
rfcl.fit(X_train, y_train)
y_pred = rfcl.predict(X_test)
accuracy = accuracy_score(y_test,y_pred)
print('The accuracy score is:',accuracy)
print()
print(classification_report(y_test, y_pred))


# In[51]:


#  Modeling using Logistic Regression
logres = LogisticRegression( solver='lbfgs',max_iter=10000)
logres.fit(X_train, y_train)
y_pred = logres.predict(X_test)

accuracy = accuracy_score(y_test,y_pred)
print('The accuracy score is:',accuracy)
print()
print(classification_report(y_test, y_pred))


# In[52]:


# Modeling using KNN
knn = KNeighborsClassifier(n_neighbors=4)
knn.fit(X_train, y_train)
y_pred = knn.predict(X_test)
accuracy = accuracy_score(y_test,y_pred)
print('The accuracy score is:',accuracy)
print()
print(classification_report(y_test, y_pred))


# # Solving class imbalance using oversampling

# In[53]:


ros = RandomOverSampler()
X_train_over, y_train_over = ros.fit_resample(X_train, y_train)

logres_o = LogisticRegression( solver='lbfgs',max_iter=10000)
logres_o.fit(X_train_over, y_train_over)
y_pred = logres_o.predict(X_test)

accuracy = accuracy_score(y_test,y_pred)
print('The accuracy score is:',accuracy)
print()
print(classification_report(y_test, y_pred))


# In[54]:


knn_o = KNeighborsClassifier(n_neighbors=4)
knn_o.fit(X_train_over, y_train_over)
y_pred = knn_o.predict(X_test)
accuracy = accuracy_score(y_test,y_pred)
print('The accuracy score is:',accuracy)
print()
print(classification_report(y_test, y_pred))


# In[55]:


#random forest classifier
rfcl_o = RandomForestClassifier(n_estimators=10000, random_state=42)
rfcl_o.fit(X_train_over, y_train_over)
y_pred = rfcl_o.predict(X_test)
accuracy = accuracy_score(y_test,y_pred)
print('The accuracy score is:',accuracy)
print()
print(classification_report(y_test, y_pred))


# # Solving class imbalance using undersampling

# In[56]:


# Logistic Regression UnderSampling
rus = RandomUnderSampler()
X_train_under, y_train_under = ros.fit_resample(X_train, y_train)

logres_u = LogisticRegression( solver='lbfgs',max_iter=10000)
logres_u.fit(X_train_under, y_train_under)
y_pred = logres_u.predict(X_test)
accuracy = accuracy_score(y_test,y_pred)
print('The accuracy score is:',accuracy)
print()
print(classification_report(y_test, y_pred))


# In[57]:


# knn undersampling
knn_u = KNeighborsClassifier(n_neighbors=4)
knn_u.fit(X_train_under, y_train_under)
y_pred = knn_u.predict(X_test)
accuracy = accuracy_score(y_test,y_pred)
print('The accuracy score is:',accuracy)
print()
print(classification_report(y_test, y_pred))


# In[58]:


# RandomForest Undersampling
rfcl_u = RandomForestClassifier(n_estimators=10000, random_state=42)
rfcl_u.fit(X_train_under, y_train_under)
y_pred = rfcl_u.predict(X_test)
accuracy = accuracy_score(y_test,y_pred)
print('The accuracy score is:',accuracy)
print()
print(classification_report(y_test, y_pred))

