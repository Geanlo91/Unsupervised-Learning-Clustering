import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt


data = pd.read_csv('final_data_clustered.csv')


#mode per each feature and cluster      
mode_per_cluster = data.groupby('cluster').agg(lambda x: x.mode().iloc[0])
#plot bar chart of the mode per cluster
plt.figure(figsize=(10, 5))
sns.barplot(mode_per_cluster.index)

#mean per each feature and cluster
mean_per_cluster = data.groupby('cluster').mean()
#print(mean_per_cluster)

#plot age average per cluster
sns.barplot(x='cluster', y='What is your age?', data=data)
plt.title('Age Average per Cluster')
plt.show()

#plot the number of gender per cluster
gender_count = data.groupby('cluster')['What is your gender?'].value_counts().unstack().plot(kind='bar', stacked=True)
plt.title('Gender per cluster')
#assign legend
plt.legend('MF')

#Describe the data in the 'Mental health effects on work treated effectively' column
mental_health = data.groupby('cluster')['Mental health effects on work when treated effectively'].describe()
print(mental_health)

#plot 'Support' column per cluster
sns.barplot(x='cluster', y='Support', data=data)
plt.title('Support role per Cluster')
plt.show()


#plot 'Supervisor/Team Lead' column per cluster
sns.barplot(x='cluster', y='Supervisor/Team Lead', data=data)
plt.title('Supervisor role per Cluster')
plt.show()

#plot 'Sales' column per cluster
sns.barplot(x='cluster', y='Sales', data=data)
plt.title('Sales role per Cluster')
plt.show()

#plot 'HR' column per cluster
sns.barplot(x='cluster', y='HR', data=data)
plt.title('HR role per Cluster')
plt.show()

#plot 'Front-end Developer' column per cluster
sns.barplot(x='cluster', y='Front-end Developer', data=data)
plt.title('Front-end Developer role per Cluster')
plt.show()

#plot 'Executive Leadership' column per cluster
sns.barplot(x='cluster', y='Executive Leadership', data=data)
plt.title('Executive Leadership role per Cluster')
plt.show()

#plot 'DevOps/SysAdmin' column per cluster
sns.barplot(x='cluster', y='DevOps/SysAdmin', data=data)
plt.title('DevOps/SysAdmin role per Cluster')
plt.show()

#plot 'Dev Evangelist/Advocate' column per cluster
sns.barplot(x='cluster', y='Dev Evangelist/Advocate', data=data)
plt.title('Dev Evangelist/Advocate role per Cluster')
plt.show()