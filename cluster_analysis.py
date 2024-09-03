import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt


data = pd.read_csv('final_data_clustered.csv')
xdata = pd.read_csv('clustered_data.csv')


#plot age average per cluster
sns.barplot(x='cluster', y='What is your age?', data=data)
plt.title('Average age per Cluster')
plt.show()

#plot the number of gender per cluster
gender_count = data.groupby('cluster')['What is your gender?'].value_counts().unstack().plot(kind='bar', stacked=True)
plt.title('Gender per cluster')
plt.legend(title='Gender ', loc='upper right', labels =['Female','Male'])
plt.show()


#plot 'Support' column per cluster
support_count = data.groupby('cluster')['Support'].value_counts().unstack().plot(kind='bar', stacked=True)
plt.title('Support roles per Cluster')
plt.legend(title='Responce categories', loc='upper right', labels=['No', 'Yes'])
plt.show()


#plot 'Supervisor/Team Lead' column per cluster
sns.barplot(x='cluster', y='Supervisor/Team Lead', data=data)
plt.title('Supervisor roles per Cluster')
plt.show()

#plot 'Sales' column per cluster
sns.barplot(x='cluster', y='Sales', data=data)
plt.title('Sales roles per Cluster')
plt.show()

#plot 'HR' column per cluster
sns.barplot(x='cluster', y='HR', data=data)
plt.title('HR roles per Cluster')
plt.show()

#plot 'Front-end Developer' column per cluster
sns.barplot(x='cluster', y='Front-end Developer', data=data)
plt.title('Front-end Developer roles per Cluster')
plt.show()

#plot 'Executive Leadership' column per cluster
sns.barplot(x='cluster', y='Executive Leadership', data=data)
plt.title('Executive Leadership roles per Cluster')
plt.show()

#plot 'DevOps/SysAdmin' column per cluster
sns.barplot(x='cluster', y='DevOps/SysAdmin', data=data)
plt.title('DevOps/SysAdmin roles per Cluster')
plt.show()

#plot 'Dev Evangelist/Advocate' column per cluster
sns.barplot(x='cluster', y='Dev Evangelist/Advocate', data=data)
plt.title('Dev Evangelist/Advocate roles  per Cluster')
plt.legend(title='Responce categories', loc='upper right', labels=['Yes', 'No'])
plt.show()


#plot the average employee size per cluster from the 'How many employees does your company or organization have?' column
sns.barplot(x='cluster', y='How many employees does your company or organization have?', data=data)
plt.title('Average employee size per Cluster')
plt.show()

#plot number of people pre cluster on the 'Do you know the options for mental health care available under your employer-provided coverage?' column
sns.countplot(x='cluster', hue='Do you know the options for mental health care available under your employer-provided coverage?', data=data)
plt.title('Do you know the options for mental health care available under your employer-provided coverage?')
plt.legend(title='Responce categories', loc='upper right', labels=['No response', 'No','I am not sure', 'Yes'])
plt.show()

#plot number of people pre cluster on the 'Were you aware of the options for mental health care provided by your previous employers?' column
sns.countplot(x='cluster', hue='Were you aware of the options for mental health care provided by your previous employers?', data=data)
plt.title('Awareness of mental health care options provided by previous employers per Cluster')
plt.legend(title='Responce categories', loc='upper right', labels=['No response', 'No','Was aware of some', 'Yes'])
plt.show()

#plot the top 3 countries per cluster
country_counts = data.groupby(['cluster', 'What country do you work in?']).size().reset_index(name='count')
top_countries_per_cluster = country_counts.groupby('cluster').apply(lambda x: x.nlargest(3, 'count')).reset_index(drop=True)
plt.figure(figsize=(10, 6))
sns.barplot(x='cluster', y='count', hue='What country do you work in?', data=top_countries_per_cluster)
plt.legend(title='Country code', loc='upper right')
#add legend explaining the country codes
plt.text(0.5, 0.5, '2: Australia\n10: Canada\n21: Germany\n33:Netherlands\n49:United Kingdom\n50:USA', fontsize=10, transform=plt.gcf().transFigure)
plt.title('Top 3 Countries per Cluster')
plt.show()

#plot number of people pre cluster on the 'Do you work remotely?' column
sns.countplot(x='cluster', hue='Do you work remotely?', data=data)
plt.title('Remote working per Cluster')
plt.legend(title='Responce categories', loc='upper right', labels=['Never', 'Sometimes','Always'])
plt.show()

sns.countplot(x='cluster', hue='Do you have a family history of mental illness?', data=data)
plt.title('Family history of mental illness per Cluster')
plt.legend(title='Responce categories', loc='upper right', labels=["I don't know", 'No','Yes'])
plt.show()


sns.countplot(x='cluster', hue='Have you observed or experienced an unsupportive or badly handled response to a mental health issue in your current or previous workplace?', data=data)
plt.title('Have you observed or experienced Unsupportive response to mental health issue per Cluster')
plt.legend(title='Responce categories', loc='upper right', labels=['No response', 'No','Maybe','Yes'])
plt.show()


sns.countplot(x='cluster', hue='Have you had mental health benefits before', data=data)
plt.title('Prior mental health benefits per Cluster')
plt.legend(title='Responce categories', loc='upper right', labels=["I don't know", 'No','Some', 'Yes'])
plt.show()

sns.countplot(x='cluster', hue='How easy is it to ask for mental health leave?', data=data)
plt.title('Ease of asking for mental health leave per Cluster')
plt.legend(title='Response categories', loc='upper right', labels=['Very difficult','Somewhat difficult','Neither easy nor difficult','Somewhat easy','Very easy'])
plt.show()

sns.countplot(x='cluster', hue='Mental health effects on work when treated effectively', data=data)
plt.title('Mental health effects on work when treated effectively per Cluster')
plt.legend(title='Response categories', loc='upper right', labels=['No response', 'Never','Rarely','Often'])
plt.show()

sns.countplot(x='cluster', hue='Do you think that discussing a mental health disorder with previous employers would have negative consequences?', data=data)
plt.title('Negative consequancesfrom discussing mental health disorder with previous employers per Cluster')
plt.legend(title='Response categories', loc='upper right', labels=["I don't know", 'None of them','Some of them','Yes, all of them'])
plt.show()

sns.countplot(x='cluster', hue='Do you think that discussing a physical health issue with previous employers would have negative consequences?', data=data)
plt.title('Negative consequances from discussing physical health issue with previous employers per Cluster')
plt.legend(title='Response categories', loc='upper right', labels=["I don't know", 'None of them','Some of them','Yes, all of them'])
plt.show()

sns.countplot(x='cluster', hue='Would you have been willing to discuss a mental health issue with your direct supervisor(s)?', data=data)
plt.title('Willingness to discuss mental health issue with direct supervisor per Cluster')
plt.legend(title='Response categories', loc='upper right', labels=["I don't know", 'No','Some','Yes,all'])
plt.show()

#plot number of people pre cluster on the 'Do you think that discussing a physical health issue with your employer would have negative consequences?' column
sns.countplot(x='cluster', hue='Do you think that discussing a physical health issue with your employer would have negative consequences?', data=data)
plt.title('Physical health issue discussion with employer per Cluster')
plt.legend(title='Responce categories', loc='upper right', labels=["I don't know", 'No','Maybe', 'Yes'])
plt.show()

#plot number of people pre cluster on the 'Would you feel comfortable discussing a mental health disorder with your coworkers?' column
sns.countplot(x='cluster', hue='Would you feel comfortable discussing a mental health disorder with your coworkers?', data=data)
plt.title('Comfort discussing mental health disorder with coworkers per Cluster')
plt.legend(title='Responce categories', loc='upper right', labels=['No response', 'No','Maybe','Yes'])
plt.show()



