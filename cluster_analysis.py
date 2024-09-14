import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from scipy.stats import f_oneway


data = pd.read_csv('clustered_data.csv')


fig, ax = plt.subplots(1, 2, figsize=(15, 6))
fig.suptitle('Effects of Mental Health issues on work per Cluster', fontsize=10)

#Plotting 'Do you think that team members/co-workers would view you more negatively if they knew you suffered from a mental health issue?' per cluster
sns.countplot(x='cluster', hue='Do you think that team members/co-workers would view you more negatively if they knew you suffered from a mental health issue?', data=data, palette='viridis', ax=ax[0])
ax[0].set_title('Would co-workers view you negatively if they knew you had a mental health issue?', fontsize=8)
ax[0].set_xlabel('Cluster')
ax[0].set_ylabel('Count')
handles, labels = ax[0].get_legend_handles_labels()
custom_labels = ['No','Maybe','Yes']
ax[0].legend(handles, custom_labels, title='Perception', loc='upper right', fontsize='small')

#Plotting 'Do you feel that being identified as a person with a mental health issue would hurt your career?' per cluster
sns.countplot(x='cluster', hue='Do you feel that being identified as a person with a mental health issue would hurt your career?', data=data, palette='viridis', ax=ax[1])  
ax[1].set_title('Would being identified with a mental health issue hurt your career?', fontsize=8)
ax[1].set_xlabel('Cluster')
ax[1].set_ylabel('Count')
handles, labels = ax[1].get_legend_handles_labels()
custom_labels = ['No','Maybe','Yes']
ax[1].legend(handles, custom_labels, title='Responses', loc='upper right', fontsize='small')

plt.tight_layout()
plt.show()



# Plotting the averages age per cluster
fig, ax = plt.subplots(2, 3, figsize=(15, 6))
#Assign the name  of the figure
fig.suptitle('Personal Mental Health & Demographics per cluster', fontsize=10)

# Plotting the average age per cluster from the merged data, subplot 1
sns.barplot(x='cluster', y='What is your age?', data=data, palette='viridis', ax=ax[0, 0])
ax[0, 0].set_title('Average Age', fontsize=8)
ax[0, 0].set_xlabel('Cluster')
ax[0, 0].set_ylabel('Average Age')


# Plotting the average number of employees per cluster, subplot 2
sns.barplot(x='cluster', y='How many employees does your company or organization have?', data=data, palette='viridis', ax=ax[0, 1])
ax[0, 1].set_title('Average company size', fontsize=8)
ax[0, 1].set_xlabel('Cluster')
ax[0, 1].set_ylabel('Average Number of Employees')


# Plotting 'do you work remotely?' per cluster, subplot 3
sns.countplot(x='cluster', hue='Do you work remotely?', data=data, palette='viridis', ax=ax[0, 2])
ax[0, 2].set_title('Remote Work Status', fontsize=8)
ax[0, 2].set_xlabel('Cluster')
ax[0, 2].set_ylabel('Count')
handles, labels = ax[0, 2].get_legend_handles_labels()
custom_labels = ['Never','Sometimes','Always']
ax[0, 2].legend(handles, custom_labels, title='Working remotely', loc='upper right', fontsize='small')

#Plotting Do you currently have a mental health disorder? per cluster
sns.countplot(x='cluster', hue='Do you currently have a mental health disorder?', data=data, palette='viridis', ax=ax[1, 0])
ax[1, 0].set_title('Do you currently have a mental health disorder?', fontsize=8)
ax[1, 0].set_xlabel('Cluster')
ax[1, 0].set_ylabel('Count')
handles, labels = ax[1, 0].get_legend_handles_labels()
custom_labels = ['No','Maybe','Yes']
ax[1, 0].legend(handles, custom_labels, title='Current Mental-Health state', loc='upper right', fontsize='small')

#Plotting 'Do you have a family history of mental illness?' per cluster, subplot 5
sns.countplot(x='cluster', hue='Do you have a family history of mental illness?', data=data, palette='viridis', ax=ax[1, 1])
ax[1, 1].set_title('Family History of Mental Illness', fontsize=8)
ax[1, 1].set_xlabel('Cluster')
ax[1, 1].set_ylabel('Count')
handles, labels = ax[1, 1].get_legend_handles_labels()
custom_labels = ["I don't know",'No','Yes']
ax[1, 1].legend(handles, custom_labels, title='Family History', loc='upper right', fontsize='small')

#Plotting ' Would you be willing to bring up a physical health issue with a potential employer in an interview?' per cluster, subplot 6
sns.countplot(x='cluster', hue='Would you be willing to bring up a physical health issue with a potential employer in an interview?', data=data, palette='viridis', ax=ax[1, 2])
ax[1, 2].set_title('Willingness to Discuss Physical Health Issues in an Interview', fontsize=8)
ax[1, 2].set_xlabel('Cluster')
ax[1, 2].set_ylabel('Count')
handles, labels = ax[1, 2].get_legend_handles_labels()
custom_labels = ['No','Maybe','Yes']
ax[1, 2].legend(handles, custom_labels, title='Willingness', loc='upper right', fontsize='small')

plt.tight_layout()
plt.show()




# Plot the effects of discussing mental health and physical health issues.
fig, ax = plt.subplots(3, 2, figsize=(15, 10))
fig.suptitle('Effects of Discussing and sharing Mental Health and Physical Health Issues per Cluster', fontsize=10)

#Plotting 'Do you think that discussing a mental health disorder with previous employers would have negative consequences?' per cluster
sns.countplot(x='cluster', hue='Do you think that discussing a mental health disorder with previous employers would have negative consequences?', data=data, palette='viridis', ax=ax[0, 0])
ax[0, 0].set_title('Would discussing a Mental health issue with ex-employers have negative consequences?', fontsize=8)
ax[0, 0].set_xlabel('Cluster')
ax[0, 0].set_ylabel('Count')
handles, labels = ax[0, 0].get_legend_handles_labels()
custom_labels = ["I don't know",'No','Maybe','Yes']
ax[0, 0].legend(handles, custom_labels, title='Consequences', loc='upper right', fontsize='small')

#Plotting 'Do you think that discussing a physical health issue with previous employers would have negative consequences?' per cluster
sns.countplot(x='cluster', hue='Do you think that discussing a physical health issue with previous employers would have negative consequences?', data=data, palette='viridis', ax=ax[0, 1])
ax[0, 1].set_title('Would discussing physical health issues with ex-employers have negative consequences?', fontsize=8)
ax[0, 1].set_xlabel('Cluster')
ax[0, 1].set_ylabel('Count')
handles, labels = ax[0, 1].get_legend_handles_labels()
custom_labels = ["I don't know",'No','Maybe','Yes']
ax[0, 1].legend(handles, custom_labels, title='Consequences', loc='upper right', fontsize='small')

# Plotting How willing would you be to share with friends and family that you have a mental illness per cluster, subplot 4
sns.countplot(x='cluster', hue='Sharing mental health issue with family and friends', data=data, palette='viridis', ax=ax[1, 0])
ax[1, 0].set_title('Willingness to Share Mental Illness with Friends and Family', fontsize=8)
ax[1, 0].set_xlabel('Cluster')
ax[1, 0].set_ylabel('Count')
handles, labels = ax[1, 0].get_legend_handles_labels()
custom_labels = ['Not open','N/A','Somewhat not open','Somewhat Open','Very open']
ax[1, 0].legend(handles, custom_labels, title='Willingness', loc='upper right', fontsize='small')

# Plotting Have you observed or experienced an unsupportive or badly handled response to a mental health issue in your current or previous workplace? per cluster, subplot 5
sns.countplot(x='cluster', hue='Have you observed or experienced an unsupportive or badly handled response to a mental health issue in your current or previous workplace?', data=data, palette='viridis', ax=ax[1, 1])
ax[1, 1].set_title('Have you received unsupportive Responses to Mental Health Issues', fontsize=8)
ax[1, 1].set_xlabel('Cluster')
ax[1, 1].set_ylabel('Count')
handles, labels = ax[1, 1].get_legend_handles_labels()
custom_labels = ['No response','No','Maybe','Yes']
ax[1, 1].legend(handles, custom_labels, title='Responses', loc='upper right', fontsize='small')

#Plotting 'Did you hear of or observe negative consequences for co-workers with mental health issues in your current or previous workplace?' per cluster
sns.countplot(x='cluster', hue='Did you hear of or observe negative consequences for co-workers with mental health issues in your previous workplaces?', data=data, palette='viridis', ax=ax[2, 0])
ax[2, 0].set_title('Observing Negative Consequences for Co-workers with Mental Health Issues', fontsize=8)
ax[2, 0].set_xlabel('Cluster')
ax[2, 0].set_ylabel('Count')
handles, labels = ax[2, 0].get_legend_handles_labels()
custom_labels = ['No response','None of them','Some of them','Yes, all of them']
ax[2, 0].legend(handles, custom_labels, title='Observations', loc='upper right', fontsize='small')

#Plotting 'Was your anonymity protected if you chose to take advantage of mental health or substance abuse treatment resources with previous employers?' per cluster
sns.countplot(x='cluster', hue='Was your anonymity protected if you chose to take advantage of mental health or substance abuse treatment resources with previous employers?', data=data, palette='viridis', ax=ax[2, 1])
ax[2, 1].set_title('Was there Anonymity Protection for Mental Health Treatment?', fontsize=8)
ax[2, 1].set_xlabel('Cluster')
ax[2, 1].set_ylabel('Count')
handles, labels = ax[2, 1].get_legend_handles_labels()
custom_labels = ["I don't know",'No','Sometimes','Yes']
ax[2, 1].legend(handles, custom_labels, title='Anonymity', loc='upper right', fontsize='small')

plt.tight_layout()
plt.show()




fig, ax = plt.subplots(2, 3, figsize=(15, 6))
fig.suptitle('Mental Health Awareness and Support per cluster', fontsize=10)

#Plotting 'Do you know know the options for mental health care available under your employer-provided coverage?' per cluster
sns.countplot(x='cluster', hue='Do you know the options for mental health care available under your employer-provided coverage?', data=data, palette='viridis', ax=ax[0, 0])
ax[0, 0].set_title('Do you have Knowledge of Mental Health Care Options?', fontsize=8)
ax[0, 0].set_xlabel('Cluster')
ax[0, 0].set_ylabel('Count')
handles, labels = ax[0, 0].get_legend_handles_labels()
custom_labels = ['No response','No',"I'm not sure",'Yes']
ax[0, 0].legend(handles, custom_labels, title='Knowledge', loc='upper right', fontsize='small')

#Plotting 'Does your employer offer resources to learn more about mental health concerns and options for seeking help?' per cluster
sns.countplot(x='cluster', hue='Does your employer offer resources to learn more about mental health concerns and options for seeking help?', data=data, palette='viridis', ax=ax[0, 1])
ax[0, 1].set_title('Does Employer offer Resources for Mental Health Concerns', fontsize=8)
ax[0, 1].set_xlabel('Cluster')
ax[0, 1].set_ylabel('Count')
handles, labels = ax[0, 1].get_legend_handles_labels()
custom_labels = ["I don't know",'No','Yes']
ax[0, 1].legend(handles, custom_labels, title='Responses', loc='upper right', fontsize='small')

#Plotting 'Do you feel that your employer takes mental health as seriously as physical health?' per cluster
sns.countplot(x='cluster', hue='Do you feel that your employer takes mental health as seriously as physical health?', data=data, palette='viridis', ax=ax[1, 0])
ax[1, 0].set_title('Does employer take Mental Health seriously vs physical health', fontsize=8)
ax[1, 0].set_xlabel('Cluster')
ax[1, 0].set_ylabel('Count')
handles, labels = ax[1, 0].get_legend_handles_labels()
custom_labels = ["I don't know",'No','Yes']
ax[1, 0].legend(handles, custom_labels, title='Responses', loc='upper right', fontsize='small')

#Plotting 'Did your previous employers provide resources to learn more about mental health issues and how to seek help?' per cluster
sns.countplot(x='cluster', hue='Did your previous employers provide resources to learn more about mental health issues and how to seek help?', data=data, palette='viridis', ax=ax[1, 1])
ax[1, 1].set_title('Did previous Employer offer Resources to learn about Mental Health', fontsize=8)
ax[1, 1].set_xlabel('Cluster')
ax[1, 1].set_ylabel('Count')
handles, labels = ax[1, 1].get_legend_handles_labels()
custom_labels = ['No response', 'None did', 'Some did', 'Yes, all did']
ax[1, 1].legend(handles, custom_labels, title='Responses', loc='upper right', fontsize='small')

#Plotting 'Did you feel that your previous employers took mental health as seriously as physical health?' per cluster
sns.countplot(x='cluster', hue='Did you feel that your previous employers took mental health as seriously as physical health?', data=data, palette='viridis', ax=ax[1, 2])
ax[1, 2].set_title('Did your ex-employers take mental health as seriously as physical health?', fontsize=8)
ax[1, 2].set_xlabel('Cluster')
ax[1, 2].set_ylabel('Count')
handles, labels = ax[1, 2].get_legend_handles_labels()
custom_labels = ["I don't know",'None did','Some did','Yes, all did']
ax[1, 2].legend(handles, custom_labels, title='Responses', loc='upper right', fontsize='small')

plt.tight_layout()
plt.show()


#Plotting values in each cluster in one graph and show the count inside the bar
sns.countplot(x='cluster', data=data, palette='viridis')
plt.title('Number of Participants in each Cluster', fontsize=10)
plt.xlabel('Cluster')
plt.ylabel('Count')
# Add the count of each cluster inside the bar
for i in range(2):
    count = data['cluster'].value_counts().values[i]
    plt.text(i, count, str(count), ha='center', va='bottom')
plt.show()

fig, ax = plt.subplots(1, 2, figsize=(15, 6))
fig.suptitle('Deep dive into Current Mental health and Family history', fontsize=10)

#Plotting 'Do you currently have a mental health disorder?' only choosing answer 2 per cluster
sns.countplot(x='cluster', hue='Do you currently have a mental health disorder?', data=data[data['Do you currently have a mental health disorder?'] == 2], palette='viridis', ax=ax[0])
ax[0].set_title('Current Mental Health Disorder', fontsize=8)
ax[0].set_xlabel('Cluster')
ax[0].set_ylabel('Count')
handles, labels = ax[0].get_legend_handles_labels()
custom_labels = ['Yes']
ax[0].legend(handles, custom_labels, title='Current Mental Health Disorder', loc='upper right', fontsize='small')

#Plotting 'Do you have a family history of mental illness?' only choosing answer 2 per cluster
sns.countplot(x='cluster', hue='Do you have a family history of mental illness?', data=data[data['Do you have a family history of mental illness?'] == 2], palette='viridis', ax=ax[1])
ax[1].set_title('Family History of Mental Illness', fontsize=8)
ax[1].set_xlabel('Cluster')
ax[1].set_ylabel('Count')
handles, labels = ax[1].get_legend_handles_labels()
custom_labels = ['Yes']
ax[1].legend(handles, custom_labels, title='Family History', loc='upper right', fontsize='small')

plt.tight_layout()
plt.show()


#Running ANOVA test to check if there is a significant difference in the average age of the clusters
cluster_0 = data[data['cluster'] == 0]['What is your age?']
cluster_1 = data[data['cluster'] == 1]['What is your age?']
f_oneway(cluster_0, cluster_1)
print('Average age:', f_oneway(cluster_0, cluster_1))

#Running ANOVA test to check if there is a significant difference in the average company size of the clusters
cluster_0 = data[data['cluster'] == 0]['How many employees does your company or organization have?']
cluster_1 = data[data['cluster'] == 1]['How many employees does your company or organization have?']
f_oneway(cluster_0, cluster_1)
print('Average company size:', f_oneway(cluster_0, cluster_1))

#Running ANOVA test to check if there is a significant difference in 'Do you have a family history of mental illness?' of the clusters
cluster_0 = data[data['cluster'] == 0]['Do you have a family history of mental illness?']
cluster_1 = data[data['cluster'] == 1]['Do you have a family history of mental illness?']
f_oneway(cluster_0, cluster_1)
print('Family history of mental illness:', f_oneway(cluster_0, cluster_1))

#Running ANOVA test to check if there is a significant difference in 'Do you currently have a mental health disorder?' of the clusters     
cluster_0 = data[data['cluster'] == 0]['Do you currently have a mental health disorder?']
cluster_1 = data[data['cluster'] == 1]['Do you currently have a mental health disorder?']
f_oneway(cluster_0, cluster_1)
print('Current mental health disorder:', f_oneway(cluster_0, cluster_1))

#Running ANOVA test to check if there is a significant difference in 'Did your previous employers provide resources to learn more about mental health issues and how to seek help?' of the clusters
cluster_0 = data[data['cluster'] == 0]['Did your previous employers provide resources to learn more about mental health issues and how to seek help?']
cluster_1 = data[data['cluster'] == 1]['Did your previous employers provide resources to learn more about mental health issues and how to seek help?']
f_oneway(cluster_0, cluster_1)
print('Previous Employer Resources:', f_oneway(cluster_0, cluster_1))


#read the data with cluster labels
clustered_data = pd.read_csv('clustered_data.csv')

#Add the cluster labels to the original data
data_file = 'mental-heath-in-tech-2016_20161114.csv'
data = pd.read_csv(data_file)
data['cluster'] = clustered_data['cluster']

#Filter the data based on the age of the participants and remove null values
data = data[data['What is your age?'].notnull() & (data['What is your age?'] <= 80)]

#Plot the average age of each cluster
grouped_data = data.groupby('cluster')
average_age = grouped_data['What is your age?'].mean()
average_age.plot(kind='bar')
plt.title('Average Age of Each Cluster',size=10)
plt.xlabel('Cluster')
plt.xticks(rotation=0)
plt.ylabel('Average Age')
plt.ylim(5, 40)
for i, v in enumerate(average_age):
    plt.text(i, v + 1, str(round(v, 2)), ha='center')
plt.show()