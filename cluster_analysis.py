import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt


data = pd.read_csv('clustered_data.csv')

fig, ax = plt.subplots(1, 4, figsize=(15, 6))
fig.suptitle('Effects of Mental Health issues on work per Cluster', fontsize=10)

#Plotting Mental health effects on work when not treated effectively per cluster, subplot 2
sns.countplot(x='cluster', hue='Mental health effects on work when not treated effectively', data=data, palette='viridis', ax=ax[0])
ax[0].set_title('Mental Health Effects on Work when not Treated Effectively', fontsize=8)
ax[0].set_xlabel('Cluster')
ax[0].set_ylabel('Count')
ax[0].legend(title='Response', loc='upper right', fontsize='small')

#Plotting Mental health effects on work when treated effectively per cluster, subplot 3
sns.countplot(x='cluster', hue='Mental health effects on work when treated effectively', data=data, palette='viridis', ax=ax[1])
ax[1].set_title('Mental Health Effects on Work when Treated Effectively', fontsize=8)
ax[1].set_xlabel('Cluster')
ax[1].set_ylabel('Count')
ax[1].legend(title='Response', loc='upper right', fontsize='small')

#Plotting 'Do you feel that being identified as a person with a mental health issue would hurt your career?' per cluster
sns.countplot(x='cluster', hue='Do you feel that being identified as a person with a mental health issue would hurt your career?', data=data, palette='viridis', ax=ax[2])
ax[2].set_title('Career Impacts due to Mental Health Issues per Cluster', fontsize=8)
ax[2].set_xlabel('Cluster')
ax[2].set_ylabel('Count')
ax[2].legend(title='Career Impact', loc='upper right', fontsize='small')

#Plotting 'How easy is it to ask for mental health leave?' per cluster
sns.countplot(x='cluster', hue='How easy is it to ask for mental health leave?', data=data, palette='viridis', ax=ax[3])
ax[3].set_title('Ease of Asking for Mental Health Leave per Cluster', fontsize=8)
ax[3].set_xlabel('Cluster')
ax[3].set_ylabel('Count')
ax[3].legend(title='Ease of Asking', loc='upper right', fontsize='small')

plt.tight_layout()
plt.show()


# Plotting the averages age per cluster
fig, ax = plt.subplots(2, 2, figsize=(15, 6))
#Assign the name  of the figure
fig.suptitle('Average Age and Number of Employees per Cluster', fontsize=10)

# Plotting the average age per cluster, subplot 1
sns.barplot(x='cluster', y='What is your age?', data=data, palette='viridis', ax=ax[0, 0])
ax[0, 0].set_title('Average Age per Cluster', fontsize=8)
ax[0, 0].set_xlabel('Cluster')
ax[0, 0].set_ylabel('Average Age')

# Plotting the average number of employees per cluster, subplot 2
sns.barplot(x='cluster', y='How many employees does your company or organization have?', data=data, palette='viridis', ax=ax[0, 1])
ax[0, 1].set_title('Average company size per Cluster', fontsize=8)
ax[0, 1].set_xlabel('Cluster')
ax[0, 1].set_ylabel('Average Number of Employees')

# Plotting 'do you work remotely?' per cluster, subplot 3
sns.countplot(x='cluster', hue='Do you work remotely?', data=data, palette='viridis', ax=ax[1, 0])
ax[1, 0].set_title('Remote Work Status per Cluster', fontsize=8)
ax[1, 0].set_xlabel('Cluster')
ax[1, 0].set_ylabel('Count')
ax[1, 0].legend(title='Remote Work', loc='upper right', fontsize='small')

#Plotting 'What country do you work in?' per cluster, subplot 4
sns.countplot(x='cluster', hue='What country do you work in?', data=data, palette='viridis', ax=ax[1, 1])
ax[1, 1].set_title('Country of Work per Cluster', fontsize=8)
ax[1, 1].set_xlabel('Cluster')
ax[1, 1].set_ylabel('Count')
ax[1, 1].legend(title='Country', loc='upper right', fontsize='small')


plt.tight_layout()
plt.show()


# Plot the effects of discussing mental health and physical health issues.
fig, ax = plt.subplots(3, 2, figsize=(15, 10))
fig.suptitle('Effects of Discussing and sharing Mental Health and Physical Health Issues per Cluster', fontsize=10)

# Plotting the effects of discussing mental health issues with ex-employers, subplot 1
sns.countplot(x='cluster', hue='Did your previous employers ever formally discuss mental health (as part of a wellness campaign or other official communication)?', data=data, palette='viridis', ax=ax[0, 0])
ax[0, 0].set_title('Discussion of Mental Health by Previous Employers per Cluster', fontsize=8)
ax[0, 0].set_xlabel('Cluster')
ax[0, 0].set_ylabel('Count')
ax[0, 0].legend(title='Responses', loc='upper right', fontsize='small')

# Plotting the Negative Consequences of Discussing Mental Health Disorder per Cluster , subplot 2
sns.countplot(x='cluster', hue='Do you think that discussing a mental health disorder with previous employers would have negative consequences?', data=data, palette='viridis', ax=ax[0, 1])
ax[0, 1].set_title('Negative Consequences of Discussing Mental Health Disorder per Cluster', fontsize=8)
ax[0, 1].set_xlabel('Cluster')
ax[0, 1].set_ylabel('Count')
ax[0, 1].legend(title='Consequences', loc='upper right', fontsize='small')

# Plotting the effects of discussing physical health issues with previous employers would have negative consequences, subplot 3
sns.countplot(x='cluster', hue='Do you think that discussing a physical health issue with previous employers would have negative consequences?', data=data, palette='viridis', ax=ax[1, 0])
ax[1, 0].set_title('Negative Consequences of Discussing Physical Health Issue per Cluster', fontsize=8)
ax[1, 0].set_xlabel('Cluster')
ax[1, 0].set_ylabel('Count')
ax[1, 0].legend(title='Responses', loc='upper right', fontsize='small')

# Plotting How willing would you be to share with friends and family that you have a mental illness per cluster, subplot 4
sns.countplot(x='cluster', hue='Sharing mental health issue with family and friends', data=data, palette='viridis', ax=ax[1, 1])
ax[1, 1].set_title('Willingness to Share Mental Illness with Friends and Family per Cluster', fontsize=8)
ax[1, 1].set_xlabel('Cluster')
ax[1, 1].set_ylabel('Count')
ax[1, 1].legend(title='Willingness', loc='upper right', fontsize='small')

# Plotting Have you observed or experienced an unsupportive or badly handled response to a mental health issue in your current or previous workplace? per cluster, subplot 5
sns.countplot(x='cluster', hue='Have you observed or experienced an unsupportive or badly handled response to a mental health issue in your current or previous workplace?', data=data, palette='viridis', ax=ax[2, 0])
ax[2, 0].set_title('Unsupportive Responses to Mental Health Issues per Cluster', fontsize=8)
ax[2, 0].set_xlabel('Cluster')
ax[2, 0].set_ylabel('Count')
ax[2, 0].legend(title='Unsupportive Responses', loc='upper right', fontsize='small')


plt.tight_layout()
plt.show()


fig, ax = plt.subplots(2, 2, figsize=(15, 6))
fig.suptitle('Understanding employees state of mind', fontsize=10)

# Plotting Do you know the options for mental health care available under your employer-provided coverage?
sns.countplot(x='cluster', hue='Do you know the options for mental health care available under your employer-provided coverage?', data=data, palette='viridis', ax=ax[0, 0])
ax[0, 0].set_title('Knowledge of Mental Health Care Options per Cluster', fontsize=8)
ax[0, 0].set_xlabel('Cluster')
ax[0, 0].set_ylabel('Count')
ax[0, 0].legend(title='Knowledge', loc='upper right', fontsize='small')

#Plotting Mental health history in the family per cluster
sns.countplot(x='cluster', hue='Do you have a family history of mental illness?', data=data, palette='viridis', ax=ax[0, 1])
ax[0, 1].set_title('Family History of Mental Illness per Cluster', fontsize=8)
ax[0, 1].set_xlabel('Cluster')
ax[0, 1].set_ylabel('Count')
ax[0, 1].legend(title='Family History', loc='upper right', fontsize='small')

#Plotting Do you currently have a mental health disorder? per cluster
sns.countplot(x='cluster', hue='Do you currently have a mental health disorder?', data=data, palette='viridis', ax=ax[1, 0])
ax[1, 0].set_title('Current Mental Health Disorder Status per Cluster', fontsize=8)
ax[1, 0].set_xlabel('Cluster')
ax[1, 0].set_ylabel('Count')
ax[1, 0].legend(title='Mental Health Disorder', loc='upper right', fontsize='small')


plt.tight_layout()
plt.show()



# Set the visual theme for better aesthetics
sns.set_theme(style="whitegrid")



