import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt


data = pd.read_csv('clustered_data.csv')

fig, ax = plt.subplots(1, 3, figsize=(15, 6))

#Plotting Sharing mental health issue with family and friends per cluster, subplot 1
sns.countplot(x='cluster', hue='Sharing mental health issue with family and friends', data=data, palette='Greys', ax=ax[0])
ax[0].set_title('Willingness to Share Mental Health Issues with Family and Friends', fontsize=8)
ax[0].set_xlabel('Cluster')
ax[0].set_ylabel('Count')
ax[0].legend(title='Willingness', loc='upper right', fontsize='small')

#Plotting Mental health effects on work when not treated effectively per cluster, subplot 2
sns.countplot(x='cluster', hue='Mental health effects on work when not treated effectively', data=data, palette='viridis', ax=ax[1])
ax[1].set_title('Mental Health Effects on Work when not Treated Effectively', fontsize=8)
ax[1].set_xlabel('Cluster')
ax[1].set_ylabel('Count')
ax[1].legend(title='Response', loc='upper right', fontsize='small')

#Plotting Mental health effects on work when treated effectively per cluster, subplot 3
sns.countplot(x='cluster', hue='Mental health effects on work when treated effectively', data=data, palette='viridis', ax=ax[2])
ax[2].set_title('Mental Health Effects on Work when Treated Effectively', fontsize=8)
ax[2].set_xlabel('Cluster')
ax[2].set_ylabel('Count')
ax[2].legend(title='Response', loc='upper right', fontsize='small')

plt.tight_layout()
plt.show()

# Set the visual theme for better aesthetics
sns.set_theme(style="whitegrid")

# Plot average age per cluster
plt.figure(figsize=(10, 6))
sns.barplot(x='cluster', y='What is your age?', data=data, palette='viridis')
plt.title('Average Age per Cluster')
plt.xlabel('Cluster')
plt.ylabel('Average Age')
plt.show()

# Plot number of employees per cluster
plt.figure(figsize=(10, 6))
sns.barplot(x='cluster', y='How many employees does your company or organization have?', data=data, palette='coolwarm')
plt.title('Average Number of Employees per Cluster')
plt.xlabel('Cluster')
plt.ylabel('Average Number of Employees')
plt.show()

# Plot knowledge of mental health care options per cluster
plt.figure(figsize=(10, 6))
sns.countplot(x='cluster', hue='Do you know the options for mental health care available under your employer-provided coverage?', data=data, palette='Set1')
plt.title('Knowledge of Mental Health Care Options per Cluster')
plt.xlabel('Cluster')
plt.ylabel('Count')
plt.legend(title='Knowledge')
plt.show()


# Plot employer seriousness about mental health vs physical health per cluster
plt.figure(figsize=(10, 6))
sns.countplot(x='cluster', hue='Do you feel that your employer takes mental health as seriously as physical health?', data=data, palette='Set2')
plt.title('Employer Seriousness about Mental Health vs Physical Health per Cluster')
plt.xlabel('Cluster')
plt.ylabel('Count')
plt.legend(title='Seriousness')
plt.show()

# Plot previous employer's discussion of mental health per cluster
plt.figure(figsize=(10, 6))
sns.countplot(x='cluster', hue='Did your previous employers ever formally discuss mental health (as part of a wellness campaign or other official communication)?', data=data, palette='Dark2')
plt.title('Discussion of Mental Health by Previous Employers per Cluster')
plt.xlabel('Cluster')
plt.ylabel('Count')
plt.legend(title='Discussion')
plt.show()

# Plot negative consequences of discussing mental health issues with previous employers per cluster
plt.figure(figsize=(10, 6))
sns.countplot(x='cluster', hue='Do you think that discussing a mental health disorder with previous employers would have negative consequences?', data=data, palette='Blues')
plt.title('Negative Consequences of Discussing Mental Health Disorder per Cluster')
plt.xlabel('Cluster')
plt.ylabel('Count')
plt.legend(title='Negative Consequences')
plt.show()

# Plot negative consequences of discussing physical health issues with previous employers per cluster
plt.figure(figsize=(10, 6))
sns.countplot(x='cluster', hue='Do you think that discussing a physical health issue with previous employers would have negative consequences?', data=data, palette='Greens')
plt.title('Negative Consequences of Discussing Physical Health Issue per Cluster')
plt.xlabel('Cluster')
plt.ylabel('Count')
plt.legend(title='Negative Consequences')
plt.show()

# Plot perceptions of career impact due to mental health issues per cluster
plt.figure(figsize=(10, 6))
sns.countplot(x='cluster', hue='Do you feel that being identified as a person with a mental health issue would hurt your career?', data=data, palette='Oranges')
plt.title('Perceptions of Career Impact due to Mental Health Issues per Cluster')
plt.xlabel('Cluster')
plt.ylabel('Count')
plt.legend(title='Career Impact')
plt.show()

# Plot unsupportive responses to mental health issues in the workplace per cluster
plt.figure(figsize=(10, 6))
sns.countplot(x='cluster', hue='Have you observed or experienced an unsupportive or badly handled response to a mental health issue in your current or previous workplace?', data=data, palette='Reds')
plt.title('Unsupportive Responses to Mental Health Issues per Cluster')
plt.xlabel('Cluster')
plt.ylabel('Count')
plt.legend(title='Unsupportive Responses')
plt.show()

# Plot family history of mental illness per cluster
plt.figure(figsize=(10, 6))
sns.countplot(x='cluster', hue='Do you have a family history of mental illness?', data=data, palette='Purples')
plt.title('Family History of Mental Illness per Cluster')
plt.xlabel('Cluster')
plt.ylabel('Count')
plt.legend(title='Family History')
plt.show()

# Plot current mental health disorder status per cluster
plt.figure(figsize=(10, 6))
sns.countplot(x='cluster', hue='Do you currently have a mental health disorder?', data=data, palette='cividis')
plt.title('Current Mental Health Disorder Status per Cluster')
plt.xlabel('Cluster')
plt.ylabel('Count')
plt.legend(title='Mental Health Disorder')
plt.show()

# Plot remote work status per cluster
plt.figure(figsize=(10, 6))
sns.countplot(x='cluster', hue='Do you work remotely?', data=data, palette='BuPu')
plt.title('Remote Work Status per Cluster')
plt.xlabel('Cluster')
plt.ylabel('Count')
plt.legend(title='Remote Work')
plt.show()

# Plot ease of asking for mental health leave per cluster
plt.figure(figsize=(10, 6))
sns.countplot(x='cluster', hue='How easy is it to ask for mental health leave?', data=data, palette='YlOrBr')
plt.title('Ease of Asking for Mental Health Leave per Cluster')
plt.xlabel('Cluster')
plt.ylabel('Count')
plt.legend(title='Ease of Asking')
plt.show()

# Plot sharing mental health issues with family and friends per cluster
plt.figure(figsize=(10, 6))
sns.countplot(x='cluster', hue='Sharing mental health issue with family and friends', data=data, palette='Greys')
plt.title('Sharing Mental Health Issues with Family and Friends per Cluster')
plt.xlabel('Cluster')
plt.ylabel('Count')
plt.legend(title='Sharing Mental Health Issues')
plt.show()