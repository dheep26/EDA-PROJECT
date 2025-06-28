import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
df = pd.read_csv("load_dataset.csv")
print(df.shape)
print(df.columns)
print(df.info())
print(df.head())
print(df.isnull().sum())
print(df.duplicated().sum())
print(df.describe())


print(df["ApplicantIncome"].value_counts())


print(df.groupby("Credit_History")["ApplicantIncome"].value_counts(normalize=True))


sns.countplot(data=df, x="ApplicantIncome")
plt.show()


sns.boxplot(data=df, y="ApplicantIncome", x="Education")
plt.show()


numeric_df = df.select_dtypes(include=["number"]).dropna()
sns.heatmap(numeric_df.corr(), annot=True, cmap="coolwarm")
plt.show()


