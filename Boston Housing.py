import pandas as pd
import seaborn as sns
import matplotlib.pyplot as pyplot
import scipy.stats
import statsmodels.api as sm
from statsmodels.formula.api import ols

boston_df=pd.read_csv('https://cf-courses-data.s3.us.cloud-object-storage.appdomain.cloud/IBMDeveloperSkillsNetwork-ST0151EN-SkillsNetwork/labs/boston_housing.csv')

# Task 1

# Display first 5 rows
print(boston_df.head())

# Display data types
print(boston_df.dtypes)

# Check for missing values
print(boston_df.isnull().sum())

# Check for duplicate rows
print(boston_df.duplicated().sum())

# Correlation matrix
print(boston_df.corr())

# Task 2

# Descriptive statistics
print(boston_df.describe())

pyplot.figure()
pyplot.boxplot(boston_df['MEDV'])
pyplot.title("Boxplot of MEDV")
pyplot.ylabel("Median Value")
pyplot.show()

boston_df['CHAS'].value_counts().plot(kind='bar')
pyplot.title("Count of Houses Bounded by Charles River")
pyplot.xlabel("CHAS (0 = No, 1 = Yes)")
pyplot.ylabel("Count")
pyplot.show()

boston_df['AGE_GROUP'] = pd.cut(boston_df['AGE'],
                                 bins=[0, 35, 70, 100],
                                 labels=['Young', 'Middle', 'Old'])

sns.boxplot(x='AGE_GROUP', y='MEDV', data=boston_df)
pyplot.title("MEDV vs AGE Group")
pyplot.show()

pyplot.scatter(boston_df['INDUS'], boston_df['NOX'])
pyplot.xlabel("INDUS")
pyplot.ylabel("NOX")
pyplot.title("NOX vs INDUS")
pyplot.show()

pyplot.hist(boston_df['PTRATIO'])
pyplot.title("Histogram of PTRATIO")
pyplot.xlabel("Pupil-Teacher Ratio")
pyplot.ylabel("Frequency")
pyplot.show()

# Task 3

scipy.stats.levene(
    boston_df[boston_df['CHAS'] == 0]['MEDV'],
    boston_df[boston_df['CHAS'] == 1]['MEDV']
)

scipy.stats.ttest_ind(
    boston_df[boston_df['CHAS'] == 0]['MEDV'],
    boston_df[boston_df['CHAS'] == 1]['MEDV'],
    equal_var=True
)

model = ols('MEDV ~ AGE_GROUP', data=boston_df).fit()
anova_table = sm.stats.anova_lm(model, typ=2)
anova_table

print(anova_table)

scipy.stats.pearsonr(boston_df['NOX'], boston_df['INDUS'])

X = sm.add_constant(boston_df['DIS'])
y = boston_df['MEDV']

model = sm.OLS(y, X).fit()
model.summary()