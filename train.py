import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np

from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import StandardScaler, RobustScaler, MinMaxScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, confusion_matrix

data = pd.read_csv('dataset\diabetes.csv')

# print(data.head(5))
# data.info()

# sns.histplot(data['Outcome'])
# plt.title ("Outcome distribution")
# plt.savefig("Diebetes Distribution")

# corr_img = sns.heatmap(data.corr(), annot=True)
# plt.savefig('correlation')

target = 'Outcome'

# Set Feature
x = data.drop(target, axis=1)

# Set Target
y = data['Outcome']

x_train, x_test, y_train, y_test = train_test_split(
    x, y, test_size=0.2, random_state=42)

scaler = StandardScaler()  # RobustScaler

# result = scaler.fit_transform(x_train[['Pregnancies']])
# for i, j in zip(x_train['Pregnancies'], result):
#     print('Before {} After {}'.format(i, j))

x_train = scaler.fit_transform(x_train)
x_test = scaler.transform(x_test)

parameters = {
    'n_estimators': [50, 100, 200],
    'criterion': ["gini", "entropy", "log_loss"],
    'max_depth': [None, 5, 10],
    'max_features': ['sqrt', 'log2', None]
}

cls = GridSearchCV(
    estimator=RandomForestClassifier(),
    param_grid=parameters,
    cv=5,
    verbose=1,
    n_jobs=-1
)
cls.fit(x_train, y_train)

# # cls = LogisticRegression()
# cls = RandomForestClassifier()
# cls.fit(x_train, y_train)

y_predict = cls.predict(x_test)
print(cls.best_estimator_)
print(cls.best_score_)
print(cls.best_params_)


# print(classification_report(y_test, y_predict))
# print(confusion_matrix(y_test, y_predict))

# cm = np.array(confusion_matrix(y_test, y_predict, labels=[0, 1]))
# confusion = pd.DataFrame(cm, index=['Not Diabetic', 'Diabetic'], columns=['Predicted Not Diabetic', 'Predicted Diabetic'])
# sns.heatmap(confusion, annot=True, fmt='g')
# plt.savefig('DIABETES.png')
