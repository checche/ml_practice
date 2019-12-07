import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
warnings.filterwarnings('ignore')

#Validation用
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

#Modeling用
import xgboost as xgb

train = pd.read_csv("./data/train.csv")
test = pd.read_csv("./data/test.csv")


#家族連れの数を計算
train['Family'] = train['SibSp'] + train['Parch'] + 1
test['Family'] = test['SibSp'] + test['Parch'] + 1

train["IsAlone"] = train.Family.apply(lambda x: 1 if x == 1 else 0)
test["IsAlone"] = test.Family.apply(lambda x: 1 if x == 1 else 0)

#集計がやりやすいよう、Ageをグルーピング
train["Age"] = train["Age"].fillna(-0.5)
test["Age"] = test["Age"].fillna(-0.5)
bins = [-1, 0, 5, 12, 18, 24, 35, 60, np.inf]
labels = ['Unknown', '1-5', '6-12', '13-18', '19-24', '25-35', '36-60', '60-']
train['AgeGroup'] = pd.cut(train["Age"], bins, labels = labels)
test['AgeGroup'] = pd.cut(test["Age"], bins, labels = labels)

#部屋がある人=1 / 部屋がある人=0となるよう分類
train["CabinBool"] = (train["Cabin"].notnull().astype('int'))
test["CabinBool"] = (test["Cabin"].notnull().astype('int'))

#Cabin内のアルファベットを抽出
train['Cabin'] = train['Cabin'].fillna('Unknown')
train['Deck']=train['Cabin'].str.get(0)
test['Cabin'] = test['Cabin'].fillna('Unknown')
test['Deck']=test['Cabin'].str.get(0)

#集計のためFareをグルーピング
test["Fare"] = test["Fare"].fillna(-2.0)
bins = [-10, -1, 1, 8, 14, 31, np.inf]
labels = ['Unknown', '0-1', '2-8', '9-14', '15-31', '31-']
train['FareGroup'] = pd.cut(train["Fare"], bins, labels = labels)
test['FareGroup'] = pd.cut(test["Fare"], bins, labels = labels)

#NameからTitleを抽出
combine = [train, test]
for dataset in combine:
    dataset['Title'] = dataset.Name.str.extract(' ([A-Za-z]+)\.', expand=False)

#表記ゆれを修正
for dataset in combine:
    dataset['Title'] = dataset['Title'].replace(['Lady', 'Countess','Capt', 'Col', \
 	'Don', 'Dr', 'Major', 'Rev', 'Sir', 'Jonkheer', 'Dona'], 'Other')

    dataset['Title'] = dataset['Title'].replace('Mlle', 'Miss')
    dataset['Title'] = dataset['Title'].replace('Ms', 'Miss')
    dataset['Title'] = dataset['Title'].replace('Mme', 'Mrs')

#Ticket内のアルファベット、長さを抽出
for dataset in combine: 
        dataset['Ticket_Lett'] = dataset['Ticket'].apply(lambda x: str(x)[0])
        dataset['Ticket_Lett'] = dataset['Ticket_Lett'].apply(lambda x: str(x)) 
        dataset['Ticket_Lett'] = np.where((dataset['Ticket_Lett']).isin(['1', '2', '3', 'S', 'P', 'C', 'A']), dataset['Ticket_Lett'], np.where((dataset['Ticket_Lett']).isin(['W', '4', '7', '6', 'L', '5', '8']), '0','0')) 
        dataset['Ticket_Len'] = dataset['Ticket'].apply(lambda x: len(x))

## data preprocessing
#不要なカラムを削除
train = train.drop(['Cabin'], axis = 1)
test = test.drop(['Cabin'], axis = 1)

train = train.drop(['Ticket'], axis = 1)
test = test.drop(['Ticket'], axis = 1)

train = train.drop(['SibSp'], axis = 1)
test = test.drop(['SibSp'], axis = 1)

train = train.drop(['Parch'], axis = 1)
test = test.drop(['Parch'], axis = 1)

train = train.drop(['Name'], axis = 1)
test = test.drop(['Name'], axis = 1)

#EmbarkedのNullをSで埋める
train = train.fillna({"Embarked": "S"})

#欠損値をFare=中央値, Age=平均値で埋める
train['Fare'] = train['Fare'].fillna(train['Fare'].median())
test['Fare'] = test['Fare'].fillna(test['Fare'].median())
train['Age'] = train['Age'].fillna(train['Age'].mean())
test['Age'] = test['Age'].fillna(test['Age'].mean())

#カテゴリカル変数のエンコード
from sklearn import preprocessing

for column in ['Sex', 'Embarked', 'Title', 'AgeGroup', 'FareGroup', 'Deck', 'Ticket_Len', 'Ticket_Lett']:
    le = preprocessing.LabelEncoder()
    le.fit(train[column])
    train[column] = le.transform(train[column])
    
for column in ['Sex', 'Embarked', 'Title', 'AgeGroup', 'FareGroup', 'Deck', 'Ticket_Len', 'Ticket_Lett']:
    le = preprocessing.LabelEncoder()
    le.fit(test[column])
    test[column] = le.transform(test[column])


## modeling
X = train.drop(['Survived', 'PassengerId'], axis=1)
y = train["Survived"]

#学習データを検証用にスプリット
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

#Xgboost用のMatrix形式にデータを変換
dtrain = xgb.DMatrix(data=X_train, label=y_train)
dtest = xgb.DMatrix(data=X_test, label=y_test)

#Hyper Parameterを指定。（本当は工夫すべきだが今回はほぼデフォルト値）
xgb_params = {'max_depth':3, 
              'learning_rate': 0.1, 
              'objective':'binary:logistic',
              'eval_metric': 'logloss'}

# 学習時に用いる検証用データ
evals = [(dtrain, 'train'), (dtest, 'eval')]

# 学習過程を記録するための辞書
evals_result = {}
clf = xgb.train(xgb_params,
                dtrain,
                num_boost_round=1000,
                early_stopping_rounds=100,
                evals=evals,
                evals_result=evals_result,
                )

#検証用データでモデルのAccracyを確認
y_pred_proba = clf.predict(dtest)
y_pred = np.where(y_pred_proba > 0.5, 1, 0)
acc = accuracy_score(y_test, y_pred)
print('Accuracy:', acc)

# 学習の過程を折れ線グラフとしてプロット
train_metric = evals_result['train']['logloss']
plt.plot(train_metric, label='train logloss')
eval_metric = evals_result['eval']['logloss']
plt.plot(eval_metric, label='eval logloss')
plt.grid()
plt.legend()
plt.xlabel('rounds')
plt.ylabel('logloss')

#モデルのFeature Importanceを確認
_, ax = plt.subplots(figsize=(12, 4))
xgb.plot_importance(clf,
                    ax=ax,
                    importance_type='gain',
                    show_values=False)
plt.show()

#Testデータで予測
target = xgb.DMatrix(test.drop('PassengerId', axis=1))
xgb_pred = clf.predict(target, ntree_limit=clf.best_ntree_limit)
#Submitようのデータに変換
tes["Survived"] = np.where(xgb_pred > 0.5, 1, 0)
test[["passengerId", "Survived"]].to_csv(('submit.csv'),index=False)