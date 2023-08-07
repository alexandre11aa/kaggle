# Link do desafio: https://www.kaggle.com/competitions/titanic

# # Ajustando Dados

import pandas as pd

test = pd.read_csv('test.csv')
train = pd.read_csv('train.csv')

train['modify'] = 'train'

test['modify'] = 'test'

df = pd.concat([train, test])

# # Sex

df['Sex'] = df['Sex'].map({'male': 1, 'female': 0})

# # Embarked

df['Embarked'] = df['Embarked'].map({'S': 1, 'C': 2, 'Q': 3})

df[(df['Cabin'].str.contains('b2', case=False)) & (df['Pclass'] == 1) & (df['Sex'] == 0)][['Cabin', 'Embarked', 'Ticket', 'Pclass']]

'''
Levando em consideração os dois tickets dos NAN iguais, as classes 
e embarques semelhantes entre os passageiros das vigésimas cabines
tipo B, suponho com convicção que embarcaram em 1.
'''

df.loc[df['Embarked'].isna(), 'Embarked'] = df.loc[df['Embarked'].isna(), 'Embarked'].fillna(1)

# # Fare

'''
Como os valores de até 20 ocupam mais da metade dos valores que ultrapassam 500
a média será feita dos valores que vão até aí.
'''

mean_fare = int(df[df['Fare'] <= 20][['Fare']].mean())

df.loc[df['Fare'].isna(), 'Fare'] = df.loc[df['Fare'].isna(), 'Fare'].fillna(mean_fare)

# # Age

# Media de idade dos sexos de acordo com grau de sobrevivência e classe social

for i in range(1,4):
    for j in range(2):
        for k in range(2):
            filtro_nan = ((df['Sex'] == k) & (df['Survived'] == j) & (df['Pclass'] == i) & (df['Age'].isna()))
            
            filtro_mean = ((df['Sex'] == k) & (df['Survived'] == j) & (df['Pclass'] == i))
            
            df.loc[filtro_nan, 'Age'] = df.loc[filtro_nan, 'Age'].fillna(int(df[filtro_mean]['Age'].mean()))
            
# Media de idade dos sexos de acordo com classe social

for i in range(1,4):
    for j in range(2):
        for k in range(2):
            filtro_nan = ((df['Sex'] == k) & (df['Pclass'] == i) & (df['Age'].isna()))
            
            filtro_mean = ((df['Sex'] == k) & (df['Pclass'] == i))
            
            df.loc[filtro_nan, 'Age'] = df.loc[filtro_nan, 'Age'].fillna(int(df[filtro_mean]['Age'].mean()))

# # Definindo dados

import numpy as np

from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split

modelo_acuracia = RandomForestClassifier(n_estimators=100, n_jobs=-1, random_state=0)
modelo_test = RandomForestClassifier(n_estimators=100, n_jobs=-1, random_state=0)

df_train_x = df[df['modify'] == 'train'][['Pclass', 'Sex', 'Age', 'SibSp', 'Parch', 'Fare', 'Embarked']]
df_train_y = df[df['modify'] == 'train'][['Survived']]

df_test_x = df[df['modify'] == 'test'][['Pclass', 'Sex', 'Age', 'SibSp', 'Parch', 'Fare', 'Embarked']]
df_test_y = df[df['modify'] == 'test'][['Survived']]

# # Treinando máquina para acurácia

# Separando dados para teste de acurácia

np.random.seed(0)

x_train, x_valid, y_train, y_valid = train_test_split(df_train_x, df_train_y, test_size=0.5)

modelo_acuracia.fit(x_train, y_train)

# Pedindo para máquina predizer Survived dos dados x_valid

p_acuracia = modelo_acuracia.predict(x_valid)

# Média de igualdade entre o teste e os dados separados

np.mean(y_valid['Survived'] == p_acuracia)

# # Treinando máquina

modelo_test.fit(df_train_x, df_train_y)

# Pedindo para máquina predizer Survived dos dados x_valid

p_test = modelo_test.predict(df_test_x)

print(p_test)

# # Ajustando treino para submissão no kaggle

# Desenvolvendo série para csv

result = pd.Series(p_test, index=test['PassengerId'], name='Survived')

result = result.astype(int)

# Exportando para csv

result.to_csv('titanic.csv', header=True)
