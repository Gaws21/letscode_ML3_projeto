import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from collections import Counter
from imblearn.combine import SMOTEENN
from imblearn.combine import SMOTETomek
from imblearn.over_sampling import SMOTE
from imblearn.over_sampling import RandomOverSampler
from imblearn.under_sampling import RandomUnderSampler
from imblearn.under_sampling import TomekLinks
from imblearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.feature_selection import RFE
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report
from sklearn.metrics import ConfusionMatrixDisplay
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import RandomizedSearchCV
from sklearn.model_selection import StratifiedKFold
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler 
from sklearn.preprocessing import OneHotEncoder
from sklearn.preprocessing import StandardScaler
from sklearn.tree import DecisionTreeClassifier


# ANÁLISE E EXPLORAÇÃO DOS DADOS

#ler os datasets
train_data_read = pd.read_csv('train_data.csv')
test_data_read = pd.read_csv('test_data.csv')

drop_columns = [
     'loading'
    ,'measurement_3'
    ,'measurement_4'
    ,'measurement_5'
    ,'measurement_6'
    ,'measurement_7'
    ,'measurement_8'
    ,'measurement_9'
    ,'measurement_13'
]

#DataFrame de treino resultante após exclusão das features com baixa correlação
train_data_without_low_correlation_features = train_data_read.drop(axis=1, columns=drop_columns, inplace=False)

#DataFrame de teste resultante após exclusão das features com baixa correlação
test_data_without_low_correlation_features = test_data_read.drop(axis=1, columns=drop_columns, inplace=False)

#Setando as features de treino
X_train = train_data_without_low_correlation_features.drop(columns=['failure'])

#Setando o target de treino
y_train = train_data_without_low_correlation_features['failure']

#Setando as features de teste
X_test = test_data_without_low_correlation_features.drop(columns=['failure'])

#Setando o target de teste
y_test = test_data_without_low_correlation_features['failure']


# Preprocessamento de dados -> FEATURE ENGINEERING

#Primeramente é necessário separar as colunas categóricas das numéricas.
#Uma das formas utilizadas é aplicando a filtragem pelo tipo da coluna com o método select_dtypes

# Colunas categoricas
categorical_columns = list(X_train.select_dtypes('object').columns)

# Colunas numéricas
numerical_columns = list(X_train.select_dtypes('number').columns)

#Definida o conjunto de colunas pelo tipo, o próximo passo é definir as tratativas do pipeline.
#Nesse caso os valores nulos serão substituídos pela string # e posteriormente os dados serão transformados
#em valores numéricos utilizando a técnica de OneHotEncoder
categorical_pipeline = Pipeline([ 
     ('impute', SimpleImputer(strategy='constant', fill_value='#')),
     ('encode', OneHotEncoder(handle_unknown='ignore', sparse=False, drop='first'))
])

#Pipeline para tratamento numérico. Nesse caso foi escolhido a estratégia de preenchimento pela média
#e MinMaxScaler como método de escalonamento.
numeric_pipeline = Pipeline([
    ('imputer', SimpleImputer(strategy='median')),
    ('scaler', MinMaxScaler()) 
])

#Como será aplicado tratamento separados para as colunas do tipo categórica e numérica
#será utilizado a técnica de ColumnTransformer. Que permite criar um pipeline com tratamentos
#distintos por conjunto de colunas.
column_transformer_pipeline = ColumnTransformer([
    ('cat', categorical_pipeline, categorical_columns),
    ('num', numeric_pipeline, numerical_columns)
])

#Transformado os dados cria-se o pipeline de treinamento
train_pipeline = Pipeline([
    ('transformer_pipe', column_transformer_pipeline),
    ('balancear', RandomOverSampler(random_state=42)),
    ('dt', DecisionTreeClassifier(random_state=42))
])

# Instanciar um estimador
descicion_tree_estimator = DecisionTreeClassifier(random_state=42)

#Instanciar um método de feature selection
recursive_feature_elimination = RFE(estimator=DecisionTreeClassifier(random_state=42))

# Criar um pipeline
pipe = Pipeline([
    ('transformer_pipe', column_transformer_pipeline),
    ('balanced_method', RandomOverSampler(random_state=42)),
    ("rfe", recursive_feature_elimination),
    ('scaler', MinMaxScaler()),
    ("estimator", descicion_tree_estimator)])

# Definir os parâmetros do grid search
param_grid_dt = {"rfe__n_features_to_select" : range(1, X_train.shape[1]+1),
                 "dt__criterion" : ['gini', 'entropy', 'log_loss']}

# Instanciar os samples de treinamento
splitter = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

# Instanciar GridSearchCV
grid_dt = GridSearchCV(pipe, 
                       param_grid_dt, 
                       scoring="roc_auc", 
                       cv=splitter, 
                       n_jobs=-1) 

# Fit do modelo
grid_dt.fit(X_train, y_train)


