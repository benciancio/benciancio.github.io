---
 layout: wide_default
 ---    
    



# Real Estate Prediction Model

## Necessary Imports


```python
import os
import pandas as pd
import numpy as np
from sklearn.preprocessing import OneHotEncoder 
from sklearn.compose import ColumnTransformer, make_column_selector
from sklearn.preprocessing import StandardScaler 
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.pipeline import make_pipeline 
from sklearn.impute import SimpleImputer
from sklearn.model_selection import KFold, cross_validate, GridSearchCV
from sklearn.linear_model import Lasso, LassoCV, LogisticRegression
from sklearn.feature_selection import SelectKBest, f_regression
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.model_selection import GridSearchCV
from sklearn.linear_model import LinearRegression
from sklearn.decomposition import PCA
from sklearn.compose import make_column_selector as selector
from sklearn.feature_selection import SelectKBest, f_regression
import warnings
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import r2_score

```

## Data


```python
housing = pd.read_csv('input_data2/housing_train.csv')
holdout = pd.read_csv('input_data2/housing_holdout.csv')
```


```python
X_train = housing.drop(columns=["v_SalePrice"])
y_train = np.log(housing["v_SalePrice"])

```

## Pipeline


```python
numerical_pipeline = make_pipeline(
    SimpleImputer(strategy='mean'),
    StandardScaler()
)

categorical_pipeline = make_pipeline(
    SimpleImputer(strategy='most_frequent'),
    OneHotEncoder(drop='first', sparse_output=False, handle_unknown='ignore')
)


preprocessor = ColumnTransformer(
    transformers=[
        ('numerical', numerical_pipeline, selector(dtype_exclude="object")),
        ('categorical', categorical_pipeline, selector(dtype_include="object"))
    ]
)


pipeline = make_pipeline(
    preprocessor,
    SelectKBest(f_regression),
    HistGradientBoostingRegressor()
)

pipeline
```




<style>#sk-container-id-1 {color: black;}#sk-container-id-1 pre{padding: 0;}#sk-container-id-1 div.sk-toggleable {background-color: white;}#sk-container-id-1 label.sk-toggleable__label {cursor: pointer;display: block;width: 100%;margin-bottom: 0;padding: 0.3em;box-sizing: border-box;text-align: center;}#sk-container-id-1 label.sk-toggleable__label-arrow:before {content: "▸";float: left;margin-right: 0.25em;color: #696969;}#sk-container-id-1 label.sk-toggleable__label-arrow:hover:before {color: black;}#sk-container-id-1 div.sk-estimator:hover label.sk-toggleable__label-arrow:before {color: black;}#sk-container-id-1 div.sk-toggleable__content {max-height: 0;max-width: 0;overflow: hidden;text-align: left;background-color: #f0f8ff;}#sk-container-id-1 div.sk-toggleable__content pre {margin: 0.2em;color: black;border-radius: 0.25em;background-color: #f0f8ff;}#sk-container-id-1 input.sk-toggleable__control:checked~div.sk-toggleable__content {max-height: 200px;max-width: 100%;overflow: auto;}#sk-container-id-1 input.sk-toggleable__control:checked~label.sk-toggleable__label-arrow:before {content: "▾";}#sk-container-id-1 div.sk-estimator input.sk-toggleable__control:checked~label.sk-toggleable__label {background-color: #d4ebff;}#sk-container-id-1 div.sk-label input.sk-toggleable__control:checked~label.sk-toggleable__label {background-color: #d4ebff;}#sk-container-id-1 input.sk-hidden--visually {border: 0;clip: rect(1px 1px 1px 1px);clip: rect(1px, 1px, 1px, 1px);height: 1px;margin: -1px;overflow: hidden;padding: 0;position: absolute;width: 1px;}#sk-container-id-1 div.sk-estimator {font-family: monospace;background-color: #f0f8ff;border: 1px dotted black;border-radius: 0.25em;box-sizing: border-box;margin-bottom: 0.5em;}#sk-container-id-1 div.sk-estimator:hover {background-color: #d4ebff;}#sk-container-id-1 div.sk-parallel-item::after {content: "";width: 100%;border-bottom: 1px solid gray;flex-grow: 1;}#sk-container-id-1 div.sk-label:hover label.sk-toggleable__label {background-color: #d4ebff;}#sk-container-id-1 div.sk-serial::before {content: "";position: absolute;border-left: 1px solid gray;box-sizing: border-box;top: 0;bottom: 0;left: 50%;z-index: 0;}#sk-container-id-1 div.sk-serial {display: flex;flex-direction: column;align-items: center;background-color: white;padding-right: 0.2em;padding-left: 0.2em;position: relative;}#sk-container-id-1 div.sk-item {position: relative;z-index: 1;}#sk-container-id-1 div.sk-parallel {display: flex;align-items: stretch;justify-content: center;background-color: white;position: relative;}#sk-container-id-1 div.sk-item::before, #sk-container-id-1 div.sk-parallel-item::before {content: "";position: absolute;border-left: 1px solid gray;box-sizing: border-box;top: 0;bottom: 0;left: 50%;z-index: -1;}#sk-container-id-1 div.sk-parallel-item {display: flex;flex-direction: column;z-index: 1;position: relative;background-color: white;}#sk-container-id-1 div.sk-parallel-item:first-child::after {align-self: flex-end;width: 50%;}#sk-container-id-1 div.sk-parallel-item:last-child::after {align-self: flex-start;width: 50%;}#sk-container-id-1 div.sk-parallel-item:only-child::after {width: 0;}#sk-container-id-1 div.sk-dashed-wrapped {border: 1px dashed gray;margin: 0 0.4em 0.5em 0.4em;box-sizing: border-box;padding-bottom: 0.4em;background-color: white;}#sk-container-id-1 div.sk-label label {font-family: monospace;font-weight: bold;display: inline-block;line-height: 1.2em;}#sk-container-id-1 div.sk-label-container {text-align: center;}#sk-container-id-1 div.sk-container {/* jupyter's `normalize.less` sets `[hidden] { display: none; }` but bootstrap.min.css set `[hidden] { display: none !important; }` so we also need the `!important` here to be able to override the default hidden behavior on the sphinx rendered scikit-learn.org. See: https://github.com/scikit-learn/scikit-learn/issues/21755 */display: inline-block !important;position: relative;}#sk-container-id-1 div.sk-text-repr-fallback {display: none;}</style><div id="sk-container-id-1" class="sk-top-container"><div class="sk-text-repr-fallback"><pre>Pipeline(steps=[(&#x27;columntransformer&#x27;,
                 ColumnTransformer(transformers=[(&#x27;numerical&#x27;,
                                                  Pipeline(steps=[(&#x27;simpleimputer&#x27;,
                                                                   SimpleImputer()),
                                                                  (&#x27;standardscaler&#x27;,
                                                                   StandardScaler())]),
                                                  &lt;sklearn.compose._column_transformer.make_column_selector object at 0x130615310&gt;),
                                                 (&#x27;categorical&#x27;,
                                                  Pipeline(steps=[(&#x27;simpleimputer&#x27;,
                                                                   SimpleImputer(strategy=&#x27;most_frequent&#x27;)),
                                                                  (&#x27;onehotencoder&#x27;,
                                                                   OneHotEncoder(drop=&#x27;first&#x27;,
                                                                                 handle_unknown=&#x27;ignore&#x27;,
                                                                                 sparse_output=False))]),
                                                  &lt;sklearn.compose._column_transformer.make_column_selector object at 0x12ff52410&gt;)])),
                (&#x27;selectkbest&#x27;,
                 SelectKBest(score_func=&lt;function f_regression at 0x12eab2de0&gt;)),
                (&#x27;histgradientboostingregressor&#x27;,
                 HistGradientBoostingRegressor())])</pre><b>In a Jupyter environment, please rerun this cell to show the HTML representation or trust the notebook. <br />On GitHub, the HTML representation is unable to render, please try loading this page with nbviewer.org.</b></div><div class="sk-container" hidden><div class="sk-item sk-dashed-wrapped"><div class="sk-label-container"><div class="sk-label sk-toggleable"><input class="sk-toggleable__control sk-hidden--visually" id="sk-estimator-id-1" type="checkbox" ><label for="sk-estimator-id-1" class="sk-toggleable__label sk-toggleable__label-arrow">Pipeline</label><div class="sk-toggleable__content"><pre>Pipeline(steps=[(&#x27;columntransformer&#x27;,
                 ColumnTransformer(transformers=[(&#x27;numerical&#x27;,
                                                  Pipeline(steps=[(&#x27;simpleimputer&#x27;,
                                                                   SimpleImputer()),
                                                                  (&#x27;standardscaler&#x27;,
                                                                   StandardScaler())]),
                                                  &lt;sklearn.compose._column_transformer.make_column_selector object at 0x130615310&gt;),
                                                 (&#x27;categorical&#x27;,
                                                  Pipeline(steps=[(&#x27;simpleimputer&#x27;,
                                                                   SimpleImputer(strategy=&#x27;most_frequent&#x27;)),
                                                                  (&#x27;onehotencoder&#x27;,
                                                                   OneHotEncoder(drop=&#x27;first&#x27;,
                                                                                 handle_unknown=&#x27;ignore&#x27;,
                                                                                 sparse_output=False))]),
                                                  &lt;sklearn.compose._column_transformer.make_column_selector object at 0x12ff52410&gt;)])),
                (&#x27;selectkbest&#x27;,
                 SelectKBest(score_func=&lt;function f_regression at 0x12eab2de0&gt;)),
                (&#x27;histgradientboostingregressor&#x27;,
                 HistGradientBoostingRegressor())])</pre></div></div></div><div class="sk-serial"><div class="sk-item sk-dashed-wrapped"><div class="sk-label-container"><div class="sk-label sk-toggleable"><input class="sk-toggleable__control sk-hidden--visually" id="sk-estimator-id-2" type="checkbox" ><label for="sk-estimator-id-2" class="sk-toggleable__label sk-toggleable__label-arrow">columntransformer: ColumnTransformer</label><div class="sk-toggleable__content"><pre>ColumnTransformer(transformers=[(&#x27;numerical&#x27;,
                                 Pipeline(steps=[(&#x27;simpleimputer&#x27;,
                                                  SimpleImputer()),
                                                 (&#x27;standardscaler&#x27;,
                                                  StandardScaler())]),
                                 &lt;sklearn.compose._column_transformer.make_column_selector object at 0x130615310&gt;),
                                (&#x27;categorical&#x27;,
                                 Pipeline(steps=[(&#x27;simpleimputer&#x27;,
                                                  SimpleImputer(strategy=&#x27;most_frequent&#x27;)),
                                                 (&#x27;onehotencoder&#x27;,
                                                  OneHotEncoder(drop=&#x27;first&#x27;,
                                                                handle_unknown=&#x27;ignore&#x27;,
                                                                sparse_output=False))]),
                                 &lt;sklearn.compose._column_transformer.make_column_selector object at 0x12ff52410&gt;)])</pre></div></div></div><div class="sk-parallel"><div class="sk-parallel-item"><div class="sk-item"><div class="sk-label-container"><div class="sk-label sk-toggleable"><input class="sk-toggleable__control sk-hidden--visually" id="sk-estimator-id-3" type="checkbox" ><label for="sk-estimator-id-3" class="sk-toggleable__label sk-toggleable__label-arrow">numerical</label><div class="sk-toggleable__content"><pre>&lt;sklearn.compose._column_transformer.make_column_selector object at 0x130615310&gt;</pre></div></div></div><div class="sk-serial"><div class="sk-item"><div class="sk-serial"><div class="sk-item"><div class="sk-estimator sk-toggleable"><input class="sk-toggleable__control sk-hidden--visually" id="sk-estimator-id-4" type="checkbox" ><label for="sk-estimator-id-4" class="sk-toggleable__label sk-toggleable__label-arrow">SimpleImputer</label><div class="sk-toggleable__content"><pre>SimpleImputer()</pre></div></div></div><div class="sk-item"><div class="sk-estimator sk-toggleable"><input class="sk-toggleable__control sk-hidden--visually" id="sk-estimator-id-5" type="checkbox" ><label for="sk-estimator-id-5" class="sk-toggleable__label sk-toggleable__label-arrow">StandardScaler</label><div class="sk-toggleable__content"><pre>StandardScaler()</pre></div></div></div></div></div></div></div></div><div class="sk-parallel-item"><div class="sk-item"><div class="sk-label-container"><div class="sk-label sk-toggleable"><input class="sk-toggleable__control sk-hidden--visually" id="sk-estimator-id-6" type="checkbox" ><label for="sk-estimator-id-6" class="sk-toggleable__label sk-toggleable__label-arrow">categorical</label><div class="sk-toggleable__content"><pre>&lt;sklearn.compose._column_transformer.make_column_selector object at 0x12ff52410&gt;</pre></div></div></div><div class="sk-serial"><div class="sk-item"><div class="sk-serial"><div class="sk-item"><div class="sk-estimator sk-toggleable"><input class="sk-toggleable__control sk-hidden--visually" id="sk-estimator-id-7" type="checkbox" ><label for="sk-estimator-id-7" class="sk-toggleable__label sk-toggleable__label-arrow">SimpleImputer</label><div class="sk-toggleable__content"><pre>SimpleImputer(strategy=&#x27;most_frequent&#x27;)</pre></div></div></div><div class="sk-item"><div class="sk-estimator sk-toggleable"><input class="sk-toggleable__control sk-hidden--visually" id="sk-estimator-id-8" type="checkbox" ><label for="sk-estimator-id-8" class="sk-toggleable__label sk-toggleable__label-arrow">OneHotEncoder</label><div class="sk-toggleable__content"><pre>OneHotEncoder(drop=&#x27;first&#x27;, handle_unknown=&#x27;ignore&#x27;, sparse_output=False)</pre></div></div></div></div></div></div></div></div></div></div><div class="sk-item"><div class="sk-estimator sk-toggleable"><input class="sk-toggleable__control sk-hidden--visually" id="sk-estimator-id-9" type="checkbox" ><label for="sk-estimator-id-9" class="sk-toggleable__label sk-toggleable__label-arrow">SelectKBest</label><div class="sk-toggleable__content"><pre>SelectKBest(score_func=&lt;function f_regression at 0x12eab2de0&gt;)</pre></div></div></div><div class="sk-item"><div class="sk-estimator sk-toggleable"><input class="sk-toggleable__control sk-hidden--visually" id="sk-estimator-id-10" type="checkbox" ><label for="sk-estimator-id-10" class="sk-toggleable__label sk-toggleable__label-arrow">HistGradientBoostingRegressor</label><div class="sk-toggleable__content"><pre>HistGradientBoostingRegressor()</pre></div></div></div></div></div></div></div>




```python
param_grid = {
    'selectkbest__k': [10, 20, 30],  
    'histgradientboostingregressor__max_iter': [100, 200, 300],  
}

```


```python
warnings.filterwarnings("ignore")


grid_search = GridSearchCV(pipeline, param_grid, cv=5, scoring='r2')
grid_search.fit(X_train, y_train)
```




<style>#sk-container-id-2 {color: black;}#sk-container-id-2 pre{padding: 0;}#sk-container-id-2 div.sk-toggleable {background-color: white;}#sk-container-id-2 label.sk-toggleable__label {cursor: pointer;display: block;width: 100%;margin-bottom: 0;padding: 0.3em;box-sizing: border-box;text-align: center;}#sk-container-id-2 label.sk-toggleable__label-arrow:before {content: "▸";float: left;margin-right: 0.25em;color: #696969;}#sk-container-id-2 label.sk-toggleable__label-arrow:hover:before {color: black;}#sk-container-id-2 div.sk-estimator:hover label.sk-toggleable__label-arrow:before {color: black;}#sk-container-id-2 div.sk-toggleable__content {max-height: 0;max-width: 0;overflow: hidden;text-align: left;background-color: #f0f8ff;}#sk-container-id-2 div.sk-toggleable__content pre {margin: 0.2em;color: black;border-radius: 0.25em;background-color: #f0f8ff;}#sk-container-id-2 input.sk-toggleable__control:checked~div.sk-toggleable__content {max-height: 200px;max-width: 100%;overflow: auto;}#sk-container-id-2 input.sk-toggleable__control:checked~label.sk-toggleable__label-arrow:before {content: "▾";}#sk-container-id-2 div.sk-estimator input.sk-toggleable__control:checked~label.sk-toggleable__label {background-color: #d4ebff;}#sk-container-id-2 div.sk-label input.sk-toggleable__control:checked~label.sk-toggleable__label {background-color: #d4ebff;}#sk-container-id-2 input.sk-hidden--visually {border: 0;clip: rect(1px 1px 1px 1px);clip: rect(1px, 1px, 1px, 1px);height: 1px;margin: -1px;overflow: hidden;padding: 0;position: absolute;width: 1px;}#sk-container-id-2 div.sk-estimator {font-family: monospace;background-color: #f0f8ff;border: 1px dotted black;border-radius: 0.25em;box-sizing: border-box;margin-bottom: 0.5em;}#sk-container-id-2 div.sk-estimator:hover {background-color: #d4ebff;}#sk-container-id-2 div.sk-parallel-item::after {content: "";width: 100%;border-bottom: 1px solid gray;flex-grow: 1;}#sk-container-id-2 div.sk-label:hover label.sk-toggleable__label {background-color: #d4ebff;}#sk-container-id-2 div.sk-serial::before {content: "";position: absolute;border-left: 1px solid gray;box-sizing: border-box;top: 0;bottom: 0;left: 50%;z-index: 0;}#sk-container-id-2 div.sk-serial {display: flex;flex-direction: column;align-items: center;background-color: white;padding-right: 0.2em;padding-left: 0.2em;position: relative;}#sk-container-id-2 div.sk-item {position: relative;z-index: 1;}#sk-container-id-2 div.sk-parallel {display: flex;align-items: stretch;justify-content: center;background-color: white;position: relative;}#sk-container-id-2 div.sk-item::before, #sk-container-id-2 div.sk-parallel-item::before {content: "";position: absolute;border-left: 1px solid gray;box-sizing: border-box;top: 0;bottom: 0;left: 50%;z-index: -1;}#sk-container-id-2 div.sk-parallel-item {display: flex;flex-direction: column;z-index: 1;position: relative;background-color: white;}#sk-container-id-2 div.sk-parallel-item:first-child::after {align-self: flex-end;width: 50%;}#sk-container-id-2 div.sk-parallel-item:last-child::after {align-self: flex-start;width: 50%;}#sk-container-id-2 div.sk-parallel-item:only-child::after {width: 0;}#sk-container-id-2 div.sk-dashed-wrapped {border: 1px dashed gray;margin: 0 0.4em 0.5em 0.4em;box-sizing: border-box;padding-bottom: 0.4em;background-color: white;}#sk-container-id-2 div.sk-label label {font-family: monospace;font-weight: bold;display: inline-block;line-height: 1.2em;}#sk-container-id-2 div.sk-label-container {text-align: center;}#sk-container-id-2 div.sk-container {/* jupyter's `normalize.less` sets `[hidden] { display: none; }` but bootstrap.min.css set `[hidden] { display: none !important; }` so we also need the `!important` here to be able to override the default hidden behavior on the sphinx rendered scikit-learn.org. See: https://github.com/scikit-learn/scikit-learn/issues/21755 */display: inline-block !important;position: relative;}#sk-container-id-2 div.sk-text-repr-fallback {display: none;}</style><div id="sk-container-id-2" class="sk-top-container"><div class="sk-text-repr-fallback"><pre>GridSearchCV(cv=5,
             estimator=Pipeline(steps=[(&#x27;columntransformer&#x27;,
                                        ColumnTransformer(transformers=[(&#x27;numerical&#x27;,
                                                                         Pipeline(steps=[(&#x27;simpleimputer&#x27;,
                                                                                          SimpleImputer()),
                                                                                         (&#x27;standardscaler&#x27;,
                                                                                          StandardScaler())]),
                                                                         &lt;sklearn.compose._column_transformer.make_column_selector object at 0x130615310&gt;),
                                                                        (&#x27;categorical&#x27;,
                                                                         Pipeline(steps=[(&#x27;simpleimputer&#x27;,
                                                                                          SimpleImputer(strategy=&#x27;m...
                                                                                                        sparse_output=False))]),
                                                                         &lt;sklearn.compose._column_transformer.make_column_selector object at 0x12ff52410&gt;)])),
                                       (&#x27;selectkbest&#x27;,
                                        SelectKBest(score_func=&lt;function f_regression at 0x12eab2de0&gt;)),
                                       (&#x27;histgradientboostingregressor&#x27;,
                                        HistGradientBoostingRegressor())]),
             param_grid={&#x27;histgradientboostingregressor__max_iter&#x27;: [100, 200,
                                                                     300],
                         &#x27;selectkbest__k&#x27;: [10, 20, 30]},
             scoring=&#x27;r2&#x27;)</pre><b>In a Jupyter environment, please rerun this cell to show the HTML representation or trust the notebook. <br />On GitHub, the HTML representation is unable to render, please try loading this page with nbviewer.org.</b></div><div class="sk-container" hidden><div class="sk-item sk-dashed-wrapped"><div class="sk-label-container"><div class="sk-label sk-toggleable"><input class="sk-toggleable__control sk-hidden--visually" id="sk-estimator-id-11" type="checkbox" ><label for="sk-estimator-id-11" class="sk-toggleable__label sk-toggleable__label-arrow">GridSearchCV</label><div class="sk-toggleable__content"><pre>GridSearchCV(cv=5,
             estimator=Pipeline(steps=[(&#x27;columntransformer&#x27;,
                                        ColumnTransformer(transformers=[(&#x27;numerical&#x27;,
                                                                         Pipeline(steps=[(&#x27;simpleimputer&#x27;,
                                                                                          SimpleImputer()),
                                                                                         (&#x27;standardscaler&#x27;,
                                                                                          StandardScaler())]),
                                                                         &lt;sklearn.compose._column_transformer.make_column_selector object at 0x130615310&gt;),
                                                                        (&#x27;categorical&#x27;,
                                                                         Pipeline(steps=[(&#x27;simpleimputer&#x27;,
                                                                                          SimpleImputer(strategy=&#x27;m...
                                                                                                        sparse_output=False))]),
                                                                         &lt;sklearn.compose._column_transformer.make_column_selector object at 0x12ff52410&gt;)])),
                                       (&#x27;selectkbest&#x27;,
                                        SelectKBest(score_func=&lt;function f_regression at 0x12eab2de0&gt;)),
                                       (&#x27;histgradientboostingregressor&#x27;,
                                        HistGradientBoostingRegressor())]),
             param_grid={&#x27;histgradientboostingregressor__max_iter&#x27;: [100, 200,
                                                                     300],
                         &#x27;selectkbest__k&#x27;: [10, 20, 30]},
             scoring=&#x27;r2&#x27;)</pre></div></div></div><div class="sk-parallel"><div class="sk-parallel-item"><div class="sk-item"><div class="sk-label-container"><div class="sk-label sk-toggleable"><input class="sk-toggleable__control sk-hidden--visually" id="sk-estimator-id-12" type="checkbox" ><label for="sk-estimator-id-12" class="sk-toggleable__label sk-toggleable__label-arrow">estimator: Pipeline</label><div class="sk-toggleable__content"><pre>Pipeline(steps=[(&#x27;columntransformer&#x27;,
                 ColumnTransformer(transformers=[(&#x27;numerical&#x27;,
                                                  Pipeline(steps=[(&#x27;simpleimputer&#x27;,
                                                                   SimpleImputer()),
                                                                  (&#x27;standardscaler&#x27;,
                                                                   StandardScaler())]),
                                                  &lt;sklearn.compose._column_transformer.make_column_selector object at 0x130615310&gt;),
                                                 (&#x27;categorical&#x27;,
                                                  Pipeline(steps=[(&#x27;simpleimputer&#x27;,
                                                                   SimpleImputer(strategy=&#x27;most_frequent&#x27;)),
                                                                  (&#x27;onehotencoder&#x27;,
                                                                   OneHotEncoder(drop=&#x27;first&#x27;,
                                                                                 handle_unknown=&#x27;ignore&#x27;,
                                                                                 sparse_output=False))]),
                                                  &lt;sklearn.compose._column_transformer.make_column_selector object at 0x12ff52410&gt;)])),
                (&#x27;selectkbest&#x27;,
                 SelectKBest(score_func=&lt;function f_regression at 0x12eab2de0&gt;)),
                (&#x27;histgradientboostingregressor&#x27;,
                 HistGradientBoostingRegressor())])</pre></div></div></div><div class="sk-serial"><div class="sk-item"><div class="sk-serial"><div class="sk-item sk-dashed-wrapped"><div class="sk-label-container"><div class="sk-label sk-toggleable"><input class="sk-toggleable__control sk-hidden--visually" id="sk-estimator-id-13" type="checkbox" ><label for="sk-estimator-id-13" class="sk-toggleable__label sk-toggleable__label-arrow">columntransformer: ColumnTransformer</label><div class="sk-toggleable__content"><pre>ColumnTransformer(transformers=[(&#x27;numerical&#x27;,
                                 Pipeline(steps=[(&#x27;simpleimputer&#x27;,
                                                  SimpleImputer()),
                                                 (&#x27;standardscaler&#x27;,
                                                  StandardScaler())]),
                                 &lt;sklearn.compose._column_transformer.make_column_selector object at 0x130615310&gt;),
                                (&#x27;categorical&#x27;,
                                 Pipeline(steps=[(&#x27;simpleimputer&#x27;,
                                                  SimpleImputer(strategy=&#x27;most_frequent&#x27;)),
                                                 (&#x27;onehotencoder&#x27;,
                                                  OneHotEncoder(drop=&#x27;first&#x27;,
                                                                handle_unknown=&#x27;ignore&#x27;,
                                                                sparse_output=False))]),
                                 &lt;sklearn.compose._column_transformer.make_column_selector object at 0x12ff52410&gt;)])</pre></div></div></div><div class="sk-parallel"><div class="sk-parallel-item"><div class="sk-item"><div class="sk-label-container"><div class="sk-label sk-toggleable"><input class="sk-toggleable__control sk-hidden--visually" id="sk-estimator-id-14" type="checkbox" ><label for="sk-estimator-id-14" class="sk-toggleable__label sk-toggleable__label-arrow">numerical</label><div class="sk-toggleable__content"><pre>&lt;sklearn.compose._column_transformer.make_column_selector object at 0x130615310&gt;</pre></div></div></div><div class="sk-serial"><div class="sk-item"><div class="sk-serial"><div class="sk-item"><div class="sk-estimator sk-toggleable"><input class="sk-toggleable__control sk-hidden--visually" id="sk-estimator-id-15" type="checkbox" ><label for="sk-estimator-id-15" class="sk-toggleable__label sk-toggleable__label-arrow">SimpleImputer</label><div class="sk-toggleable__content"><pre>SimpleImputer()</pre></div></div></div><div class="sk-item"><div class="sk-estimator sk-toggleable"><input class="sk-toggleable__control sk-hidden--visually" id="sk-estimator-id-16" type="checkbox" ><label for="sk-estimator-id-16" class="sk-toggleable__label sk-toggleable__label-arrow">StandardScaler</label><div class="sk-toggleable__content"><pre>StandardScaler()</pre></div></div></div></div></div></div></div></div><div class="sk-parallel-item"><div class="sk-item"><div class="sk-label-container"><div class="sk-label sk-toggleable"><input class="sk-toggleable__control sk-hidden--visually" id="sk-estimator-id-17" type="checkbox" ><label for="sk-estimator-id-17" class="sk-toggleable__label sk-toggleable__label-arrow">categorical</label><div class="sk-toggleable__content"><pre>&lt;sklearn.compose._column_transformer.make_column_selector object at 0x12ff52410&gt;</pre></div></div></div><div class="sk-serial"><div class="sk-item"><div class="sk-serial"><div class="sk-item"><div class="sk-estimator sk-toggleable"><input class="sk-toggleable__control sk-hidden--visually" id="sk-estimator-id-18" type="checkbox" ><label for="sk-estimator-id-18" class="sk-toggleable__label sk-toggleable__label-arrow">SimpleImputer</label><div class="sk-toggleable__content"><pre>SimpleImputer(strategy=&#x27;most_frequent&#x27;)</pre></div></div></div><div class="sk-item"><div class="sk-estimator sk-toggleable"><input class="sk-toggleable__control sk-hidden--visually" id="sk-estimator-id-19" type="checkbox" ><label for="sk-estimator-id-19" class="sk-toggleable__label sk-toggleable__label-arrow">OneHotEncoder</label><div class="sk-toggleable__content"><pre>OneHotEncoder(drop=&#x27;first&#x27;, handle_unknown=&#x27;ignore&#x27;, sparse_output=False)</pre></div></div></div></div></div></div></div></div></div></div><div class="sk-item"><div class="sk-estimator sk-toggleable"><input class="sk-toggleable__control sk-hidden--visually" id="sk-estimator-id-20" type="checkbox" ><label for="sk-estimator-id-20" class="sk-toggleable__label sk-toggleable__label-arrow">SelectKBest</label><div class="sk-toggleable__content"><pre>SelectKBest(score_func=&lt;function f_regression at 0x12eab2de0&gt;)</pre></div></div></div><div class="sk-item"><div class="sk-estimator sk-toggleable"><input class="sk-toggleable__control sk-hidden--visually" id="sk-estimator-id-21" type="checkbox" ><label for="sk-estimator-id-21" class="sk-toggleable__label sk-toggleable__label-arrow">HistGradientBoostingRegressor</label><div class="sk-toggleable__content"><pre>HistGradientBoostingRegressor()</pre></div></div></div></div></div></div></div></div></div></div></div></div>




```python
best_params = grid_search.best_estimator_
best_params
```




<style>#sk-container-id-4 {color: black;}#sk-container-id-4 pre{padding: 0;}#sk-container-id-4 div.sk-toggleable {background-color: white;}#sk-container-id-4 label.sk-toggleable__label {cursor: pointer;display: block;width: 100%;margin-bottom: 0;padding: 0.3em;box-sizing: border-box;text-align: center;}#sk-container-id-4 label.sk-toggleable__label-arrow:before {content: "▸";float: left;margin-right: 0.25em;color: #696969;}#sk-container-id-4 label.sk-toggleable__label-arrow:hover:before {color: black;}#sk-container-id-4 div.sk-estimator:hover label.sk-toggleable__label-arrow:before {color: black;}#sk-container-id-4 div.sk-toggleable__content {max-height: 0;max-width: 0;overflow: hidden;text-align: left;background-color: #f0f8ff;}#sk-container-id-4 div.sk-toggleable__content pre {margin: 0.2em;color: black;border-radius: 0.25em;background-color: #f0f8ff;}#sk-container-id-4 input.sk-toggleable__control:checked~div.sk-toggleable__content {max-height: 200px;max-width: 100%;overflow: auto;}#sk-container-id-4 input.sk-toggleable__control:checked~label.sk-toggleable__label-arrow:before {content: "▾";}#sk-container-id-4 div.sk-estimator input.sk-toggleable__control:checked~label.sk-toggleable__label {background-color: #d4ebff;}#sk-container-id-4 div.sk-label input.sk-toggleable__control:checked~label.sk-toggleable__label {background-color: #d4ebff;}#sk-container-id-4 input.sk-hidden--visually {border: 0;clip: rect(1px 1px 1px 1px);clip: rect(1px, 1px, 1px, 1px);height: 1px;margin: -1px;overflow: hidden;padding: 0;position: absolute;width: 1px;}#sk-container-id-4 div.sk-estimator {font-family: monospace;background-color: #f0f8ff;border: 1px dotted black;border-radius: 0.25em;box-sizing: border-box;margin-bottom: 0.5em;}#sk-container-id-4 div.sk-estimator:hover {background-color: #d4ebff;}#sk-container-id-4 div.sk-parallel-item::after {content: "";width: 100%;border-bottom: 1px solid gray;flex-grow: 1;}#sk-container-id-4 div.sk-label:hover label.sk-toggleable__label {background-color: #d4ebff;}#sk-container-id-4 div.sk-serial::before {content: "";position: absolute;border-left: 1px solid gray;box-sizing: border-box;top: 0;bottom: 0;left: 50%;z-index: 0;}#sk-container-id-4 div.sk-serial {display: flex;flex-direction: column;align-items: center;background-color: white;padding-right: 0.2em;padding-left: 0.2em;position: relative;}#sk-container-id-4 div.sk-item {position: relative;z-index: 1;}#sk-container-id-4 div.sk-parallel {display: flex;align-items: stretch;justify-content: center;background-color: white;position: relative;}#sk-container-id-4 div.sk-item::before, #sk-container-id-4 div.sk-parallel-item::before {content: "";position: absolute;border-left: 1px solid gray;box-sizing: border-box;top: 0;bottom: 0;left: 50%;z-index: -1;}#sk-container-id-4 div.sk-parallel-item {display: flex;flex-direction: column;z-index: 1;position: relative;background-color: white;}#sk-container-id-4 div.sk-parallel-item:first-child::after {align-self: flex-end;width: 50%;}#sk-container-id-4 div.sk-parallel-item:last-child::after {align-self: flex-start;width: 50%;}#sk-container-id-4 div.sk-parallel-item:only-child::after {width: 0;}#sk-container-id-4 div.sk-dashed-wrapped {border: 1px dashed gray;margin: 0 0.4em 0.5em 0.4em;box-sizing: border-box;padding-bottom: 0.4em;background-color: white;}#sk-container-id-4 div.sk-label label {font-family: monospace;font-weight: bold;display: inline-block;line-height: 1.2em;}#sk-container-id-4 div.sk-label-container {text-align: center;}#sk-container-id-4 div.sk-container {/* jupyter's `normalize.less` sets `[hidden] { display: none; }` but bootstrap.min.css set `[hidden] { display: none !important; }` so we also need the `!important` here to be able to override the default hidden behavior on the sphinx rendered scikit-learn.org. See: https://github.com/scikit-learn/scikit-learn/issues/21755 */display: inline-block !important;position: relative;}#sk-container-id-4 div.sk-text-repr-fallback {display: none;}</style><div id="sk-container-id-4" class="sk-top-container"><div class="sk-text-repr-fallback"><pre>Pipeline(steps=[(&#x27;columntransformer&#x27;,
                 ColumnTransformer(transformers=[(&#x27;numerical&#x27;,
                                                  Pipeline(steps=[(&#x27;simpleimputer&#x27;,
                                                                   SimpleImputer()),
                                                                  (&#x27;standardscaler&#x27;,
                                                                   StandardScaler())]),
                                                  &lt;sklearn.compose._column_transformer.make_column_selector object at 0x1300119d0&gt;),
                                                 (&#x27;categorical&#x27;,
                                                  Pipeline(steps=[(&#x27;simpleimputer&#x27;,
                                                                   SimpleImputer(strategy=&#x27;most_frequent&#x27;)),
                                                                  (&#x27;onehotencoder&#x27;,
                                                                   OneHotEncoder(drop=&#x27;first&#x27;,
                                                                                 handle_unknown=&#x27;ignore&#x27;,
                                                                                 sparse_output=False))]),
                                                  &lt;sklearn.compose._column_transformer.make_column_selector object at 0x130673150&gt;)])),
                (&#x27;selectkbest&#x27;,
                 SelectKBest(k=30,
                             score_func=&lt;function f_regression at 0x12eab2de0&gt;)),
                (&#x27;histgradientboostingregressor&#x27;,
                 HistGradientBoostingRegressor())])</pre><b>In a Jupyter environment, please rerun this cell to show the HTML representation or trust the notebook. <br />On GitHub, the HTML representation is unable to render, please try loading this page with nbviewer.org.</b></div><div class="sk-container" hidden><div class="sk-item sk-dashed-wrapped"><div class="sk-label-container"><div class="sk-label sk-toggleable"><input class="sk-toggleable__control sk-hidden--visually" id="sk-estimator-id-32" type="checkbox" ><label for="sk-estimator-id-32" class="sk-toggleable__label sk-toggleable__label-arrow">Pipeline</label><div class="sk-toggleable__content"><pre>Pipeline(steps=[(&#x27;columntransformer&#x27;,
                 ColumnTransformer(transformers=[(&#x27;numerical&#x27;,
                                                  Pipeline(steps=[(&#x27;simpleimputer&#x27;,
                                                                   SimpleImputer()),
                                                                  (&#x27;standardscaler&#x27;,
                                                                   StandardScaler())]),
                                                  &lt;sklearn.compose._column_transformer.make_column_selector object at 0x1300119d0&gt;),
                                                 (&#x27;categorical&#x27;,
                                                  Pipeline(steps=[(&#x27;simpleimputer&#x27;,
                                                                   SimpleImputer(strategy=&#x27;most_frequent&#x27;)),
                                                                  (&#x27;onehotencoder&#x27;,
                                                                   OneHotEncoder(drop=&#x27;first&#x27;,
                                                                                 handle_unknown=&#x27;ignore&#x27;,
                                                                                 sparse_output=False))]),
                                                  &lt;sklearn.compose._column_transformer.make_column_selector object at 0x130673150&gt;)])),
                (&#x27;selectkbest&#x27;,
                 SelectKBest(k=30,
                             score_func=&lt;function f_regression at 0x12eab2de0&gt;)),
                (&#x27;histgradientboostingregressor&#x27;,
                 HistGradientBoostingRegressor())])</pre></div></div></div><div class="sk-serial"><div class="sk-item sk-dashed-wrapped"><div class="sk-label-container"><div class="sk-label sk-toggleable"><input class="sk-toggleable__control sk-hidden--visually" id="sk-estimator-id-33" type="checkbox" ><label for="sk-estimator-id-33" class="sk-toggleable__label sk-toggleable__label-arrow">columntransformer: ColumnTransformer</label><div class="sk-toggleable__content"><pre>ColumnTransformer(transformers=[(&#x27;numerical&#x27;,
                                 Pipeline(steps=[(&#x27;simpleimputer&#x27;,
                                                  SimpleImputer()),
                                                 (&#x27;standardscaler&#x27;,
                                                  StandardScaler())]),
                                 &lt;sklearn.compose._column_transformer.make_column_selector object at 0x1300119d0&gt;),
                                (&#x27;categorical&#x27;,
                                 Pipeline(steps=[(&#x27;simpleimputer&#x27;,
                                                  SimpleImputer(strategy=&#x27;most_frequent&#x27;)),
                                                 (&#x27;onehotencoder&#x27;,
                                                  OneHotEncoder(drop=&#x27;first&#x27;,
                                                                handle_unknown=&#x27;ignore&#x27;,
                                                                sparse_output=False))]),
                                 &lt;sklearn.compose._column_transformer.make_column_selector object at 0x130673150&gt;)])</pre></div></div></div><div class="sk-parallel"><div class="sk-parallel-item"><div class="sk-item"><div class="sk-label-container"><div class="sk-label sk-toggleable"><input class="sk-toggleable__control sk-hidden--visually" id="sk-estimator-id-34" type="checkbox" ><label for="sk-estimator-id-34" class="sk-toggleable__label sk-toggleable__label-arrow">numerical</label><div class="sk-toggleable__content"><pre>&lt;sklearn.compose._column_transformer.make_column_selector object at 0x1300119d0&gt;</pre></div></div></div><div class="sk-serial"><div class="sk-item"><div class="sk-serial"><div class="sk-item"><div class="sk-estimator sk-toggleable"><input class="sk-toggleable__control sk-hidden--visually" id="sk-estimator-id-35" type="checkbox" ><label for="sk-estimator-id-35" class="sk-toggleable__label sk-toggleable__label-arrow">SimpleImputer</label><div class="sk-toggleable__content"><pre>SimpleImputer()</pre></div></div></div><div class="sk-item"><div class="sk-estimator sk-toggleable"><input class="sk-toggleable__control sk-hidden--visually" id="sk-estimator-id-36" type="checkbox" ><label for="sk-estimator-id-36" class="sk-toggleable__label sk-toggleable__label-arrow">StandardScaler</label><div class="sk-toggleable__content"><pre>StandardScaler()</pre></div></div></div></div></div></div></div></div><div class="sk-parallel-item"><div class="sk-item"><div class="sk-label-container"><div class="sk-label sk-toggleable"><input class="sk-toggleable__control sk-hidden--visually" id="sk-estimator-id-37" type="checkbox" ><label for="sk-estimator-id-37" class="sk-toggleable__label sk-toggleable__label-arrow">categorical</label><div class="sk-toggleable__content"><pre>&lt;sklearn.compose._column_transformer.make_column_selector object at 0x130673150&gt;</pre></div></div></div><div class="sk-serial"><div class="sk-item"><div class="sk-serial"><div class="sk-item"><div class="sk-estimator sk-toggleable"><input class="sk-toggleable__control sk-hidden--visually" id="sk-estimator-id-38" type="checkbox" ><label for="sk-estimator-id-38" class="sk-toggleable__label sk-toggleable__label-arrow">SimpleImputer</label><div class="sk-toggleable__content"><pre>SimpleImputer(strategy=&#x27;most_frequent&#x27;)</pre></div></div></div><div class="sk-item"><div class="sk-estimator sk-toggleable"><input class="sk-toggleable__control sk-hidden--visually" id="sk-estimator-id-39" type="checkbox" ><label for="sk-estimator-id-39" class="sk-toggleable__label sk-toggleable__label-arrow">OneHotEncoder</label><div class="sk-toggleable__content"><pre>OneHotEncoder(drop=&#x27;first&#x27;, handle_unknown=&#x27;ignore&#x27;, sparse_output=False)</pre></div></div></div></div></div></div></div></div></div></div><div class="sk-item"><div class="sk-estimator sk-toggleable"><input class="sk-toggleable__control sk-hidden--visually" id="sk-estimator-id-40" type="checkbox" ><label for="sk-estimator-id-40" class="sk-toggleable__label sk-toggleable__label-arrow">SelectKBest</label><div class="sk-toggleable__content"><pre>SelectKBest(k=30, score_func=&lt;function f_regression at 0x12eab2de0&gt;)</pre></div></div></div><div class="sk-item"><div class="sk-estimator sk-toggleable"><input class="sk-toggleable__control sk-hidden--visually" id="sk-estimator-id-41" type="checkbox" ><label for="sk-estimator-id-41" class="sk-toggleable__label sk-toggleable__label-arrow">HistGradientBoostingRegressor</label><div class="sk-toggleable__content"><pre>HistGradientBoostingRegressor()</pre></div></div></div></div></div></div></div>




```python
holdout_predictions = best_params.predict(holdout)

```


```python
holdout_predictions
```




    array([12.09197844, 12.32513558, 11.63979959, 12.03948137, 12.57861826,
           11.29678909, 11.68437387, 11.7984542 , 11.76080408, 11.90780959,
           11.50086652, 11.50422221, 12.17446248, 12.87207548, 12.47597739,
           11.89199924, 11.92401083, 12.68804317, 12.36508803, 12.45655775,
           12.29382727, 12.39502055, 12.52852685, 12.69337619, 12.44247797,
           12.86541654, 12.21989229, 11.76042983, 12.19725542, 10.80728357,
           11.7643537 , 11.85039973, 12.06017599, 11.92017581, 11.84084969,
           11.91874544, 11.87691539, 11.83155595, 11.91917108, 11.38835347,
           11.83415848, 11.72818464, 11.53136976, 11.80388337, 11.77705022,
           11.76898583, 11.84595034, 11.85446496, 11.38731624, 11.54536623,
           11.62442705, 11.81228965, 12.24835691, 11.88371105, 11.9061523 ,
           12.27888311, 12.19067754, 11.94850792, 12.61105936, 12.25337347,
           11.80623109, 11.93483804, 11.80791091, 11.84470395, 11.53563292,
           11.98514576, 12.46869632, 12.8017448 , 11.71995109, 11.84492426,
           11.78600553, 12.92091056, 12.8534121 , 12.25098503, 12.19758633,
           12.36193162, 12.78291807, 12.8977782 , 11.96969468, 11.24148231,
           11.59409191, 11.82135433, 11.73671436, 11.93267516, 11.77758761,
           11.92348103, 11.83035768, 11.8015978 , 11.7206575 , 12.24646681,
           12.19168018, 12.32081615, 12.25678044, 12.32476738, 12.23684683,
           12.39558453, 12.42440233, 11.5264532 , 12.82366976, 11.83475007,
           11.64500407, 11.9605037 , 11.56842405, 12.25244484, 12.54761213,
           12.0753698 , 12.29306791, 12.64573867, 12.32987134, 12.49287059,
           11.80487366, 11.69080724, 12.21584991, 11.85668229, 11.86923822,
           11.63679077, 11.77860236, 11.78775742, 11.80089923, 11.78894245,
           11.87990798, 12.37343977, 12.03653883, 11.83491784, 12.22785904,
           12.43953086, 12.26246463, 11.74600249, 12.63349298, 12.87431877,
           13.06080658, 11.75620468, 12.70540995, 11.62458576, 11.64705765,
           12.86814853, 11.86079625, 11.58199359, 11.81332494, 12.77484118,
           12.52339649, 12.00506721, 12.43261089, 12.54586035, 12.4380172 ,
           12.48360115, 12.32904845, 11.87979011, 11.74020838, 12.09096459,
           12.34892913, 11.98366165, 12.12758784, 12.08431176, 11.98861132,
           12.60847328, 11.75155212, 12.45076967, 12.10293151, 11.97689226,
           12.14368454, 11.79774377, 12.05303545, 12.00646971, 12.33591213,
           12.28643779, 12.41494556, 11.95774327, 12.00421833, 11.99843245,
           11.9647652 , 12.21879141, 12.10845231, 12.09664037, 11.99781728,
           12.01779329, 12.49019939, 11.84319323, 12.42241655, 12.76330137,
           11.5612847 , 12.13498179, 12.91051511, 12.93458372, 13.06813793,
           12.23987464, 12.35718751, 12.75767098, 12.66661273, 12.5853782 ,
           12.85171398, 12.67327354, 12.69296043, 12.26523571, 12.86280296,
           11.5372746 , 12.63459491, 12.460234  , 12.56602158, 11.69682747,
           11.93144421, 12.90252652, 13.01505864, 12.26015443, 12.06879426,
           11.88183366, 12.627365  , 12.60835058, 11.96817206, 12.14093662,
           12.52863255, 11.66591172, 12.36531131, 11.63322576, 12.26331258,
           12.13793657, 12.46184662, 12.32360304, 12.35821974, 13.1079896 ,
           12.12601833, 12.48094921, 13.01531606, 12.72512661, 12.41355734,
           12.16203196, 11.76722583, 11.71177314, 13.08853455, 12.95421609,
           11.80914476, 12.58777566, 12.61239866, 12.01516874, 12.201543  ,
           12.56900524, 12.16047361, 12.38651239, 12.08286608, 12.39022885,
           11.92315227, 12.05013702, 12.84642555, 13.10906046, 12.73220529,
           12.19217947, 12.54723109, 12.73168158, 12.38136507, 12.47380659,
           12.85169438, 12.71683343, 11.97344242, 11.92758688, 11.69149853,
           12.2079513 , 12.01270793, 11.87129883, 12.09118476, 12.295352  ,
           12.02968859, 12.3069099 , 12.60242252, 12.37221599, 12.12606682,
           11.86395818, 11.7851681 , 11.68615759, 11.81738986, 12.07506606,
           11.87906065, 12.05577344, 11.89056974, 11.81836181, 11.88201074,
           11.7633823 , 11.98820584, 12.4570531 , 11.90657608, 11.73368541,
           11.69501859, 11.77488651, 11.72393237, 11.68429817, 11.91607808,
           12.02656647, 12.0075913 , 11.77770158, 11.88686273, 12.03896656,
           11.5337411 , 11.64592209, 12.0358346 , 11.80905121, 11.92153943,
           11.88067784, 11.85942935, 12.25755538, 12.17620471, 12.27100384,
           11.58711542, 12.00867878, 12.8726513 , 11.82879636, 11.98763849,
           12.13519045, 11.99900992, 11.67818336, 11.84531602, 11.79678449,
           12.66895781, 12.70899846, 12.27549612, 11.97089526, 12.04755484,
           12.02098749, 12.0292448 , 12.0443838 , 11.86545307, 12.05447172,
           12.28720044, 11.80534964, 12.09290309, 12.11442632, 12.05186612,
           11.82697253, 11.38859763, 11.73399521, 11.70481511, 11.82759111,
           12.03928389, 11.74708469, 11.75365893, 11.84289299, 11.846515  ,
           12.4124009 , 11.73742301, 11.68509284, 12.78825502, 11.70791534,
           12.00763603, 11.72625648, 11.86547891, 11.86297016, 12.06889407,
           12.10039052, 12.0058239 , 11.95997696, 11.83957797, 11.96222247,
           11.82684328, 11.77370602, 11.86633127, 11.8017922 , 12.0629885 ,
           11.8355805 , 12.27307458, 11.68811072, 12.29445966, 12.44335033,
           11.91318236, 11.75346174, 12.10244835, 11.97552831, 11.69673465,
           11.74550709, 11.88382346, 11.80540161, 11.84040731, 11.71241045,
           11.55513989, 11.71146781, 11.79634065, 12.85230556, 11.77217778,
           11.71241493, 11.73672142, 11.86632714, 12.65855044, 11.63418544,
           11.718379  , 11.68324946, 11.64782453, 11.98844662, 11.30502511,
           11.69115358, 11.62745486, 11.26488429, 11.76759912, 11.24988373,
           11.62689709, 11.47315092, 11.45554211, 11.20632931, 11.34893015,
           11.43963116, 11.72509037, 10.91194278, 11.88493105, 11.26463227,
           10.99806866, 11.01994735, 11.37709501, 11.61396521, 11.01383943,
           11.83409148, 11.33459154, 11.70156872, 11.1985112 , 11.11376325,
           11.07897004, 11.77817858, 10.81300289, 11.26848771, 11.49888925,
           11.54476469, 10.85794166, 11.59065204, 11.57820652, 11.06487019,
           11.22358995, 11.01339683, 11.60424342, 11.64131962, 11.43251269,
           11.52159204, 11.51604322, 11.20142249, 11.18232069, 11.8406129 ,
           11.27743768, 10.77271104, 11.53762146, 11.52449597, 11.74569093,
           11.71171406, 12.22309314, 11.75448807, 11.37690716, 11.63082461,
           11.37275373, 11.97752045, 11.67541755, 12.00298159, 11.63538061,
           11.76032219, 11.565313  , 11.96672601, 11.94093899, 11.75778859,
           11.94153544, 11.63789228, 11.75005758, 12.59320954, 11.46388315,
           12.09123614, 11.70915555, 11.48039603, 12.06235378, 12.08513038,
           11.76850439, 12.34719225, 11.7442367 , 11.72312968, 11.99391438,
           11.51918665, 11.77919937, 11.75298017, 11.35691173, 11.66622609,
           11.28998402, 11.24815108, 11.85010276, 11.74278747, 12.11321351,
           11.84586595, 11.66084919, 11.82321237, 12.90866635, 12.49122716,
           12.00412465, 11.74318883, 11.25695948, 11.70876379, 11.86146237,
           11.62695942, 11.75002961, 11.63065696, 11.78784445, 11.72223258,
           11.84190188, 11.75404343, 11.39932095, 11.75262894, 11.87536518,
           11.83801895, 11.62762359, 11.87200803, 11.79643285, 11.65821066,
           11.4320345 , 12.01972179, 11.73709277, 11.56080887, 11.63354291,
           11.40868731, 12.14651146, 11.82055785, 11.91551685, 11.73686316,
           11.53391576, 11.66930334, 11.65486464, 11.77766529, 12.42158586,
           11.71396748, 11.87227296, 11.83301322, 11.82745422, 11.62601851,
           12.12634378, 12.32518361, 11.82192778, 11.53738944, 11.63296036,
           11.53210681, 11.91174373, 11.69385953, 11.80499655, 11.72460026,
           11.97615462, 11.94580862, 11.49664393, 11.84269614, 11.66335747,
           12.06282457, 12.04456563, 11.93621122, 12.45810728, 12.30281578,
           12.21872329, 12.36781395, 12.26535882, 11.85210576, 12.00565895,
           12.22812053, 12.393234  , 12.26740032, 12.23124038, 12.03069323,
           12.26685892, 12.08892406, 12.15265783, 11.85621044, 11.89832686,
           11.89218767, 12.0233474 , 11.9279196 , 12.1886535 , 12.30996525,
           12.37676673, 12.33339371, 12.06419297, 12.3094111 , 11.93595504,
           12.24978503, 12.17999454, 12.26857967, 12.20534915, 12.0847435 ,
           12.29852226, 12.38872879, 12.50383797, 12.76335022, 12.21967581,
           12.26700734, 12.14940976, 12.45986372, 12.38634591, 11.99314317,
           12.33067084, 12.02602728, 11.75780707, 11.96497775, 12.03699754,
           12.414574  , 12.26805953, 12.11360403, 12.34031362, 12.33217221,
           12.78260336, 12.12181046, 12.12503529, 11.86050578, 12.29987755,
           12.32104603, 12.41021101, 11.97348407, 12.55256575, 12.29277736,
           12.26502543, 11.86018551, 12.30029583, 12.13553329, 12.51013609,
           12.14694901, 12.12151404, 12.05751813, 12.03129177, 12.11218285,
           11.96145777, 12.06448722, 12.08564341, 12.05486946, 12.10347664,
           12.118355  , 12.25396379, 12.06044901, 12.04105068, 12.95608027,
           12.19316577, 12.79416547, 12.63898539, 12.28063284, 12.21372879,
           12.29115089, 12.71589175, 12.68377234, 12.74294449, 12.54021648,
           12.20306108, 12.15440791, 12.33059715, 12.37186854, 12.10135839,
           12.34958407, 12.22109027, 12.26352807, 11.92141125, 12.27655323,
           12.51821162, 12.61639186, 12.42225949, 12.44177781, 11.99199986,
           12.32967395, 11.99055819, 12.39249462, 12.71638551, 12.74110665,
           12.76903019, 12.72497029, 12.68983405, 12.61302426, 12.63559182,
           12.3867496 , 12.44826856, 12.70450832, 12.29889843, 12.41434194,
           12.39113081, 12.26685609, 12.84534481, 12.45854518, 12.48322087,
           12.16923756, 12.3156901 , 12.16552161, 12.23044733, 12.34288952,
           12.15069278, 12.06598373, 12.01879902, 12.25513837, 11.93253481,
           12.29817208, 12.09147106, 12.49183874, 12.14179999, 12.14940617,
           12.2780624 , 12.19118893, 12.43773881, 12.0114907 , 12.33718161,
           12.47987004, 12.36014067, 12.38388853, 12.9909767 , 12.86325456,
           12.91963716, 12.7987961 , 12.95914468, 12.89873322, 12.78335325,
           12.61838415, 12.0267944 , 12.05563721, 12.04096479, 12.09124535,
           12.09995857, 12.31734294, 12.08567317, 12.12047897, 12.10335078,
           12.07154232, 12.42307616, 12.41805053, 12.31778828, 12.75254474,
           11.35566006, 12.12694898, 11.56843772, 12.41955368, 11.81762306,
           12.00497776, 12.03143739, 11.95189031, 11.90249885, 11.43909318,
           12.27053275, 11.24532615, 11.90782271, 11.8015708 , 12.03339795,
           11.83820481, 11.75564108, 12.51732047, 12.03144028, 11.5923035 ,
           11.89379663, 12.39342193, 11.86374432, 11.68965716, 12.16044018,
           11.95755859, 11.99961701, 12.10691933, 11.7668343 , 11.85642545,
           12.38126034, 11.96655833, 11.98423504, 11.78414266, 11.46723118,
           12.01083716, 11.64046882, 11.8909957 , 11.68598664, 11.49203506,
           12.02800535, 11.65489008, 11.66152787, 12.31209051, 12.46988946,
           11.70261413, 12.0550958 , 11.79549542, 12.06460719, 11.99452292,
           11.97613971, 11.89545488, 12.06621106, 12.11228348, 12.05801736,
           12.05599753, 11.80296219, 11.78639737, 12.25792239, 12.13564809,
           12.04402232, 12.34474003, 12.05157623, 12.19314989, 12.03777343,
           12.08262585, 12.06528895, 12.04279205, 11.94689233, 12.15022081,
           11.95965145, 12.03979229, 12.09094287, 12.08439697, 12.31599884,
           11.88163476, 11.91119427, 11.95349702, 12.3940197 , 12.05491413,
           11.85787265, 12.03901861, 12.27499992, 12.07125009, 12.03445037,
           11.9486841 , 11.81865615, 12.08126355, 12.79452622, 11.97975488,
           11.99472986, 11.81170807, 11.93264189, 12.13179486, 11.82420445,
           11.90736154, 12.20919633, 11.77389366, 11.72761841, 11.71267938,
           11.85232197, 11.9535687 , 11.91892619, 12.08135204, 11.83149064,
           12.11580786, 11.42544049, 12.00697227, 11.77294245, 11.78728799,
           11.43464019, 12.00071901, 11.93548565, 11.89545739, 12.12788312,
           11.77025248, 11.57019611, 11.79026511, 11.9780373 , 11.98000483,
           11.88867221, 11.83085056, 11.89689769, 11.60743084, 11.76758503,
           11.71943133, 12.09132661, 11.92487546, 11.7218437 , 11.70278504,
           11.70278504, 11.79276665, 11.95510165, 11.70885395, 11.82944757,
           11.82944757, 11.8005536 , 11.69475025, 12.61540291, 12.01082303,
           12.06618806, 12.04155836, 11.35433472, 11.54972642, 11.87832378,
           11.70094335, 12.15797404, 11.75613936, 11.81666625, 11.80346699,
           11.69106241, 12.61540291, 12.61540291, 11.85091405, 11.85570768,
           12.70226575, 12.02173581, 11.99871916, 12.26921834, 12.44192532,
           12.29525452, 12.52687753, 12.84343213, 12.17688409, 12.07662668,
           12.58390842, 11.74764773, 12.13262184, 11.99243441, 12.23301318,
           12.62797739, 12.33525066, 11.81739011, 11.88800593, 12.21715815,
           12.25922411, 12.49460591, 12.12417727, 12.30785752, 11.77536241,
           12.16009468, 12.10468292, 12.3511855 , 11.64456797, 11.73331801,
           11.85091405, 12.13272462, 12.34193862, 12.02560998, 12.35774541,
           12.7647359 , 12.31713164, 12.33437013, 12.16136007, 12.15788634,
           12.32576054, 12.4217767 , 11.86205175, 12.31713164, 11.97511027,
           12.0451666 , 12.20578179, 12.25442489, 12.15788634, 11.85570768,
           11.83531925, 12.26246524, 11.85474337, 12.16135828, 12.29491308,
           11.95274177, 12.04516922, 11.918818  , 12.04874804, 11.94498708,
           12.10524862, 11.97792644, 11.46605126, 11.6037639 , 11.43057625,
           12.00500188, 11.96374442, 11.95801308, 11.87627956, 11.58695054,
           11.93638136, 12.64075603, 12.03059427, 12.13018192, 12.30289332,
           12.07136231, 12.05116787, 12.05507841, 12.03160174, 11.66764585,
           11.53147273, 12.07159494, 11.59828766, 11.83747191, 11.9595702 ,
           11.98432874, 11.31851808, 11.53108938, 11.40284866, 11.9672789 ,
           11.34962032, 11.96544182, 11.28521593, 11.98766493, 11.45349721,
           11.5526763 , 11.56371473, 11.77029351, 11.44853234, 11.9465705 ,
           11.83445998, 11.72076553, 11.85735833, 11.38185908, 11.30683265,
           11.45783127, 11.47862494, 11.25286278, 11.89315835, 11.47424406,
           11.3530769 , 11.78600489, 11.76226404, 12.00025028, 11.51360604,
           11.87628646, 11.75218368, 11.94996959, 11.62596222, 12.02793507,
           11.23752277, 11.70436107, 11.5268951 , 11.41216593])




```python
predictions_df = pd.DataFrame({'parcel': holdout['parcel'], 'prediction': holdout_predictions})
predictions_df.to_csv("submission/MY_PREDICTIONS.csv", index=False)
```


```python
train_predictions = best_params.predict(X_train)
train_r2 = r2_score(y_train, train_predictions)
print("Training R^2 score:", train_r2)


```

    Training R^2 score: 0.9666066087554025

