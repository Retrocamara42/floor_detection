# Training and choosing a model
import pandas as pd
from sklearn.model_selection import train_test_split
import xgboost as xgb
from xgboost import XGBClassifier
from sklearn.metrics import mean_absolute_error
from sklearn.externals import joblib
from graphConfusionMatrix import graficarMatrizConfusion

x = pd.read_csv("./input/X_train.csv")
y = pd.read_csv("./input/y_train.csv")

# Data processing
# X train - 70, X val - 20, X test - 10
x_train = x[x['series_id']<=2660]
x_train = x_train.drop(['row_id'], axis=1)

x_not_train = x[x['series_id']>2660]
x_not_train = x_not_train.drop(['row_id'], axis=1)

x_val = x_not_train[x_not_train['series_id']<=3047]
x_test = x_not_train[x_not_train['series_id']>3047]

y_1 = y[y['series_id']<=2660]
y_v = y[y['series_id']<=3047]
y_t = y[y['series_id']>3047]

train_data = x_train.set_index('series_id').join(y_1.set_index('series_id'))
y_train = train_data['surface']

y_val = x_val.set_index('series_id').join(y_v.set_index('series_id'))
y_val = y_val['surface']

y_test = x_test.set_index('series_id').join(y_t.set_index('series_id'))
y_test = y_test['surface']

# Training
model = XGBClassifier(
 learning_rate =0.1,
 n_estimators=1000,
 max_depth=10,
 min_child_weight=1,
 gamma=0,
 subsample=0.8,
 colsample_bytree=0.8,
 objective= 'binary:logistic',
 nthread=4,
 scale_pos_weight=1,
 seed=27)
#model.fit(x_train, y_train, early_stopping_rounds=10, eval_set=[(x_val, y_val)], verbose=True)
model = joblib.load('./model/mod1.model')

# Validation
predictions = model.predict(x_test)
print("Accuracy : " + str(sum(predictions == y_test)/len(predictions)))

clases = ["fine_concrete", "concrete", "soft_tiles", "tiled", "soft_pvc",
 "hard_tiles_large_space", "carpet", "hard_tiles", "wood"]
graficarMatrizConfusion(y_test, predictions, clases,
                          normalize=True,
                          title='Confusion matrix');

# Saving the model
joblib.dump(model, './model/mod1.model')
