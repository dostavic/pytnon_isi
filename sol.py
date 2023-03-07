import numpy as np
import pandas as pd
from sklearn.discriminant_analysis import QuadraticDiscriminantAnalysis, LinearDiscriminantAnalysis
from sklearn.ensemble import RandomForestClassifier, AdaBoostRegressor
from sklearn.impute import SimpleImputer
from sklearn.linear_model import Ridge, LogisticRegression, RidgeClassifier, SGDClassifier
from sklearn.metrics import accuracy_score, r2_score
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn import tree, metrics
from sklearn.preprocessing import LabelEncoder, OneHotEncoder, StandardScaler
from sklearn import svm
from sklearn.svm import SVC, NuSVC
from sklearn.tree import DecisionTreeClassifier

X_eval = np.load('X_eval_ut_7_30.npy', allow_pickle=True)
X_public = np.load('X_public_ut_7_30.npy', allow_pickle=True)
y_public = np.load('y_public_ut_7_30.npy', allow_pickle=True)

# df=pd.DataFrame(X_public)
# imp_mean = SimpleImputer(missing_values=np.nan, strategy='mean')
# imp_mean.fit(X_public)
# X_public = imp_mean.transform(X_public)
# X_eval = imp_mean.transform(X_eval)

# ohe = OneHotEncoder(handle_unknown='ignore')

for x in range(169, 198):
    enc = LabelEncoder()
    label_encoder = enc.fit(X_public[:, x])
    integer_classes = label_encoder.transform(label_encoder.classes_)
    X_public[:, x] = label_encoder.transform(X_public[:, x])

for x in range(169, 198):
    enc = LabelEncoder()
    label_encoder = enc.fit(X_eval[:, x])
    integer_classes = label_encoder.transform(label_encoder.classes_)
    X_eval[:, x] = label_encoder.transform(X_eval[:, x])
#
# # for x in X_public:
for x in range(len(X_public[0])):
    num = X_public[:, x]
    num = np.array(num)
    mean_num = np.mean(num[~pd.isnull(num)])
    num[pd.isnull(num)] = mean_num
    X_public[:, x] = num

    num = X_eval[:, x]
    num = np.array(num)
    mean_num = np.mean(num[~pd.isnull(num)])
    num[pd.isnull(num)] = mean_num
    X_eval[:, x] = num

scaler = StandardScaler()
scaler.fit(X_public)
X_public = scaler.transform(X_public)

scaler.fit(X_eval)
X_eval = scaler.transform(X_eval)

X_train, X_test, y_train, y_test = train_test_split(X_public, y_public, test_size=0.25, random_state=0)

# c = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1] # c= 0.2
# for i in c:
#     svc = SVC(kernel= 'linear', C=i)
#     svc.fit(X_train, y_train)
#     y_predict = svc.predict(X_test)
#     print("SVC(linear):", accuracy_score(y_test, y_predict))

y_predict_max = 0

# clf = SGDClassifier(loss='perceptron', alpha=0.1)
# clf.fit(X_train, y_train)
# y_predict = clf.predict(X_test)
# print("SGDClassifier:", accuracy_score(y_test,  y_predict))

reg = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]
for i in reg:
    svc = NuSVC(kernel= 'poly', degree=2, nu = i)
    svc.fit(X_train, y_train)
    y_predict = svc.predict(X_test)
    if type(y_predict_max) == int:
        y_predict_max = svc.predict(X_test)
    # print("NuSVC(poly):", accuracy_score(y_test, y_predict))
    if accuracy_score(y_test, y_predict) > accuracy_score(y_test, y_predict_max):
        y_predict_max = y_predict

print("NuSVC(poly):", accuracy_score(y_test, y_predict_max))
y_predict_max = 0

d = [1, 2, 3, 10, 100]
c = [0.1, 1, 10, 100]
# g = [0.1, 1, 10, 100] #default
for i in d:
    for j in c:
        svc = SVC(kernel= 'poly', degree=i, C=j) #0.95
        svc.fit(X_train, y_train)
        y_predict = svc.predict(X_test)
        if type(y_predict_max) == int:
            y_predict_max = svc.predict(X_test)
        # print("SVC(poly):", accuracy_score(y_test, y_predict))
        if accuracy_score(y_test, y_predict) > accuracy_score(y_test, y_predict_max):
            y_predict_max = y_predict
            # print(i, j)


print("SVC(poly) MAX:", accuracy_score(y_test, y_predict_max))
y_predict_max = 0
# c = [0.1, 1, 10, 100] #c = 10
# for i in c:
# svc = SVC(kernel= 'rbf', C=10) #0.81
# svc.fit(X_train, y_train)
# y_predict = svc.predict(X_test)
# print("SVC(poly):", accuracy_score(y_test, y_predict))

# for i in range(1, 100):
# rf =  (n_estimators=1000, criterion='log_loss', min_samples_leaf=2) #0.81
# rf.fit(X_train, y_train)
# y_predict = rf.predict(X_test)
# print("RandomForestClassifier:", accuracy_score(y_test, y_predict))

# for i in range(2, 11):
#     clf = tree.DecisionTreeClassifier(criterion='gini', max_depth=9, min_samples_leaf=4)
#     clf = clf.fit(X_train,y_train)
#     y_predict = clf.predict(X_test)
#     print("DecisionTreeClassifier:", accuracy_score(y_test, y_predict))
# print (metrics.accuracy_score(y_test,y_pred_test),"\n")

reg = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]
for i in reg:
    clf = QuadraticDiscriminantAnalysis(reg_param = i)
    clf.fit(X_train, y_train)
    y_predict = clf.predict(X_test)
    # print("QuadraticDiscriminantAnalysis:", accuracy_score(y_test, y_predict))
    if type(y_predict_max) == int:
        y_predict_max = clf.predict(X_test)
    if accuracy_score(y_test, y_predict) > accuracy_score(y_test, y_predict_max):
        y_predict_max = y_predict
        # print(i)

print("QuadraticDiscriminantAnalysis MAX:", accuracy_score(y_test, y_predict_max))

# y_predict_
# clf = LinearDiscriminantAnalysis(solver='eigen')
# clf.fit(X_train, y_train)
# y_predict = clf.predict(X_test)
# print("LinearDiscriminantAnalysis:", accuracy_score(y_test, y_predict))

clf = QuadraticDiscriminantAnalysis(reg_param = 0.1)
clf.fit(X_train, y_train)
y_predicted = clf.predict(X_eval)

np.save("y_predikcia.npy", y_predicted)
# print("QuadraticDiscriminantAnalysis (fin):", accuracy_score(y_test, y_predict))
# y_eval = np.load('y_eval.npy', allow_pickle=True)