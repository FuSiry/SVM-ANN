import sklearn.svm as svm
import sklearn.metrics as sm
import sklearn.model_selection as ms
from sklearn.metrics import accuracy_score
from Code.Dataload import TableDataLoad




def SVM_Classs_test(train_x ,train_y,test_x,test_y):
    params = [
            {'kernel': ['linear'], 'C': [ 0.1,0.5, 1, 1.5,2,3,5, 10, 15,50,100],'gamma': [1e-7,1e-6,1e-5, 1e-4, 1e-3, 1e-2, 1e-1, 1]},
            {'kernel': ['poly'], 'C': [0.1, 1, 2, 10, 15], 'degree': [2, 3]}
            # {'kernel': ['rbf'], 'C': [0.1, 1, 2, 10], 'gamma':[1e-3, 1e-2, 1e-1, 1, 2]}
            ]

    model = ms.GridSearchCV(svm.SVC(probability=True),
                            params,
                            refit=True,
                            return_train_score=True,        # 后续版本需要指定True才有score方法
                            cv=10)
    model.fit(train_x, train_y)
    model_best = model.best_estimator_
    pred_test_y = model_best.predict(test_x)
    acc_test = accuracy_score(test_y, pred_test_y)
    print("best pamer is {}".format(model.best_estimator_))
    print("acc  is {}".format(acc_test))
    print(sm.classification_report(test_y, pred_test_y))

def SVM(train_x ,train_y,test_x,test_y):
    clf = svm.SVC(probability=True, C=10, gamma=1e-7,kernel='linear')
    clf.fit(train_x, train_y)
    pred = clf.predict(test_x)
    acc_test = accuracy_score(test_y, pred)
    return acc_test

if __name__ == '__main__':
    test_ratio = 0.3
    tp = 'raw'

    X_train, X_test, y_train, y_test = TableDataLoad(tp=tp, test_ratio=test_ratio, start=0, end=404, seed=80)

    SVM_Classs_test(train_x=X_train, train_y=y_train, test_x=X_test, test_y=y_test)

    acc = SVM(train_x=X_train, train_y=y_train, test_x=X_test, test_y=y_test)
    print(acc)