import numpy as np
from matplotlib import pyplot as plt
import pandas as pd
import seaborn as sns
from scipy.special import binom
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import confusion_matrix, accuracy_score, precision_score, recall_score, f1_score, roc_curve
from sklearn import tree
from sklearn.naive_bayes import GaussianNB
from sklearn import svm
from sklearn.ensemble import RandomForestClassifier
import time
import xgboost as xgb



%matplotlib qt


def standardize(X):
    X[:,0:3] = (X[:,0:3] - np.mean(X[:,0:3],axis=0)) / np.std(X[:,0:3],axis=0)
    
    return X


def show_histogram(y):
    n_t = np.count_nonzero(y == 1.) # number of true values
    n_f = y.shape[0] - n_t
    
    per_t = n_t / y.shape[0]
    per_f = n_f / y.shape[0]
    
    # labels = 'Није превара', 'Јесте превара'
    # explode = np.zeros(2)
    # explode[1] = 0.1
    # plt.figure()
    # plt.pie([per_f,per_t], explode=explode,colors=['lime','red'], autopct='%.2f%%',
    #     shadow=True, startangle=180)
    # plt.title('Уравнотеженост података')
    # plt.legend(labels,loc='lower right')
    # plt.show()
    
    
    return per_t, per_f, n_t, n_f


def mean_var_predictors(X, y, n_ct):
    mean_var_t = np.zeros((2,n_ct))
    mean_var_f = np.zeros((2,n_ct))
    
    Xt = X[(y==1).reshape(y.shape[0],),0:n_ct]
    Xf = X[(y==0).reshape(y.shape[0],),0:n_ct]
    
    mean_var_t[0,:] = np.mean(Xt,axis=0)
    mean_var_t[1,:] = np.std(Xt,axis=0)# ** 0.5
    mean_var_f[0,:] = np.mean(Xf,axis=0)
    mean_var_f[1,:] = np.std(Xf,axis=0)# ** 0.5
    
    for i in range(n_ct):
        plt.figure()
        plt.bar('Није',0)
        plt.bar('Јесте',0)
        plt.bar(-0.125,mean_var_f[0,i],color='royalblue',width=0.25,label="средња вредност")
        plt.bar(0.125,mean_var_f[1,i],color='orange',width=0.25,label='варијанса')
        plt.bar(0.875,mean_var_t[0,i],color='royalblue',width=0.25)
        plt.bar(1.125,mean_var_t[1,i],color='orange',width=0.25)
        plt.title('{}. обележје'.format(i+1))
        plt.xlabel('$Превара$')
        plt.ylabel('$Средња$ $вредност,$ $варијанса$')
        plt.legend()
        plt.show()


def preds_y_correlation(X, y, n_ct):
    # correlation between predictors
    cor01 = np.abs(correlation(X[:,0:1],X[:,1:2]))
    cor02 = np.abs(correlation(X[:,0:1],X[:,2:3]))
    cor12 = np.abs(correlation(X[:,1:2],X[:,2:3]))
    
    plt.figure()
    plt.bar('1. и 2.',cor01,color='royalblue',width=0.25)
    plt.bar('1. и 3.',cor02,color='royalblue',width=0.25)
    plt.bar('2. и 3.',cor12,color='royalblue',width=0.25)
    plt.title('Међусобна корелисаност обележја')
    plt.xlabel('$Комбинације$ $обележја$')
    plt.ylabel('$Коефицијент$ $корелације$')
    plt.show()
    
    
    # correlation between predictors and target
    p_y_corr = np.zeros(n_ct)
    for i in range(n_ct):
        p_y_corr[i] = correlation(X[:,i:i+1], y)
    
    plt.figure()
    plt.bar('1. и y',p_y_corr[0],color='royalblue',width=0.25)
    plt.bar('2. и y',p_y_corr[1],color='royalblue',width=0.25)
    plt.bar('3. и y',p_y_corr[2],color='royalblue',width=0.25)
    plt.title('Корелисаност обележја и циљне променљиве')
    plt.xlabel('$Ред.$ $бр.$ $обележја$')
    plt.ylabel('$Коефицијент$ $корелације$')
    plt.show()


def correlation(X, Y):
    return (np.mean(X*Y) - np.mean(X)*np.mean(Y))/np.sqrt((np.mean(X*X)-np.mean(X)**2)*(np.mean(Y*Y)-np.mean(Y)**2))


def predictors_hist(X, y, n_ct):
    # pred_hist = np.zeros((2,n_ct)) # 1st row is for true, 2nd row is for false
    Xt = X[(y==1).reshape(y.shape[0],),0:n_ct]
    Xf = X[(y==0).reshape(y.shape[0],),0:n_ct]
    
    n_of_bins = 1000
    
    for i in range(n_ct):
        plt.figure()
        plt.hist(Xf[:,i:i+1],n_of_bins,alpha=0.8,color='orange')
        plt.hist(Xt[:,i:i+1],n_of_bins,alpha=0.9,color='royalblue')
        plt.title('Хистограм \n{}. обележја'.format(i+1))
        plt.legend(['Није превара','Јесте превара'])
        plt.show()


def class_by_balance(X,per_t):
    m = X.shape[0]
    
    #y = np.random.uniform(size=(m,1))
    y = np.zeros((m,1))
    
    for i in range(m):
        y[i] = np.random.uniform()
    
    y = y < per_t
    y = np.float16(y)
    
    return y


def class_ipt(X,t): # classification by informative predictor with threshold
    y = X > t
    y = np.float16(y)
    
    return y


def cm(y_pred, y_true):
    cm = np.zeros((2,2))
    for i in range(y_pred.shape[0]):
        if y_pred[i,0] == y_true[i,0]:
            if y_pred[i,0] == 1:
                cm[0,0] += 1
            else:
                cm[1,1] += 1
        else:
            if y_pred[i,0] == 1:
                cm[0,1] += 1
            else:
                cm[1,0] += 1
    
    return cm

def acc(cm):
    TP = cm[0,0]
    TN = cm[1,1]
    FN = cm[1,0]
    FP = cm[0,1]
    
    return (TP+TN)/(TP+TN+FN+FP)



# ---------------------------------main----------------------------------------
if __name__ == '__main__':
    
    # --- reading data ---
    
    data = pd.read_csv('card_transdata.csv')
    data1 = pd.DataFrame(data).to_numpy() # conversion from DataFrame to np type
    
    r,c = data1.shape
    
    X = np.zeros((r,c-1))
    y = np.zeros((r,1))
    X[:,0:c-1] = data1[:,0:c-1]
    y[:,0] = data1[:,c-1]
    
    # --- reading data ---
    
    
    
    
    # --- exploratory data analysis ---
    
    
    n_ct = 3 # number of continuous predictors
    per_t, per_f, n_t, n_f = show_histogram(y)
    # mean_var_predictors(X, y, n_ct)
    # preds_y_correlation(X, y, n_ct)
    # predictors_hist(X, y, n_ct)
    
    
    # --- exploratory data analysis ---
    
    
    threshold = (4. - np.mean(X[:,2:3],axis=0)) / np.std(X[:,2:3],axis=0) # for the second simple model (4 <=> 0.78)
    X = standardize(X)
    # train_test split
    X_train, X_test, y_train, y_test = train_test_split(X, y, stratify=y, test_size=0.2)
    
    
    
    
    
    # --- models ---
    
    
    # classification by balance
    per_t1, per_f1, n_t1, n_f1 = show_histogram(y_train)
    y_pred_by_balance = class_by_balance(X_test, per_t1)
    print('--------------------')
    print('Класификација на основу бројности\n')
    cm = confusion_matrix(y_true=y_test, y_pred=y_pred_by_balance)
    print('Матрица конфузије:')
    print(cm)
    print('Тачност: {}'.format(accuracy_score(y_true=y_test, y_pred=y_pred_by_balance)))
    print('Прецизност: {}'.format(precision_score(y_true=y_test, y_pred=y_pred_by_balance)))
    print('Осетљивост: {}'.format(recall_score(y_true=y_test, y_pred=y_pred_by_balance)))
    print('F1-скор: {}\n\n'.format(f1_score(y_true=y_test, y_pred=y_pred_by_balance)))
    
    
    
    
    # classification by the most informative predictor and convenient threshold
    y_pred_ipt = class_ipt(X_test[:,2:3], threshold) # informative predictor threshold
    print('--------------------')
    print('Класификација помоћу најинформативнијег обележја и погодног прага\n')
    cm = confusion_matrix(y_true=y_test, y_pred=y_pred_ipt)
    print('Матрица конфузије:')
    print(cm)
    print('Тачност: {}'.format(accuracy_score(y_true=y_test, y_pred=y_pred_ipt)))
    print('Прецизност: {}'.format(precision_score(y_true=y_test, y_pred=y_pred_ipt)))
    print('Осетљивост: {}'.format(recall_score(y_true=y_test, y_pred=y_pred_ipt)))
    print('F1-скор: {}\n\n'.format(f1_score(y_true=y_test, y_pred=y_pred_ipt)))
    
    
    
    
    # logistic regression
    log_reg = LogisticRegression()
    # # log_reg.fit(np.concatenate((X_train[:,0:3],X_train[:,4:7]),axis=1), y_train.reshape(y_train.shape[0]))
    # # y_pred_log = log_reg.predict(np.concatenate((X_test[:,0:3],X_test[:,4:7]),axis=1))
    log_reg.fit(X_train, y_train.reshape(y_train.shape[0]))
    y_pred_log = log_reg.predict(X_test)
    print('--------------------')
    print('Логистичка регресија\n')
    cm = confusion_matrix(y_true=y_test, y_pred=y_pred_log)
    print('Матрица конфузије:')
    print(cm)
    print('Тачност: {}'.format(accuracy_score(y_true=y_test, y_pred=y_pred_log)))
    print('Прецизност: {}'.format(precision_score(y_true=y_test, y_pred=y_pred_log)))
    print('Осетљивост: {}'.format(recall_score(y_true=y_test, y_pred=y_pred_log)))
    print('F1-скор: {}\n\n'.format(f1_score(y_true=y_test, y_pred=y_pred_log)))
    
    
    y_pred_log_proba = log_reg.predict_proba(X_test)[::,1]
    fpr, tpr, _ = roc_curve(y_test, y_pred_log_proba)
    
    plt.figure()
    plt.plot(fpr, tpr)
    plt.xlabel('$Стопа$ $лажних$ $аларма$')
    plt.ylabel('$Стопа$ $тачних$ $аларма$')
    plt.title('ROC крива')
    plt.show()
    
    
    
    
    # GNB model
    t0 = time.time()
    clf_gnb = GaussianNB()
    clf_gnb.fit(X_train, y_train)
    y_pred_gnb = clf_gnb.predict(X_test).reshape(-1,1)
    print('--------------------')
    print('Гаусовски наивни Бејз\n')
    cm = confusion_matrix(y_true=y_test, y_pred=y_pred_gnb)
    print('Матрица конфузије:')
    print(cm)
    print('Тачност: {}'.format(accuracy_score(y_true=y_test, y_pred=y_pred_gnb)))
    print('Прецизност: {}'.format(precision_score(y_true=y_test, y_pred=y_pred_gnb)))
    print('Осетљивост: {}'.format(recall_score(y_true=y_test, y_pred=y_pred_gnb)))
    print('F1-скор: {}\n'.format(f1_score(y_true=y_test, y_pred=y_pred_gnb)))
    t1 = time.time()
    print('Време извршења: {} s\n\n'.format(t1-t0))
    
    
    
    
    # # SVM - not suitable for large data sets
    # t0 = time.time()
    # clf_svm = svm.SVC(kernel='rbf') # rbf is Gaussian kernel!
    # clf_svm.fit(X_train, y_train.reshape(y_train.shape[0]))
    # y_pred_svm = clf_svm.predict(X_test).reshape(-1,1)
    # print('--------------------')
    # print('Метода носећих вектора\n')
    # cm = confusion_matrix(y_true=y_test, y_pred=y_pred_svm)
    # print('Матрица конфузије:')
    # print(cm)
    # print('Тачност: {}'.format(accuracy_score(y_true=y_test, y_pred=y_pred_svm)))
    # print('Прецизност: {}'.format(precision_score(y_true=y_test, y_pred=y_pred_svm)))
    # print('Осетљивост: {}'.format(recall_score(y_true=y_test, y_pred=y_pred_svm)))
    # print('F1-скор: {}\n'.format(f1_score(y_true=y_test, y_pred=y_pred_svm)))
    # t1 = time.time()
    # print('Време извршења: {} s\n\n'.format(t1-t0))
    
    
    
    
    # decision tree
    t0 = time.time()
    clf_tree = tree.DecisionTreeClassifier(max_depth=30)
    clf_tree = clf_tree.fit(X_train, y_train)
    y_pred_tree = clf_tree.predict(X_test).reshape(-1,1)
    print('--------------------')
    print('Стабло одлуке\n')
    cm = confusion_matrix(y_true=y_test, y_pred=y_pred_tree)
    print('Матрица конфузије:')
    print(cm)
    print('Тачност: {}'.format(accuracy_score(y_true=y_test, y_pred=y_pred_tree)))
    print('Прецизност: {}'.format(precision_score(y_true=y_test, y_pred=y_pred_tree)))
    print('Осетљивост: {}'.format(recall_score(y_true=y_test, y_pred=y_pred_tree)))
    print('F1-скор: {}\n'.format(f1_score(y_true=y_test, y_pred=y_pred_tree)))
    t1 = time.time()
    print('Време извршења: {} s\n\n'.format(t1-t0))
    tree.plot_tree(clf_tree)
    
    
    
    
    # random forrest
    t0 = time.time()
    clf_rf = RandomForestClassifier(n_estimators=100, max_depth=3, max_features=2)
    clf_rf.fit(X_train, y_train.reshape(y_train.shape[0]))
    y_pred_rf = clf_rf.predict(X_test).reshape(-1,1)
    print('--------------------')
    print('Случајна шума\n')
    cm = confusion_matrix(y_true=y_test, y_pred=y_pred_rf)
    print('Матрица конфузије:')
    print(cm)
    print('Тачност: {}'.format(accuracy_score(y_true=y_test, y_pred=y_pred_rf)))
    print('Прецизност: {}'.format(precision_score(y_true=y_test, y_pred=y_pred_rf)))
    print('Осетљивост: {}'.format(recall_score(y_true=y_test, y_pred=y_pred_rf)))
    print('F1-скор: {}\n'.format(f1_score(y_true=y_test, y_pred=y_pred_rf)))
    t1 = time.time()
    print('Време извршења: {} s\n\n'.format(t1-t0))
    
    
    
    
    # XGBoost
    t0 = time.time()
    clf_xgb = xgb.XGBClassifier(n_estimators=10, max_depth=4, max_features=3)
    clf_xgb.fit(X_train, y_train.reshape(y_train.shape[0]))
    print(clf_xgb)
    y_pred_xgb = clf_xgb.predict(X_test).reshape(-1,1)
    print('--------------------')
    print('XGBoost\n')
    cm = confusion_matrix(y_true=y_test, y_pred=y_pred_xgb)
    print('Матрица конфузије:')
    print(cm)
    print('Тачност: {}'.format(accuracy_score(y_true=y_test, y_pred=y_pred_xgb)))
    print('Прецизност: {}'.format(precision_score(y_true=y_test, y_pred=y_pred_xgb)))
    print('Осетљивост: {}'.format(recall_score(y_true=y_test, y_pred=y_pred_xgb)))
    print('F1-скор: {}\n'.format(f1_score(y_true=y_test, y_pred=y_pred_xgb)))
    t1 = time.time()
    print('Време извршења: {} s\n\n'.format(t1-t0))
    
    
    # --- models ---
    
    
    
    
    
    
    
    
    
    
    
    
    
 

    
    
    
    
    
    
    
    
    
    
