#!/usr/bin/env python2.7
# -*- coding: utf-8 -*-

from pandas import read_csv, DataFrame, Series, concat
from sklearn.preprocessing import LabelEncoder
from sklearn import cross_validation, svm, grid_search

from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.ensemble import GradientBoostingClassifier

from sklearn.tree import DecisionTreeClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import GaussianNB

from sklearn.metrics import roc_curve, auc
from sklearn.metrics import accuracy_score, roc_auc_score

import pylab as pl
import matplotlib.pyplot as plt

def plot_train():
    print 'Plot data...'
    data = read_csv('./train.csv', sep = ',')

    for k in range(1, 5):
        param = 'Wilderness_Area%s' % k
        f = plt.figure(figsize = (8, 6))
        p = data.pivot_table('Id', param, 'Cover_Type', 'count').plot(kind = 'barh', stacked = True, ax = f.gca())
        img = './wilderness_area_cover_type_plot/Wilderness_Area%s_cover_type.png' % k
        f.savefig(img)
    
    for k in range(1, 41):
        param = 'Soil_Type%s' % k
        f = plt.figure(figsize = (8, 6))
        p = data.pivot_table('Id', param, 'Cover_Type', 'count').plot(kind = 'barh', stacked = True, ax = f.gca())
        img = './soil_type_cover_type_plot/Soil_Type%s_cover_type.png' % k
        f.savefig(img)

def plot_elevation():
    data = read_csv('./train.csv')
    data = data.sort(['Elevation'])

    print 'Plot Elevation...'
    fig, axes = plt.subplots(ncols=1)
    e = data.pivot_table('Id', ['Elevation'], 'Cover_Type', 'count').plot(ax=axes, title='Elevation')
    f = e.get_figure()
    f.savefig('./train_data_plot/elevation_cover_type.png')

# def plot_box_elevation():
#     print "plot box elevation..."
#     data = read_csv("./train.csv")
#     df = concat([data['Elevation'], data['Cover_Type']], axis=1, keys=['Elevation', 'Cover_Type'])
#     f = plt.figure(figsize=(8, 6))
#     p = df.boxplot(by='Cover_Type', ax = f.gca())
#     f.savefig('./train_data_plot/box_elevation_cover_type.png')

def plot_aspect():
    data = read_csv('./train.csv')
    data = data.sort(['Aspect'])

    print 'Plot Aspect...'
    fig, axes = plt.subplots(ncols=1)
    e = data.pivot_table('Id', ['Aspect'], 'Cover_Type', 'count').plot(ax=axes, title='Aspect')
    f = e.get_figure()
    f.savefig('./train_data_plot/aspect_cover_type.png')

def plot_slope():
    data = read_csv('./train.csv')
    data = data.sort(['Slope'])

    print 'Plot Slope...'
    fig, axes = plt.subplots(ncols=1)
    e = data.pivot_table('Id', ['Slope'], 'Cover_Type', 'count').plot(ax=axes, title='Slope')
    f = e.get_figure()
    f.savefig('./train_data_plot/slope_cover_type.png')

def plot_horizontal_distance_to_hydrology():
    data = read_csv('./train.csv')
    data = data.sort(['Horizontal_Distance_To_Hydrology'])

    print 'Plot Horizontal_Distance_To_Hydrology...'
    fig, axes = plt.subplots(ncols=1)
    e = data.pivot_table('Id', ['Horizontal_Distance_To_Hydrology'], 'Cover_Type', 'count').plot(ax=axes, title='Horizontal Distance To Hydrology')
    f = e.get_figure()
    f.savefig('./train_data_plot/horizontal_distance_to_hydrology_cover_type.png')

def plot_vertical_distance_to_hydrology():
    data = read_csv('./train.csv')
    data = data.sort(['Vertical_Distance_To_Hydrology'])

    print 'Plot Vertical_Distance_To_Hydrology...'
    fig, axes = plt.subplots(ncols=1)
    e = data.pivot_table('Id', ['Vertical_Distance_To_Hydrology'], 'Cover_Type', 'count').plot(ax=axes, title='Vertical Distance To Hydrology')
    f = e.get_figure()
    f.savefig('./train_data_plot/vertical_distance_to_hydrology_cover_type.png')

def plot_horizontal_distance_to_roadways():
    data = read_csv('./train.csv')
    data = data.sort(['Horizontal_Distance_To_Roadways'])

    print 'Plot Horizontal_Distance_To_Roadways...'
    fig, axes = plt.subplots(ncols=1)
    e = data.pivot_table('Id', ['Horizontal_Distance_To_Roadways'], 'Cover_Type', 'count').plot(ax=axes, title='Horizontal Distance To Roadways')
    f = e.get_figure()
    f.savefig('./train_data_plot/horizontal_distance_to_roadways_cover_type.png')

def plot_box():

    data = read_csv("./train.csv")
    headers = ['Elevation', 'Aspect', 'Horizontal_Distance_To_Hydrology', 'Vertical_Distance_To_Hydrology', 'Horizontal_Distance_To_Roadways']

    for k in headers:
        print "box plot %s..." % k.lower().replace("_", " ")
        df = concat([data[k], data['Cover_Type']], axis=1, keys=[k, 'Cover_Type'])
        f = plt.figure(figsize=(8, 6))
        p = df.boxplot(by='Cover_Type', ax = f.gca())
        f.savefig('./train_data_plot/box_%s_cover_type.png' % k.lower())


def get_train_data():

    print 'Get train data...'
    data = read_csv('./train.csv')
    data = data.drop(['Id'], axis = 1)

    # удаляем столбец Wilderness_Area2
    data = data.drop(['Wilderness_Area2'], axis = 1)

    # удаляем столбцы SoilType1,...,SoilType40
    drop_soil_type_cols = []
    for k in range(1, 41):
        cname = 'Soil_Type%s' % k
        drop_soil_type_cols.append(cname)
    data = data.drop(drop_soil_type_cols, axis = 1)

    return data

def get_test_data():

    print 'Get test data...'
    data = read_csv('./test.csv')
    result = DataFrame(data.Id)

    # удаляем столбцы Id, Wilderness_Area2
    data = data.drop(['Id', 'Wilderness_Area2'], axis = 1)

    # удаляем столбцы SoilType1,...,SoilType40
    drop_soil_type_cols = []
    for k in range(1, 41):
        cname = 'Soil_Type%s' % k
        drop_soil_type_cols.append(cname)
    data = data.drop(drop_soil_type_cols, axis = 1)

    return (data, result)

def cross_validation_test():
    data = get_train_data()
    target = data.Cover_Type
    train = data.drop(['Cover_Type'], axis = 1)
    kfold = 10
    cross_val_final = {}

    print 'Cross validation test...'
    model_rfc = RandomForestClassifier(n_estimators = 1000, criterion='entropy', n_jobs = -1)
    # model_etc = ExtraTreesClassifier(n_estimators = 100)
    # model_dtc = DecisionTreeClassifier(max_depth = None, min_samples_split=1, random_state=0)
    # model_knc = KNeighborsClassifier(n_neighbors = 18)
    model_lr = LogisticRegression(penalty='l1', C=1e5)
    # model_nbc = GaussianNB()
    model_svc = svm.SVC(kernel = "poly", degree = 2)

    scores = cross_validation.cross_val_score(model_rfc, train, target, cv = kfold)
    cross_val_final['RFC'] = scores.mean()
    print 'RFC: ', scores.mean()

    # scores = cross_validation.cross_val_score(model_nbc, train, target, cv = kfold)
    # cross_val_final['NBC'] = scores.mean()
    # print 'NBC: ', scores.mean()

    # scores = cross_validation.cross_val_score(model_etc, train, target, cv = kfold)
    # cross_val_final['ETC'] = scores.mean()

    # scores = cross_validation.cross_val_score(model_dtc, train, target, cv = kfold)
    # cross_val_final['DTC'] = scores.mean()

    # scores = cross_validation.cross_val_score(model_knc, train, target, cv = kfold)
    # cross_val_final['KNC'] = scores.mean()

    scores = cross_validation.cross_val_score(model_svc, train, target, cv = kfold)
    cross_val_final['SVC'] = scores.mean()
    print 'SVM: ', scores.mean()

    scores = cross_validation.cross_val_score(model_lr, train, target, cv = kfold)
    cross_val_final['LR'] = scores.mean()
    print 'LR: ', scores.mean()

    f = plt.figure(figsize = (8, 6))
    p = DataFrame.from_dict(data = cross_val_final, orient='index').plot(kind='barh', legend=False, ax = f.gca())
    f.savefig('./test_plot/cross_validation_rfc_svm.png')

def get_scores():
    data = get_train_data()
    target = data.Cover_Type
    train = data.drop(['Cover_Type'], axis = 1)
    all_scores = {}

    print 'Get scores...'

    model_rfc = RandomForestClassifier(n_estimators = 100)
    model_etc = ExtraTreesClassifier(n_estimators = 100)
    model_dtc = DecisionTreeClassifier(max_depth = None, min_samples_split=1, random_state=0)
    model_knc = KNeighborsClassifier(n_neighbors = 18)
    model_svc = svm.SVC()

    TRNtrain, TRNtest, TARtrain, TARtest = cross_validation.train_test_split(train, target, test_size=0.4)

    model_rfc.fit(TRNtrain, TARtrain)
    model_etc.fit(TRNtrain, TARtrain)
    model_dtc.fit(TRNtrain, TARtrain)
    model_knc.fit(TRNtrain, TARtrain)
    model_svc.fit(TRNtrain, TARtrain)

    all_scores['RFC'] = accuracy_score(TARtest, model_rfc.predict(TRNtest))
    all_scores['ETC'] = accuracy_score(TARtest, model_etc.predict(TRNtest))
    all_scores['DTC'] = accuracy_score(TARtest, model_dtc.predict(TRNtest))
    all_scores['KNC'] = accuracy_score(TARtest, model_knc.predict(TRNtest))
    all_scores['SVC'] = accuracy_score(TARtest, model_svc.predict(TRNtest))

    f = plt.figure(figsize = (8, 6))
    p = DataFrame.from_dict(data = all_scores, orient='index').plot(kind='barh', legend=False, ax = f.gca())
    f.savefig('./test_plot/scores.png')

    # print 'RFC: ', accuracy_score(TARtest, model_rfc.predict(TRNtest))
    # print 'ETC: ', accuracy_score(TARtest, model_etc.predict(TRNtest))
    # print 'DTC: ', accuracy_score(TARtest, model_dtc.predict(TRNtest))
    # print 'KNC: ', accuracy_score(TARtest, model_knc.predict(TRNtest))
    # print 'SVC: ', accuracy_score(TARtest, model_svc.predict(TRNtest))

# финальная функция
def go():
    data = get_train_data()

    model_rfc = RandomForestClassifier(n_estimators = 1000, criterion = 'entropy', n_jobs = -1)
    # model_etc = ExtraTreesClassifier(n_estimators = 100, criterion = 'entropy')

    # так как лучшие результаты у Extra Trees и Random Forest
    print 'Go!!!'

    # print 'ETC...'
    # test, result = get_test_data()
    # target = data.Cover_Type
    # train = data.drop(['Cover_Type'], axis = 1)
    # model_rfc.fit(train, target)
    # result.insert(1,'Cover_Type', model_etc.predict(test))
    # result.to_csv('./test_etc.csv', index=False)

    print 'RFC...'
    test, result = get_test_data()
    test = test.drop(['Vertical_Distance_To_Hydrology'], axis = 1)
    target = data.Cover_Type
    train = data.drop(['Cover_Type', 'Vertical_Distance_To_Hydrology'], axis = 1)
    model_rfc.fit(train, target)
    result.insert(1,'Cover_Type', model_rfc.predict(test))
    result.to_csv('./test_rfc_4.csv', index=False)

def go_gbc():
    data = get_train_data()

    model_gbc = GradientBoostingClassifier(n_estimators = 1600)

    print 'Go!!!'

    print 'GBC...'
    test, result = get_test_data()
    target = data.Cover_Type
    train = data.drop(['Cover_Type'], axis = 1)
    model_gbc.fit(train, target)
    result.insert(1,'Cover_Type', model_gbc.predict(test))
    result.to_csv('./test_gbc_1600.csv', index=False)

def grid_search_test():
    data = get_train_data()
    target = data.Cover_Type
    train = data.drop(['Cover_Type'], axis = 1)

    model_rfc = RandomForestClassifier()
    params = {"n_estimators" : [100, 250, 500, 625], "criterion" : ('entropy', 'gini')}

    clf = grid_search.GridSearchCV(model_rfc, params)
    clf.fit(train, target)

    # summarize the results of the grid search
    print(clf.best_score_)
    print(clf.best_estimator_.criterion)
    print(clf.best_estimator_.n_estimators)

# plot_elevation()
# plot_aspect()
# plot_slope()
# plot_horizontal_distance_to_hydrology()
# plot_vertical_distance_to_hydrology()
# plot_horizontal_distance_to_roadways()
# plot_train()
# get_scores()
# cross_validation_test()
# grid_search_test()
go()
# go_gbc()
# plot_box()
