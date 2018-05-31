from sklearn.preprocessing import LabelEncoder
import pandas as pd
import string
import numpy as np


# DEPRECATED Apply Label encoder
def _encode_categoricals(train,test, sColumn):
    le = LabelEncoder()
    le.fit(list(train[sColumn].fillna('NaN').apply(str.lower).values) + list(test[sColumn].fillna('NaN').apply(str.lower).values))

    def _trans(df, sColumn, le):
        encoded = le.transform(df[sColumn].fillna('NaN').apply(str.lower))
        df['encoded_' + sColumn] = encoded
        return df
    return _trans(train, sColumn, le),_trans(test, sColumn, le)

def tr_etat(train, test, y, folds, cache_file):
    trn, tst = _encode_categoricals(train, test,"etat")
    return trn, tst, y, folds, cache_file

def tr_magasin(train, test, y, folds, cache_file):
    trn, tst = _encode_categoricals(train, test,"nom_magasin")
    return trn, tst, y, folds, cache_file

def tr_categorie(train, test, y, folds, cache_file):
    trn, tst = _encode_categoricals(train, test,"categorie")
    return trn, tst, y, folds, cache_file

def tr_categorie1(train, test, y, folds, cache_file):
    trn, tst = _encode_categoricals(train, test,"sous_categorie_1")
    return trn, tst, y, folds, cache_file

def tr_categorie2(train, test, y, folds, cache_file):
    trn, tst = _encode_categoricals(train, test,"sous_categorie_2")
    return trn, tst, y, folds, cache_file

def tr_categorie3(train, test, y, folds, cache_file):
    trn, tst = _encode_categoricals(train, test,"sous_categorie_3")
    return trn, tst, y, folds, cache_file

def tr_categorie4(train, test, y, folds, cache_file):
    trn, tst = _encode_categoricals(train, test,"sous_categorie_4")
    return trn, tst, y, folds, cache_file

def tr_couleur(train, test, y, folds, cache_file):
    trn, tst = _encode_categoricals(train, test,"couleur")
    return trn, tst, y, folds, cache_file

