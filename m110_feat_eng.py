# Copyright 2018 Mamy André-Ratsimbazafy. All rights reserved.

from src.star_command import feat_engineering_pipe
from sklearn.preprocessing import Normalizer, OneHotEncoder, Imputer
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
from sklearn.decomposition import TruncatedSVD, NMF
from src.preprocessing.transformers_categorical import *

pipe_transforms = feat_engineering_pipe(
  tr_etat,
  tr_magasin,
  tr_categorie,
  tr_categorie1,   tr_categorie2,   tr_categorie3,   tr_categorie4,
  tr_couleur
)

select_feat = [
    ("encoded_etat", None),
    ("encoded_nom_magasin", None),
    ("prix", None),
    ("nb_images", None),
    ("longueur_image", None),
    ("largeur_image", None),
    ("poids", None),
    ("encoded_categorie", None),
    ("encoded_sous_categorie_1", None),
    ("encoded_sous_categorie_2", None),
    ("encoded_sous_categorie_3", None),
    ("encoded_sous_categorie_4", None),
    ("description_produit", [TfidfVectorizer(max_features=2**16,
                             min_df=2, stop_words='english',
                             use_idf=True),
                    TruncatedSVD(2)]),
    ("encoded_couleur", None),
    ("vintage", None)
]

