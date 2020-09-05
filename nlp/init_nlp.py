# Importar librerías necesarias para el PLN
import nltk

# Tokenización
#nltk.download('punkt')
from nltk import word_tokenize

# Etiquetado
from nltk.tag import StanfordPOSTagger

# Fragmentación
from nltk.chunk import *
from nltk.chunk.util import *
from nltk.chunk.regexp import *
from nltk import Tree, FreqDist

# Preprocesación
#nltk.download('stopwords')
from nltk.corpus import stopwords
from nltk.stem import SnowballStemmer
spanish_stops = stopwords.words('spanish')
spanish_stemmer = SnowballStemmer('spanish')

# Entrenamiento
import collections
from nltk.util import ngrams
import gensim 
from gensim.models import word2vec

# Visualización del vocabulario
from sklearn.manifold import TSNE
from sklearn.metrics.pairwise import cosine_similarity

# otras librerías
import zipfile
import itertools
import os
import pandas as pd
import numpy as np
import seaborn as sns 
import matplotlib.pyplot as plt





