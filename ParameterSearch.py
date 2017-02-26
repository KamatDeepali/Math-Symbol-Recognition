import os
import sys
import codecs

import numpy as np
import pickle

from time import time
from operator import itemgetter
from scipy.stats import randint as sp_randint
from multiprocessing import Pool
from bs4 import BeautifulSoup

import featureextractor as extractor

from sklearn.grid_search import GridSearchCV, RandomizedSearchCV
from sklearn.datasets import load_digits
from sklearn.ensemble import RandomForestClassifier



def usage():
  print("-Train <TrainingDataDir>")

def getfeatures(infilename):
  """
  Returns an array of [feature vector, class label, trace ids].
  1 per every character in the file
  """
  infile = open(infilename, 'r')
  #Identify all of the symbols in the document
  try:
    soup = BeautifulSoup(infile, 'html.parser')
  except UnicodeDecodeError: #File Corruption
    # print("Bad File: {}".format(infilename))
    #Attempt to load file by ignoring corrupted characters
    with codecs.open(infilename, "r", encoding='utf-8', errors='ignore') as fdata:
      soup = BeautifulSoup(fdata, 'html.parser')

  #Determine all tracegroups (removing the first since it is a group of groups)
  tracegroups = soup.find_all("tracegroup")
  #Abort if tracegroup data not available (segmentation test file)
  if len(tracegroups) == 0:
    soup.decompose()
    infile.close()
    return []
  tracegroups = tracegroups[1:]

  featpairs = []

  #Identify all traces within the group
  for group in tracegroups:
    traceviews = group.find_all("traceview")
    tracedata = []
    traceids = []
    for trace in traceviews:
      data = soup.find("trace", id=trace['tracedataref'])
      data = data.contents
      data = ''.join(data)
      xypairs = [d.strip() for d in data.split(",")]
      data = np.zeros((len(xypairs), 2))
      for i, pair in enumerate(xypairs):
        data[i][0] = float(pair.split(" ")[0])
        data[i][1] = float(pair.split(" ")[1])
      tracedata.append(data)
      traceids.append(trace['tracedataref'])

    #Compute the features based on the traces
    features = extractor.computefeatures(tracedata)

    #Determine the true symbol
    symbol = '\\unknown'
    if group.find("annotation") is not None:
      symbol = ''.join((group.find("annotation")).contents)

    featpairs.append([features, symbol, traceids])

  soup.decompose() #Free memory
  infile.close()
  return featpairs

def loadink(inkfolder):
  """
  Loads all .inkml files in a folder, performs feature extraction, and returns
  a list of feature/symbol pairings. Assumes that classes are labeled.
  """
  #Find all .inkml files from the subdirectories of the input folder
  files = [os.path.join(root, name)
    for root, dirs, files in os.walk(inkfolder)
    for name in files
    if name.endswith((".inkml"))]

  ## Parallel Loading ##

  sys.stdout.write("\rLoading: 0.0%")
  pool = Pool(8)
  featclassdata = []
  for i, result in enumerate(pool.imap_unordered(getfeatures, (f for f in files), chunksize=10)):
    sys.stdout.write("\rLoading: {0:0.1f}%".format(100*(i+1)/len(files)))
    featclassdata.extend(result)
  sys.stdout.write("\n")

  return featclassdata

# Utility function to report best scores
def report(grid_scores, n_top=3):
    top_scores = sorted(grid_scores, key=itemgetter(1), reverse=True)[:n_top]
    for i, score in enumerate(top_scores):
        print("Model with rank: {0}".format(i + 1))
        print("Mean validation score: {0:.3f} (std: {1:.3f})".format(
              score.mean_validation_score,
              np.std(score.cv_validation_scores)))
        print("Parameters: {0}".format(score.parameters))
        print("")

def main(argv):
  #get data
  if not argv:
    usage()
    return
  #If the user wants to load data to set up a 1NN classifier...
  if argv[0] == "-Train":
    if not os.path.isdir(argv[1]):
      print("Error: Must supply a directory with -Train flag")
      return

    #Load training data
    if not os.path.isfile('DTreeCorpus'):
      featclassdata = loadink(argv[1])
      corpusfile = open('DTreeCorpus', 'wb')
      pickle.dump(featclassdata, corpusfile)
      corpusfile.close()
    else:
      corpusfile = open('DTreeCorpus', 'rb')
      featclassdata = pickle.load(corpusfile)
      corpusfile.close()

    #Train the model
    attributes = [f[0] for f in featclassdata]
    classes = [f[1] for f in featclassdata]

    # build a classifier
    clf = RandomForestClassifier(n_estimators=100)

    # specify parameters and distributions to sample from
    param_dist = {"max_depth": [3, None],
                  "max_features": sp_randint(1, 11),
                  "min_samples_split": sp_randint(1, 11),
                  "min_samples_leaf": sp_randint(1, 11),
                  "bootstrap": [True, False],
                  "criterion": ["gini", "entropy"]}

    # run randomized search
    n_iter_search = 20
    random_search = RandomizedSearchCV(clf, param_distributions=param_dist,
                                       n_iter=n_iter_search)

    start = time()
    random_search.fit(attributes, classes)
    print("RandomizedSearchCV took %.2f seconds for %d candidates"
          " parameter settings." % ((time() - start), n_iter_search))
    report(random_search.grid_scores_)

if __name__ == '__main__':
  main(sys.argv[1:])