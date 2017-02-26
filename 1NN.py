"""
File: 1NN.py
Authors: Michael Potter, Deepali Kamat
Description: A program to classify symbols based on the 1NN method
"""

import sys
import os
import codecs
import pickle
from multiprocessing import Pool
import numpy as np
from bs4 import BeautifulSoup
import featureextractor as extractor

def usage():
	"""
	Prints a usage message?
	"""
	print('Usage Options:\n1NN.py <file>.inkml\n1NN.py -Train <Training Data Directory>\n1NN.py -Dir <Test Data Directory>')

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
		print("Bad File: {}".format(infilename))
		#Attempt to load file by ignoring corrupted characters
		with codecs.open(infilename, "r", encoding='utf-8', errors='ignore') as fdata:
			soup = BeautifulSoup(fdata, 'html.parser')

	#Determine all tracegroups (removing the first since it is a group of groups)
	tracegroups = soup.find_all("tracegroup")
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
		### REMOVE CATCH ONCE DEBUG DONE ###
		try:
			features = extractor.computefeatures(tracedata)
		except:
			print(infilename)
			features = []
		else:
			pass
		finally:
			pass
			# features = []

		#Determine the true symbol
		symbol = '\\unknown'
		if group.find("annotation") is not None:
			symbol = ''.join((group.find("annotation")).contents)

		featpairs.append([features, symbol, traceids])

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
	
	## Serial Loading ##

	# #Create a corpus of feature/class pairs to compare against
	# sys.stdout.write("\rLoading: 0.0%")
	# featclassdata = []
	# for i,f in enumerate(files):
	# 	featclassdata.extend(getfeatures(f))
	# 	sys.stdout.write("\rLoading: {0:0.1f}%".format(100*(i+1)/len(files)))
	# sys.stdout.write("\n")

	## Parallel Loading ##

	sys.stdout.write("\rLoading: 0.0%")
	pool = Pool(8)
	featclassdata = []
	for i, result in enumerate(pool.imap_unordered(getfeatures, (f for f in files), chunksize=10)):
		sys.stdout.write("\rLoading: {0:0.1f}%".format(100*(i+1)/len(files)))
		featclassdata.extend(result)
	sys.stdout.write("\n")

	return featclassdata

def loadfeat(featfolder):
	"""
	Loads all .features files in a folder and returns
	a list of feature/symbol pairings. Assumes that classes are labeled. 
	"""
	#Find all .inkml files from the subdirectories of the input folder
	files = [os.path.join(root, name) 
		for root, dirs, files in os.walk(featfolder)
		for name in files
		if name.endswith((".features"))]
	
	#Create a corpus of feature/class pairs to compare against
	featclassdata = []

	#Create a feature file for each input file 
	for f in files:
		infile = open(f)
		#Identify all of the symbols in the document
		try:
			soup = BeautifulSoup(infile, 'html.parser')
		except UnicodeDecodeError: #File Corruption
			print("Bad File: {}".format(f))
			#Attempt to load file by ignoring corrupted characters
			with codecs.open(f, "r", encoding='utf-8', errors='ignore') as fdata:
				soup = BeautifulSoup(fdata, 'html.parser')

		#Determine all symbols
		symbols = soup.find_all("symbol")

		#For each symbol...
		for symbol in symbols:
			#Read the features
			features = np.fromstring(''.join((symbol.find("features")).contents), sep=',').tolist()

			#Determine the true type
			symbol = ''.join((symbol.find("character")).contents)

			featclassdata.append([features, symbol])

	return featclassdata

def distance(features1, features2):
	dis = 0
	for i in range(len(features1)):
		dis += pow((features1[i] - features2[i]), 2)
	return dis

def distancelauncher(args):
	"""Used since multiprocessing map takes single-argument functions"""
	return distance(*args)

def classify(instance, corpus):
	"""
	Classify a given instance based on 1NN against a corpus of data
	"""
	nnlist = [[features[0], features[1], float("inf")] for features in corpus]
	for i, point in enumerate(corpus):
		nnlist[i][2] = distance(instance, point[0])
	lowestdistance = min([d[2] for d in nnlist])
	for elem in nnlist:
		if elem[2] == lowestdistance:
			return elem[1]
	return '\\unknown'

def emitoutput(name, classlabels, componentlabels, cheat=False):
	#Initialize the file and create the header
	if cheat == False:
		outfile = open("1NNResults/"+name+'.lg', 'w')
	else:
		outfile = open("GroundTruth/"+name+'.lg', 'w')
	outfile.write("# IUD, {}\n# [ OBJECTS ]\n".format(name))
	nodecount = 0
	for component in componentlabels:
		for stroke in component:
			nodecount += 1
	outfile.write("# Primitive Nodes (N): {}\n".format(nodecount))
	outfile.write("#    Objects (O): {}\n".format(len(componentlabels)))
	#Write out the object labels
	for i, label in enumerate(classlabels):
		componentstring = ''
		for stroke in componentlabels[i]:
			componentstring = componentstring + stroke + ', '
		if label == ',':
			label = 'COMMA'
		outfile.write("O, char_{}, {}, 1.0, {}\n".format(i, label, componentstring[0:-2]))
	outfile.write('\n')
	#Write out the summary text
	outfile.write("# [ SUMMARY ]\n")
	outfile.write("# Primitive Nodes (N): {}\n".format(nodecount))
	outfile.write("#    Objects (O): {}\n".format(len(componentlabels)))

	outfile.close()

def classandsave(filename, corpus, normalizationdata):
	#Load the input file symbols into feature arrays
	featset = getfeatures(filename)
	#Normalize the feature data
	for sym in range(len(featset)):
		for feat in range(len(featset[0][0])):
			if normalizationdata[feat] != 0:
				featset[sym][0][feat] /= normalizationdata[feat]

	#Determine the output filename
	name = filename.split('/')[-1].split('.')[0]
	#Create truth .lg values to compare against
	emitoutput(name, [c[1] for c in featset], [c[2] for c in featset], cheat=True)
	
	#Envoke the 1NN classifier
	for i, symbol in enumerate(featset):
		#Assign class based on 1NN
		featset[i][1] = classify(symbol[0], corpus)
	#Save an output file
	emitoutput(name, [c[1] for c in featset], [c[2] for c in featset])

def classandsavelauncher(args):
	"""Used since multiprocessing map takes single-argument functions"""
	classandsave(*args)

def main(argv):
	"""
	Takes input .inkml files and generates classification files
	"""
	if not argv:
		usage()
		return
	#If the user wants to load data to set up a 1NN classifier...
	if argv[0] == "-Train":
		if not os.path.isdir(argv[1]):
			print("Error: Must supply a directory with -Train flag")
			return
		featclassdata = loadink(argv[1])
		
		#Compute the maximum value for each feature
		features = [feat[0] for feat in featclassdata]
		maxfeatures = [max(features, key=lambda e: e[i])[i] for i in range(len(features[0]))]

		#Normalize each feature by its maximum value
		for i in range(len(featclassdata)):
			for j in range(len(features[0])):
				if maxfeatures[j] != 0:
					featclassdata[i][0][j] /= maxfeatures[j]

		#Save the normalized feature data
		corpusfile = open('1NNCorpus', 'wb')
		pickle.dump(featclassdata, corpusfile)

		#Save the normalization values
		normfile = open('NormalizationValues', 'wb')
		pickle.dump(maxfeatures, normfile)
		return

	#Otherwise the user should already have 1NNCorpus and normalization files
	if not os.path.isfile('1NNCorpus'):
		print("Error: Data Corpus not found. Run with -Train flag to create one.")
		usage()
		return
	corpusfile = open('1NNCorpus', 'rb')
	featclassdata = pickle.load(corpusfile)
	if not os.path.isfile('NormalizationValues'):
		print("Error: Normalization file not found. Run with -Train flag to create one.")
		usage()
		return
	normalizationfile = open('NormalizationValues', 'rb')
	normalizationdata = pickle.load(normalizationfile)

	#Set up output folders to save into
	if not os.path.exists("1NNResults"):
		os.makedirs("1NNResults")
	if not os.path.exists("GroundTruth"):
		os.makedirs("GroundTruth")
	
	#See if user wants to run on a directory of files or on a single file
	filenames = []
	if argv[0] == "-Dir":
		if not os.path.isdir(argv[1]):
			print("Error: Must supply a directory with -Dir flag")
			return
		#Find all .inkml files from the subdirectories of the input folder
		filenames = [os.path.join(root, name) 
			for root, dirs, files in os.walk(argv[1])
			for name in files
			if name.endswith((".inkml"))]
	#Otherwise ensure appropriate file type is requested
	else:
		#Ensure that requested file exists
		if not os.path.isfile(argv[0]):
			print("Error: Input must be a file.")
			usage()
			return
		if argv[0].split('.')[1] != 'inkml':
			print("Error: Must supply .inkml file")
			usage()
			return
		filenames.append(argv[0])

	pool = Pool(8)
	pool.map(classandsavelauncher, ((f, featclassdata, normalizationdata) for f in filenames))

if __name__ == '__main__':
	main(sys.argv[1:])
