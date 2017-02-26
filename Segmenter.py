"""
File: Segmenter.py
Authors: Michael Potter, Deepali Kamat
Description: A program to segment and classify math symbols
"""

import sys
import os
import codecs
import pickle
import time
import threading
import gc
import itertools
import copy
from multiprocessing import Pool, Process, Value, Lock
import numpy as np
from bs4 import BeautifulSoup
import featureextractor as extractor
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.externals import joblib

def usage():
	"""
	Prints a usage message
	"""
	print('Usage Options:\nSegmenter.py <file>.inkml\nSegmenter.py -Train <Training Data Directory>\nSegmenter.py -Dir <Test Data Directory>')

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

def getstrokes(infilename):
	"""
	Returns an array of [data,traceid] pairs
	"""
	infile = open(infilename, 'r')
	#Identify all of the symbols in the document
	try:
		soup = BeautifulSoup(infile, 'html.parser')
	except UnicodeDecodeError: #File Corruption
		#Attempt to load file by ignoring corrupted characters
		with codecs.open(infilename, "r", encoding='utf-8', errors='ignore') as fdata:
			soup = BeautifulSoup(fdata, 'html.parser')

	#Identify all strokes in the file
	traces = soup.find_all("trace")
	strokeidpairs = [[] for i in range(len(traces))]

	for i, trace in enumerate(traces):
		data = trace.contents
		data = ''.join(data)
		xypairs = [d.strip() for d in data.split(",")]
		data = np.zeros((len(xypairs), 2))
		for j, pair in enumerate(xypairs):
			data[j][0] = float(pair.split(" ")[0])
			data[j][1] = float(pair.split(" ")[1])
		strokeidpairs[i] = [data, trace['id']]

	soup.decompose() #Free memory
	infile.close()
	return strokeidpairs

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

def emitoutput(name, classlabels, componentlabels, cheat=False):
	"""
	Save a .lg file based on the segmentation results
	"""
	#Initialize the file and create the header
	if cheat == False:
		outfile = open("ForestResults/"+name+'.lg', 'w')
	else:
		outfile = open("ForestGroundTruth/"+name+'.lg', 'w')
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

def euclidean(x1, x2, y1, y2):
	"""
	Returns the euclidean distance between two points
	"""
	return np.sqrt((x1-x2)**2+(y1-y2)**2)



def segment(strokelist,classifier):
    array = []
    x1 = np.median(strokelist[0])
    y1 = np.median(strokelist[1])
    features = extractor.computefeatures([x[0] for x in strokelist])
    for j in strokelist:
        array.append(classifier.predict_proba([features])[0])

    bestscore = -float('inf')
    bestsymbol = 'UNKNOWN'
    for i, stroke in enumerate(strokelist):
        array[i] = [max(array[i-1])+ classifier.predict_proba(array[j]) , max(array[i-2])+ classifier.predict_proba(array[j, j-1])]
        if (max(stroke[i][0])+x1/8 < min(stroke[i][0] + x1/8)):
            array[i] = array[i]+ classifier.predict_proba([features])[0]
            score = score + max(array[i])
        if score > bestscore:
            bestscore = score
        bestsymbol = classifier.classes_[np.argmax(score)]
        symbolstrokelist.append(bestsymbol)
    return symbolstrokelist

def segandsave(filename, classifier, classifierlock):
	"""
	A function to segment and classify symbols
	"""

	#Determine the output filename
	name = filename.split('/')[-1].split('.')[0]

	#Attempt to create ground truth file for comparison
	featset = getfeatures(filename)
	if len(featset) != 0:
		#Create truth .lg values to compare against
		emitoutput(name, [c[1] for c in featset], [c[2] for c in featset], cheat=True)

	#Perform segmentation based on strokes in the file
	strokedata = getstrokes(filename) #[[pointlist, name], ...]

	### NN method ###

	#Force single point strokes to merge with others if they are nearby
	for i, stroke in extractor.reverseenumerate(strokedata):
		#If you only have 1 point
		if len(stroke[0]) == 1:
			#See if it's identical to the point in the stroke immediately before
			if i > 0:
				if stroke[0][0][0] == strokedata[i-1][0][-1][0] and stroke[0][0][1] == strokedata[i-1][0][-1][1]:
					#stroke i should be merged with i-1
					strokedata[i-1][1] += ', '+stroke[1]
					del strokedata[i]
					continue
			#See if it's identical to the point in the stroke immediately after
			if i < len(strokedata)-1:
				if stroke[0][0][0] == strokedata[i+1][0][0][0] and stroke[0][0][1] == strokedata[i+1][0][0][1]:
					#stroke i should be merged with i+1
					strokedata[i+1][1] += ', '+stroke[1]
					del strokedata[i]
					continue

	#Compute the median width of all strokes
	widths = np.zeros(len(strokedata))
	for i, stroke in enumerate(strokedata):
		points = stroke[0]
		widths[i] = max([point[0] for point in points]) - min([point[0] for point in points])
	medianwidth = np.median(widths)

	#Compute the mean x and y coordinates for each stroke. [x, y, stroke]
	for i, stroke in enumerate(strokedata):
		strokedata[i] = [np.mean([x[0] for x in stroke[0]]), np.mean([y[1] for y in stroke[0]]), stroke]

	#Sort stroke list by x, then by y so that the y sort takes precedent
	strokedata.sort(key=lambda x: x[0])
	strokedata.sort(key=lambda y: y[1])

	#Compute the nearest neighbors (might need to ignore some neighbors once consumed)
	#neigbors[i] = [[neighbor index, distance from i], ...]
	neighbors = [[[a, euclidean(strokedata[a][0], strokedata[b][0], strokedata[a][1], strokedata[b][1])]
		for a in range(len(strokedata))] for b in range(len(strokedata))]

	#Sort each point to get it's nearest neighbors first in list
	for point in neighbors:
		point.sort(key=lambda x: x[1])

	#Record strokes which have already been claimed for a symbol
	alreadyclaimed = [False for x in range(len(strokedata))]

	#Record [[symbol, strokenames], ...] pairs
	symbolstrokelist = []

	#Traverse from top left to bottom right (y axis major to deal with stacking)
	for i, stroke in enumerate(strokedata):
		#Don't go any further if stroke has already been assigned to a symbol
		if alreadyclaimed[i]:
			continue
		alreadyclaimed[i] = True

		nnlist = neighbors[i]
		#Go through all neighbors to find remaining closest 3 (excluding itself)
		nnstrokeids = []
		for j in range(len(nnlist)):
			if not alreadyclaimed[nnlist[j][0]]:
				nnstrokeids.append(nnlist[j][0])
				#Only consider up to 3 nearest neighbors
				if len(nnstrokeids) == 3:
					break

		#Delete neighbors which have an x gap
		xmaxi = max([x[0] for x in stroke[2][0]]) #Max x-coordinate of stroke i
		xmini = min([x[0] for x in stroke[2][0]]) #Min x-coordinate of stroke i
		# print("Xmaxi: {}, i: {}, id: {}".format(xmaxi, i, stroke[2][1]))
		for j, elem in extractor.reverseenumerate(nnstrokeids):
			xmaxj = max([x[0] for x in strokedata[nnstrokeids[j]][2][0]])
			xminj = min([x[0] for x in strokedata[nnstrokeids[j]][2][0]])
			# print("Xminj: {}, j: {}, id: {}".format(xminj, nnstrokeids[j], strokedata[nnstrokeids[j]][2][1]))
			if xmini <= xminj:
				if xmaxi + medianwidth/8.0 < xminj - medianwidth/8.0:
					del nnstrokeids[j]
			else:
				if xmaxj + medianwidth/8.0 < xmini - medianwidth/8.0:
					del nnstrokeids[j]

		#Build the desired test combinations
		testset = [[i]]
		for l in range(1, len(nnstrokeids)+1):
			for subset in itertools.combinations(nnstrokeids, l):
				#stroke i must be in every combination
				subsetlist = list(subset)
				subsetlist.extend([i])
				testset.append(subsetlist)

		bestscore = -float('inf')
		bestsymbol = 'UNKNOWN'
		bestsetid = 0

		for t, test in enumerate(testset):
			#Generate features based on the strokes corresponding to the proposed test list
			features = extractor.computefeatures(copy.deepcopy([x[0] for x in [strokedata[s][2] for s in test]]))
			scores = 0
			# with classifierlock:
			scores = classifier.predict_proba([features])[0]

			#TODO Multiply score by prior probability, composition probability
			# scores = np.multiply(scores,classpriors)
			# scores = np.multiply(scores,lengthpriors)
			# print(test)
			# print([x[1] for x in [strokedata[s][2] for s in test]])
			# print(scores)
			#Give a slight advantage to more complex characters
			scores += (len(test)/4.0) * (1.0/6.0)
			# print(scores)
			score = max(scores)
			if score > bestscore:
				bestscore = score
				bestsymbol = classifier.classes_[np.argmax(scores)]
				bestsetid = t

		#Mark all desired strokes as claimed
		for strokeid in testset[bestsetid]:
			alreadyclaimed[strokeid] = True

		#Record the classification and segmentation decision
		symbolstrokelist.append([bestsymbol, [x[1] for x in [strokedata[s][2] for s in testset[bestsetid]]]])

	# ### In-order method

	# symbolstrokelist = []
	# start = 0
	# while start < len(strokedata):

	# 	bestscore = -float('inf')
	# 	bestend = start
	# 	bestsymbol = 'UNKNOWN'

	# 	#Limit the max length of a symbol to 10 strokes
	# 	for end in range(start+1, min(start+12, len(strokedata)+1)):

	# 		features = extractor.computefeatures([x[0] for x in strokedata[start:end]])

	# 		scores = classifier.predict_proba([features])[0]
	# 		#TODO Multiply score by prior probability, composition probability
	# 		# scores = np.multiply(scores,classpriors)
	# 		# scores = np.multiply(scores,lengthpriors)
	# 		score = max(scores)
	# 		if score > bestscore:
	# 			bestscore = score
	# 			bestsymbol = classifier.classes_[np.argmax(scores)]
	# 			# bestsymbol = classifier.predict([features])[0]
	# 			bestend = end

	# 	symbolstrokelist.append([bestsymbol, [x[1] for x in strokedata[start:bestend]]])
	# 	start = bestend

	#Save an output file
	emitoutput(name, [s[0] for s in symbolstrokelist], [s[1] for s in symbolstrokelist])

def segandsavelauncher(filearray, classifier, classifierlock, progresstracker, lock):
	"""Used to report progress over multiple files"""
	# classifier = joblib.load(classifierpath)
	for filename in filearray:
		segandsave(filename, classifier, classifierlock)
		with lock:
			progresstracker.value += 1
		gc.collect()

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

		classifier = ExtraTreesClassifier(
			n_estimators=200, #Number of trees in forest
			criterion="gini", #Split based on information gain (entropy or gini)
			max_features=1.0, #How many features considered at each split
			max_depth=None, #Cut trees off at depth x (5)
			min_samples_split=2, #Min samples to split an internal node
			random_state=0, #Random number generator seed
			n_jobs=-1, #Use all available cores
			)

		classifier.fit(attributes, classes)

		"""classlist = classifier.classes_
		classindex = {}
		for i, c in enumerate(classlist):
			classindex[c] = i
		classpriors = np.zeros((1, len(classlist)))
		for instance in classes:
			classpriors[classindex[instance]] += 1.0/len(classes)
		classpriorsfile = open('DTreePriors', 'wb')
		pickle.dump(classpriors, classpriorsfile)
		classpriorsfile.close()"""

		print("Training Accuracy: {}".format(classifier.score(attributes, classes)))

		#Set up save directory
		if not os.path.exists("Forest"):
			os.makedirs("Forest")
		joblib.dump(classifier, 'Forest/trainedclassifier.pkl')

		return

	#Otherwise the user should already have trained forest file
	if not os.path.isdir('Forest'):
		print("Error: Trained Model not found. Run with -Train flag to create one.")
		usage()
		return
	if not os.path.isfile('Forest/trainedclassifier.pkl'):
		print("Error: Trained Model not found. Run with -Train flag to create one.")
		usage()
		return

	#Set up output folders to save into
	if not os.path.exists("ForestResults"):
		os.makedirs("ForestResults")
	if not os.path.exists("ForestGroundTruth"):
		os.makedirs("ForestGroundTruth")

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


	#### MULTITHREADED ####
	threadcount = 3
	#Instantiate classifier copies
	classifiers = [0 for x in range(0, threadcount)]
	# classifiers[0] = joblib.load('Forest/trainedclassifier.pkl')
	for i in range(0, threadcount):
		classifiers[i] = joblib.load('Forest/trainedclassifier.pkl')

	# classifier = joblib.load('Forest/trainedclassifier.pkl')
	classifierlock = Lock()

	#Instantiate threads
	threads = []
	# progresstracker = Counter()
	progresstracker = Value('i', 0)
	lock = Lock()
	sys.stdout.write("\rSegmenting: 0.0%")
	for i in range(threadcount):
		if i < len(filenames):
			t = Process(target=segandsavelauncher, args=(filenames[i::threadcount], classifiers[i], classifierlock, progresstracker, lock))
			threads.append(t)
			t.start()

	while True:
		with lock:
			sys.stdout.write("\rSegmenting: {0:0.1f}%".format(100*(progresstracker.value)/len(filenames)))
			if progresstracker.value == len(filenames):
				break
		time.sleep(3)

	sys.stdout.write("\rSegmenting: 100.0%")
	sys.stdout.write("\n")
	for thread in threads:
		thread.join()

	# ### SERIAL ####
	# classifier = joblib.load('Forest/trainedclassifier.pkl')

	# sys.stdout.write("\rSegmenting: 0.0%")
	# for i, name in enumerate(filenames):
	# 	segandsave(name, classifier)
	# 	sys.stdout.write("\rSegmenting: {0:0.1f}%".format(100*(i+1)/len(filenames)))
	# sys.stdout.write("\n")



if __name__ == '__main__':
	main(sys.argv[1:])
