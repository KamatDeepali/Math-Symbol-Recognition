"""
File: Parser.py
Authors: Michael Potter, Deepali Kamat
Description: A program to parse, segment and classify math symbols
"""

import sys
import os
import codecs
import pickle
import time
import gc
import itertools
import copy
import subprocess
from multiprocessing import Pool, Process, Value, Lock
import numpy as np
from bs4 import BeautifulSoup
import featureextractor as extractor
import ParsingFeatureExtractor as parsingextractor
from sklearn.ensemble import RandomForestClassifier
from sklearn.externals import joblib

from sklearn.grid_search import GridSearchCV, RandomizedSearchCV
from operator import itemgetter
from scipy.stats import randint as sp_randint
from scipy.stats import uniform

def usage():
	"""
	Prints a usage message
	"""
	print('Usage Options:\nSegmenter.py <file>.inkml\nSegmenter.py -Train <Training Data Directory>\nSegmenter.py -Dir <Test Data Directory>')

def report(grid_scores, n_top=3):
	"""
	Utility function to report best scores search scores
	"""
	top_scores = sorted(grid_scores, key=itemgetter(1), reverse=True)[:n_top]
	for i, score in enumerate(top_scores):
		print("Model with rank: {0}".format(i + 1))
		print("Mean validation score: {0:.3f} (std: {1:.3f})".format(
			score.mean_validation_score,
			np.std(score.cv_validation_scores)))
		print("Parameters: {0}".format(score.parameters))
		print("")

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

def getrelationships(infilename):
	"""
	Returns an array of [feature vector, relationship label]
	"""
	#Read in the inkml file
	infile = open(infilename, 'r')
	#Identify all of the symbols in the document
	try:
		soup = BeautifulSoup(infile, 'html.parser')
	except UnicodeDecodeError: #File Corruption
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

	#Find all object relationships
	featpairs = []
	pipe = subprocess.run(["crohme2lg", infilename], universal_newlines=True, stdout=subprocess.PIPE)
	lgtext = pipe.stdout
	lgtext = lgtext.split('\n')
	for line in lgtext:
		line = line.split(',')
		if line[0] == "R":
			symbolstrokedata = [0, 0]

			for i in [1, 2]:
				#Determine the name of the involved symbol
				symbol = line[i].strip()

				#Find the corresponding symbols in the file
				tracegroup = soup.find("annotationxml", href=symbol)
				if tracegroup is not None:
					tracegroup = tracegroup.parent()
				else:
					break

				#Get the x,y data for the symbol
				traceviews = []
				for elem in tracegroup:
					if elem.name == "traceview":
						traceviews.append(elem)

				# traceviews = tracegroup.find_all("traceview")
				tracedata = []
				for trace in traceviews:
					data = soup.find("trace", id=trace['tracedataref'])
					data = data.contents
					data = ''.join(data)
					xypairs = [d.strip() for d in data.split(",")]
					data = np.zeros((len(xypairs), 2))
					for j, pair in enumerate(xypairs):
						data[j][0] = float(pair.split(" ")[0])
						data[j][1] = float(pair.split(" ")[1])
					tracedata.append(data)

				#Record the data
				symbolstrokedata[i-1] = tracedata

			#Ensure that read didn't fail
			if symbolstrokedata[0] == 0 or symbolstrokedata[1] == 0:
				continue

			#Compute features and append to list
			features = parsingextractor.computefeatures(symbolstrokedata[0], symbolstrokedata[1])
			featpairs.append([features, line[3].strip()])

	soup.decompose()
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

def loadink(inkfolder, function=getfeatures):
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
	for i, result in enumerate(pool.imap_unordered(function, (f for f in files), chunksize=10)):
		sys.stdout.write("\rLoading: {0:0.1f}%".format(100*(i+1)/len(files)))
		featclassdata.extend(result)
	sys.stdout.write("\n")

	return featclassdata

def emitoutput(name, classlabels, strokegroups, relationships, cheat=False):
	"""
	Save an .lg file based on the parsing results
	Inputs:
		name - the file basename to generate
		classlabels - a dictionary mapping stroke group ids to their class identities
		strokegroups - a dictionary mapping stroke group ids to a list of their components
		relationships - a list of [strokegroupid1, strokegroupid2, relationLabel]
	"""
	#Initialize the file and create the header
	if cheat == False:
		outfile = open("Predictions/"+name+'.lg', 'w')
	else:
		outfile = open("GroundTruth/"+name+'.lg', 'w')
	outfile.write("# IUD, {}\n# [ OBJECTS ]\n".format(name))
	nodecount = 0
	for key in strokegroups:
		for stroke in strokegroups[key]:
			nodecount += 1
	outfile.write("# Primitive Nodes (N): {}\n".format(nodecount))
	outfile.write("#	Objects (O): {}\n".format(len(strokegroups)))

	mergecount = 0

	#Write out the object labels
	for strokeid in strokegroups:
		componentstring = ''
		mergecount += len(strokegroups[strokeid]) * (len(strokegroups[strokeid]) - 1)
		for stroke in strokegroups[strokeid]:
			componentstring = componentstring + stroke + ', '
		label = classlabels[strokeid]
		if label == ',':
			label = 'COMMA'
		outfile.write("O, char_{}, {}, 1.0, {}\n".format(strokeid, label, componentstring[0:-2]))
	outfile.write('\n')

	#Count relationship edges
	relationcount = 0
	for r in relationships:
		relationcount += len(strokegroups[r[0]]) * len(strokegroups[r[1]])

	#Write out the relationship labels
	outfile.write("# [ RELATIONSHIPS ]\n")
	outfile.write("# Primitive Edges (E): {} ({} merge, {} relationship)\n".format(mergecount + relationcount, mergecount, relationcount))
	outfile.write("#	Object Relationships (R): {}\n".format(len(relationships)))
	for r in relationships:
		outfile.write("R, char_{}, char_{}, {}, 1.0\n".format(r[0], r[1], r[2]))
	outfile.write("\n")

	#Write out the summary text
	outfile.write("# [ SUMMARY ]\n")
	outfile.write("# Primitive Nodes (N): {}\n".format(nodecount))
	outfile.write("#	Objects (O): {}\n".format(len(strokegroups)))
	outfile.write("# Primitive Edges (E): {} ({} merge, {} relationship)\n".format(mergecount + relationcount, mergecount, relationcount))
	outfile.write("#	Object Relationships (R): {}\n".format(len(relationships)))
	outfile.close()

# def parseandsave(name, strokedata, classlabels, strokegroups, parser):
#	 """
#	 A function to parse relationships between segmented symbols
#	 Inputs:
#		 name - the file base name to save
#		 strokedata - [[pointlist, name], ...]
#		 classlabels - {symbolID:class}
#		 strokegroups - {symbolID:[name1,name2,...]}
#		 parser - a classifier trained on relationships
#	 """

#	 #Order the symbols from left to right by extreme value
#	 traversalorder = [[ID, 0, []] for ID in strokegroups] #[symbolID, minX, strokedataindices]
#	 for i, symbol in enumerate(traversalorder):
#		 minminX = float('inf')
#		 for j, stroke in enumerate(strokedata):
#			 if stroke[1] in strokegroups[symbol[0]]:
#				 traversalorder[i][2].append(j)
#				 minX = min(x[0] for x in stroke[0])
#				 if minX < minminX:
#					 minminX = minX
#		 traversalorder[i][1] = minminX
#	 traversalorder.sort(key=lambda x: x[1])

#	 #Greedily apply parser, recursively test substructure
#	 relationships = []
#	 for i, elem in enumerate(traversalorder):
#		 if i < len(strokegroups)-1:
#			 features = parsingextractor.computefeatures(copy.deepcopy([x[0] for x in [strokedata[s] for s in elem[2]]]), copy.deepcopy([x[0] for x in [strokedata[s] for s in traversalorder[i+1][2]]]))
#			 scores = parser.predict_proba([features])[0]
#			 # score = max(scores)
#			 relationships.append([elem[0], traversalorder[i+1][0], parser.classes_[np.argmax(scores)]])

#	 #Save an output file
#	 emitoutput(name, classlabels, strokegroups, relationships)

def parseandsave(name, strokedata, classlabels, strokegroups, parser):
	"""
	A function to parse relationships between segmented symbols
	Inputs:
		name - the file base name to save
		strokedata - [[pointlist, name], ...]
		classlabels - {symbolID:class}
		strokegroups - {symbolID:[name1,name2,...]}
		parser - a classifier trained on relationships
	"""
	#Compute the median width of all strokes
	widths = np.zeros(len(strokedata))
	for i, stroke in enumerate(strokedata):
		points = stroke[0]
		widths[i] = max([point[0] for point in points]) - min([point[0] for point in points])
	medianwidth = np.median(widths)

	#Order the symbols from left to right by extreme value
	traversalorder = [[ID, 0, []] for ID in strokegroups] #[symbolID, minX, strokedataindices]
	for i, symbol in enumerate(traversalorder):
		minminX = float('inf')
		for j, stroke in enumerate(strokedata):
			if stroke[1] in strokegroups[symbol[0]]:
				traversalorder[i][2].append(j)
				minX = min(x[0] for x in stroke[0])
				if minX < minminX:
					minminX = minX
		traversalorder[i][1] = minminX
	traversalorder.sort(key=lambda x: x[1])

	relationships = []
	relationships = parserecursive(strokedata, classlabels, strokegroups, parser, medianwidth, traversalorder, relationships)
	#Save an output file
	emitoutput(name, classlabels, strokegroups, relationships)

def parserecursive(strokedata, classlabels, strokegroups, parser, medianwidth, traversalorder, relationships):
	"""
	A recursive function to identify the next baseline character and generate relationships based on it
	"""
	if len(traversalorder) == 1:
		return relationships

	candidates = [] #[ [traversalelem, score], ...]
	#Find three candidate symbols which have a right relationship (or get to end of file
	for elem in traversalorder[1:]:
		#Compare the initial element to the next element
		features = parsingextractor.computefeatures(copy.deepcopy([x[0] for x in [strokedata[s] for s in traversalorder[0][2]]]),
                                                    copy.deepcopy([x[0] for x in [strokedata[s] for s in elem[2]]]))
		#Check the evaluated class
		scores = parser.predict_proba([features])[0]
		relation = parser.classes_[np.argmax(scores)]=
		if relation == 'Right':
			#Reject the candidate if it can be seperated from the first candidate using a vertical line
			if len(candidates) > 1:
				xmaxi = max([x[0] for x in [item for sublist in [strokedata[y][0] for y in candidates[0][0][2]] for item in sublist]])
				xmini = min([x[0] for x in [item for sublist in [strokedata[y][0] for y in candidates[0][0][2]] for item in sublist]])
				xmaxj = max([x[0] for x in [item for sublist in [strokedata[y][0] for y in elem[2]] for item in sublist]])
				xminj = min([x[0] for x in [item for sublist in [strokedata[y][0] for y in elem[2]] for item in sublist]])
				if xmini <= xminj:
					if xmaxi + medianwidth/8.0 < xminj - medianwidth/8.0:
						break #Any points coming after must also have a gap due to sort order
					else:
						if xmaxj + medianwidth/8.0 < xmini - medianwidth/8.0:
							break #Any points coming after must also have a gap due to sort order
			#Otherwise add the candidate for consideration
			candidates.append([elem, np.argmax(scores)])
			if len(candidates) == 3:
				break

	#If 3 candiates remain, try all 6 possible combinations of the candidates against eachother in the parser. If any candidate has both an above and a below relationship, use it as the baseline
	middlefound = False
	if len(candidates) == 3:
		relations = {0:[], 1:[], 2:[]}
		for combo in itertools.permutations([0, 1, 2], 2):
			features = parsingextractor.computefeatures(copy.deepcopy([x[0] for x in [strokedata[s] for s in candidates[combo[0]][0][2]]]), copy.deepcopy([x[0] for x in [strokedata[s] for s in candidates[combo[1]][0][2]]]))
			relations[combo[0]].append(parser.predict([features])[0])
		for key in relations:
			if "Above" in relations[key] and "Below" in relations[key]:
				relationships.append([traversalorder[0][0], candidates[key][0][0], "Right"])
				middlefound = True
				break
	#If the above method fails, take the candidate with the highest probability as the baseline
	if not middlefound:
		#If there are potential subsequent baseline characters
		if len(candidates) > 0:
			maxelem = -1
			maxscore = -float('inf')
			for candidate in candidates:
				if candidate[1] > maxscore:
					maxscore = candidate[1]
					maxelem = candidate
			print("maxelem")
			print(maxelem[0][0])
			relationships.append([traversalorder[0][0], maxelem[0][0], "Right"])

	#Recursively call on the remaining data
	chosenID = relationships[-1][1]
	print(chosenID)
	chosenIndex = 0
	if len(candidates) > 0:
		chosenIndex = [x[0] for x in traversalorder].index(chosenID)
	else: #No further baselines, so look to the end of the list instead
		chosenIndex = len(traversalorder)

	#Separate symbols between the two baseline characters
	above = []
	below = []
	sup = []
	sub = []
	inside = []
	right = [] #Non-baseline right relations need to be re-ordered to come after the new traversalorder root

	for elem in traversalorder[1:chosenIndex]:
		#Compare the initial element to the next element
		features = parsingextractor.computefeatures(copy.deepcopy([x[0] for x in [strokedata[s] for s in traversalorder[0][2]]]), copy.deepcopy([x[0] for x in [strokedata[s] for s in elem[2]]]))
		#Check the evaluated class
		relation = parser.predict([features])[0]

		print(relation)

		if relation == 'Above':
			above.append(elem)
		elif relation == 'Below':
			below.append(elem)
		elif relation == 'Sub':
			sub.append(elem)
		elif relation == 'Sup':
			sup.append(elem)
		elif relation == 'Inside':
			inside.append(elem)
		elif relation == 'Right':
			right.append(elem)

	#Recursively handle symbols above the current baseline character
	if len(above) >= 1:
		relationships.append([traversalorder[0][0], above[0][0], 'Above'])
		if len(above) >= 2:
		   relationships = parserecursive(strokedata, classlabels, strokegroups, parser, medianwidth, above, relationships)

	#Recursively handle symbols below the current baseline character
	if len(below) >= 1:
		relationships.append([traversalorder[0][0], below[0][0], 'Below'])
		if len(below) >= 2:
		   relationships = parserecursive(strokedata, classlabels, strokegroups, parser, medianwidth, below, relationships)

	#Recursively handle superscript symbols
	if len(sup) >= 1:
		relationships.append([traversalorder[0][0], sup[0][0], 'Sup'])
		if len(sup) >= 2:
		   relationships = parserecursive(strokedata, classlabels, strokegroups, parser, medianwidth, sup, relationships)

	#Recursively handle subscript characters
	if len(sub) >= 1:
		relationships.append([traversalorder[0][0], sub[0][0], 'Sub'])
		if len(sub) >= 2:
		   relationships = parserecursive(strokedata, classlabels, strokegroups, parser, medianwidth, sub, relationships)

	#Recursively handle symbols inside square roots
	if len(inside) >= 1:
		relationships.append([traversalorder[0][0], inside[0][0], 'Inside'])
		if len(inside) >= 2:
		   relationships = parserecursive(strokedata, classlabels, strokegroups, parser, medianwidth, inside, relationships)

	#If there are more baselines to invesigate then look into those
	if chosenIndex != len(traversalorder):
		#Modify the traversal order in the event that non-baseline characters occur as right relations to the current baseline symbol
		for i, elem in extractor.reverseenumerate(right):
			traversalorder.insert(chosenIndex+1, elem)

		#Move on to the next baseline character
		relationships = parserecursive(strokedata, classlabels, strokegroups, parser, medianwidth, traversalorder[chosenIndex:], relationships)

	return relationships

def segandsave(filename, classifier, parser):
	"""
	A function to segment and classify symbols
	"""

	#Determine the output filename
	name = filename.split('/')[-1].split('.')[0]

	#Attempt to create ground truth file for comparison
	featset = getfeatures(filename)
	if len(featset) != 0:
		#Create truth .lg values to compare against
		subprocess.run(["crohme2lg", filename, "GroundTruth/"+name+".lg"])


	#Perform segmentation based on strokes in the file
	strokedata = getstrokes(filename) #[[pointlist, name], ...]

	### Pre-Processesing ###

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

	#Assume strokes are written in temporal order, with maximum length of 4
	checklist = ['\sin', '\cos', '\\tan', '\lim', '\log', '\ldots']
	maxlength = 4
	bestscores = [[1, -1, 'unknown'] for x in range(-maxlength, len(strokedata))] #[score, backwards pointer, classification]
	for i in range(maxlength, len(bestscores)):
		bestscore = -float('inf')
		bestsymbol = 'unknown'
		bestbackpointer = -1
		for c in range(0, maxlength): #Consider back-tracking up to 3 characters
			candidateindices = [j for j in range(i-maxlength-c, i-maxlength+1)]
			#Ignore candidates involving non-existent characters
			if candidateindices[0] < 0:
				break

			#Reject candidates involving an x-gap greater than threshold level between first and last points
			#unless those candidates are known 3 letter symbols
			xmaxi = max([x[0] for x in strokedata[candidateindices[0]][0]])
			xmini = min([x[0] for x in strokedata[candidateindices[0]][0]])
			xmaxj = max([x[0] for x in strokedata[candidateindices[-1]][0]])
			xminj = min([x[0] for x in strokedata[candidateindices[-1]][0]])
			rejectflag = False
			if xmini <= xminj:
				if xmaxi + medianwidth/8.0 < xminj - medianwidth/8.0:
					rejectflag = True
			else:
				if xmaxj + medianwidth/8.0 < xmini - medianwidth/8.0:
					rejectflag = True

			features = extractor.computefeatures(copy.deepcopy([x[0] for x in [strokedata[s] for s in candidateindices]]))
			scores = classifier.predict_proba([features])[0]
			if not rejectflag:
				score = max(scores)*bestscores[i-len(candidateindices)][0]
			else:
				#Candidate must be inside pre-approved multi-stroke list
				proposedclass = classifier.classes_[np.argmax(scores)]
				if proposedclass not in checklist:
					continue
				#Must be at least 70% certain for consideration
				#to prevent biasing towards large symbols
				score = max(scores)
				if score < 0.7:
					continue
				score = max(scores)*bestscores[i-len(candidateindices)][0]

			if score > bestscore:
				bestscore = score
				bestsymbol = classifier.classes_[np.argmax(scores)]
				bestbackpointer = i-len(candidateindices)

		bestscores[i] = [bestscore, bestbackpointer, bestsymbol]

	#Recover the optimal segmentation
	classlabels = {}
	strokegroups = {}
	symbolID = 0
	decodeindex = len(bestscores)-1
	while decodeindex > maxlength-1:
		classlabels[symbolID] = bestscores[decodeindex][2]
		indicesused = [i-maxlength for i in range(bestscores[decodeindex][1]+1, decodeindex+1)]
		strokegroups[symbolID] = [x[1] for x in [strokedata[s] for s in indicesused]]
		symbolID += 1
		decodeindex = bestscores[decodeindex][1]

	parseandsave(name, strokedata, classlabels, strokegroups, parser)

	# #Perform fake parsing
	# relationships = []
	# for i in range(0, len(strokegroups)):
	# 	if i < len(strokegroups)-1:
	# 		relationships.append([i, i+1, 'Right'])

	# #Save an output file
	# emitoutput(name, classlabels, strokegroups, relationships)

def segandsavelauncher(filearray, classifier, parser, progresstracker, lock):
	"""Used to report progress over multiple files"""
	for filename in filearray:
		segandsave(filename, classifier, parser)
		with lock:
			progresstracker.value += 1
		gc.collect()

def parseandsavelauncher(filearray, parser, progresstracker, lock):
	"""
	Used to 'cheat' at parsing by starting from the correct segmentation
	"""
	for filename in filearray:
		name = filename.split('/')[-1].split('.')[0]
		#Generate a ground truth file
		subprocess.run(["crohme2lg", filename, "GroundTruth/"+name+".lg"])
		#Load the stroke data
		strokedata = getstrokes(filename) #[[pointlist, name], ...]
		#Read the true segmentations and classifications
		pipe = subprocess.run(["crohme2lg", filename], universal_newlines=True, stdout=subprocess.PIPE)
		lgtext = pipe.stdout
		lgtext = lgtext.split('\n')
		counter = 0
		classlabels = {}
		strokegroups = {}
		for line in lgtext:
			line = line.split(',')
			if line[0] == "O":
				#Assign the classification
				classlabels[counter] = line[2].strip()
				if classlabels[counter] == "COMMA":
					classlabels[counter] = ","
				#Assign the segmentation
				strokes = []
				for i in range(4, len(line)):
					strokes.append(line[i].strip())
				strokegroups[counter] = strokes
				#Increment Counter
				counter += 1
		#Launch the parser
		parseandsave(name, strokedata, classlabels, strokegroups, parser)
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

		### Classifier ###

		#Load training data
		if not os.path.isfile('ClassifierCorpus'):
			featclassdata = loadink(argv[1])
			corpusfile = open('ClassifierCorpus', 'wb')
			pickle.dump(featclassdata, corpusfile)
			corpusfile.close()
		else:
			corpusfile = open('ClassifierCorpus', 'rb')
			featclassdata = pickle.load(corpusfile)
			corpusfile.close()

		#Train the classifier
		print("Training Classifier...")
		attributes = [f[0] for f in featclassdata]
		classes = [f[1] for f in featclassdata]

		classifier = RandomForestClassifier(max_depth=None,
									max_features=0.22498063,
									criterion="entropy",
									bootstrap=False,
									min_samples_leaf=3,
									min_samples_split=7,
									n_estimators=400,
									random_state=0,
									n_jobs=-1)

		classifier.fit(attributes, classes)

		print("Classifier Training Accuracy: {}".format(classifier.score(attributes, classes)))

		#Set up save directory
		if not os.path.exists("ClassifierForest"):
			os.makedirs("ClassifierForest")
		joblib.dump(classifier, "ClassifierForest/trainedclassifier.pkl")

		### Parser ###

		#Load training data
		if not os.path.isfile('ParserCorpus'):
			featrelationdata = loadink(argv[1], function=getrelationships)
			corpusfile = open('ParserCorpus', 'wb')
			pickle.dump(featrelationdata, corpusfile)
			corpusfile.close()
		else:
			corpusfile = open('ParserCorpus', 'rb')
			featrelationdata = pickle.load(corpusfile)
			corpusfile.close()

		#Train the classifier
		print("Training Parser...")
		attributes = [f[0] for f in featrelationdata]
		classes = [f[1] for f in featrelationdata]


		# parser = RandomForestClassifier(n_jobs=-1)

		# param_grid = {'max_depth': [1,2,3,4,5,6,None],
		# 				'min_samples_leaf': sp_randint(1, 11), #3
		# 				'min_samples_split': sp_randint(1, 11), #7
		# 				'bootstrap': [True, False],
		# 				'criterion': ["gini", "entropy"],
		# 				'max_features': [1,2,3,4,5,6,7],
		# 				'n_estimators': [5,10,20,50,100,200,400,600]}

		# param_grid = {'max_depth': [None],
		# 				'min_samples_leaf': sp_randint(1, 11), #3
		# 				'min_samples_split': sp_randint(1, 11), #7
		# 				'bootstrap': [True],
		# 				'criterion': ["entropy"],
		# 				'max_features': [1,2,3,4,5,6,7,8],
		# 				'n_estimators': [5,10,20,50,100,200,400,600]}

		# n_iter_search = 50
		# random_search = RandomizedSearchCV(parser, param_grid, n_iter=n_iter_search, cv=3, refit=True, verbose=4)

		# start = time.time()
		# random_search.fit(attributes, classes)
		# print("RandomizedSearchCV took %.2f seconds for %d candidates"
		#	   " parameter settings." % ((time.time() - start), n_iter_search))
		# report(random_search.grid_scores_)

		parser = RandomForestClassifier(max_depth=None,
									max_features=2,
									criterion="entropy",
									bootstrap=True,
									min_samples_leaf=1,
									min_samples_split=4,
									n_estimators=600,
									random_state=0,
									n_jobs=-1)

		parser.fit(attributes, classes)

		print("Parser Training Accuracy: {}".format(parser.score(attributes, classes)))

		#Set up save directory
		if not os.path.exists("ParserForest"):
			os.makedirs("ParserForest")
		joblib.dump(parser, "ParserForest/trainedclassifier.pkl")

		return

	#Otherwise the user should already have trained forest file
	if not os.path.isdir('ClassifierForest'):
		print("Error: Trained Model not found. Run with -Train flag to create one.")
		usage()
		return
	if not os.path.isfile('ClassifierForest/trainedclassifier.pkl'):
		print("Error: Trained Model not found. Run with -Train flag to create one.")
		usage()
		return
	if not os.path.isdir('ParserForest'):
		print("Error: Trained Model not found. Run with -Train flag to create one.")
		usage()
		return
	if not os.path.isfile('ParserForest/trainedclassifier.pkl'):
		print("Error: Trained Model not found. Run with -Train flag to create one.")
		usage()
		return

	#Set up output folders to save into
	if not os.path.exists("Predictions"):
		os.makedirs("Predictions")
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


	#### MULTITHREADED ####
	threadcount = min(len(filenames), 8)
	#Instantiate classifier copies
	classifiers = [0 for x in range(0, threadcount)]
	for i in range(0, threadcount):
		classifiers[i] = joblib.load('ClassifierForest/trainedclassifier.pkl')

	parsers = [0 for x in range(0, threadcount)]
	for i in range(0, threadcount):
		parsers[i] = joblib.load('ParserForest/trainedclassifier.pkl')

	#Instantiate threads
	threads = []
	progresstracker = Value('i', 0)
	lock = Lock()
	sys.stdout.write("\rParsing: 0.0%")
	for i in range(threadcount):
		if i < len(filenames):
			# t = Process(target=segandsavelauncher, args=(filenames[i::threadcount], classifiers[i], parsers[i], progresstracker, lock))
			t = Process(target=parseandsavelauncher, args=(filenames[i::threadcount], parsers[i], progresstracker, lock))
			threads.append(t)
			t.start()
			#Sleep between launches helps if computer is low on RAM
			time.sleep(5)

	#Monitor progress
	while True:
		with lock:
			sys.stdout.write("\rParsing: {0:0.1f}%".format(100*(progresstracker.value)/len(filenames)))
			if progresstracker.value == len(filenames):
				break
		time.sleep(5)

	sys.stdout.write("\rParsing: 100.0%")
	sys.stdout.write("\n")
	for thread in threads:
		thread.join()

	# ### SERIAL ####
	# classifier = joblib.load('ClassifierForest/trainedclassifier.pkl')
	# parser = joblib.load('ParserForest/trainedclassifier.pkl')

	# sys.stdout.write("\rSegmenting: 0.0%")
	# for i, name in enumerate(filenames):
	# 	segandsave(name, classifier, parser)
	# 	sys.stdout.write("\rSegmenting: {0:0.1f}%".format(100*(i+1)/len(filenames)))
	# sys.stdout.write("\n")



if __name__ == '__main__':
	main(sys.argv[1:])
