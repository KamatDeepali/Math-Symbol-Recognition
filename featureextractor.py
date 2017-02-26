"""
File: featureextractor.py
Authors: Michael Potter, Deepali Kamat
Description: A program to compute features from .inkml files
"""

import sys
import os
import shutil
import codecs
import math
import numpy as np
import scipy as sp
from bs4 import BeautifulSoup
import matplotlib.pyplot as plt


newList =[]

GLOBALMAX = [-float('inf')]

def usage():
    """
    Prints the usage message in response to incorrect input
    """
    print("Usage: <Test or Train> <Data Folder>")

def folderinit(folder):
    """
    Creates and/or empties the given folder
    """
    if not os.path.exists(folder):
        os.makedirs(folder)
    else:
        #Clear out all files from the folder
        for f in os.listdir(folder):
            path = os.path.join(folder, f)
            try:
                if os.path.isfile(path):
                    os.unlink(path)
                #Clears subdirectories
                elif os.path.isdir(path): shutil.rmtree(path)
            except:
                print("Error: Could Not Clear {} Folder".format(folder))


def pointCount(strokelist):
    count = 0
    for stroke in strokelist:
         count += len(stroke);
    return count



def reverseenumerate(iterable):
    """
    Enumerate over an iterable in reverse order while retaining proper indexes
    Referenced from: http://galvanist.com/post/53478841501/python-reverse-enumerate
    """
    return zip(reversed(range(len(iterable))), reversed(iterable))

def duplicatepointfiltering(stroke):
    """
    A function to remove duplicate points from a stroke
    """
    for i,point in reverseenumerate(stroke):
        if i == 0:
            continue
        if point[0] == stroke[i-1][0] and point[1] == stroke[i-1][1]:
            stroke = np.delete(stroke, i, axis=0)
    return stroke

def normalization(strokelist, flip=False):
    """
    A function to normalize points from a list of strokes
    """
    #Identify the max and min y-coordinates (will shink to [0-1])
    minY = float('inf')
    maxY = -float('inf')
    minX = float('inf')
    for stroke in strokelist:
        if min(stroke[:,1]) < minY:
            minY = min(stroke[:,1])
        if max(stroke[:,1]) > maxY:
            maxY = max(stroke[:,1])
        if min(stroke[:,0]) < minX:
            minX = min(stroke[:,0])

    #normalize the x and y coordinates so that y ranges 0-1
    factor = maxY-minY
    #If no variability in the y direction then don't scale
    if factor == 0:
        factor = 1
        minY -= 1 #Force the y value to 1
    for stroke in strokelist:
        for point in stroke:
            point[0] = (point[0]-minX)/factor
            if flip:
                point[1] = 1-(point[1]-minY)/factor
            else:
                point[1] = (point[1]-minY)/factor

    # if factor == 0:
    #     visualize(strokelist)

    return strokelist

def smoothing(stroke):
    """
    function to smooth data point by avaraging the prior, present and next point
    """
    for i in range(1, len(stroke)-1):
        stroke[i][0] = (stroke[i-1][0]+stroke[i][0]+stroke[i+1][0])/3
        stroke[i][1] = (stroke[i-1][1]+stroke[i][1]+stroke[i+1][1])/3
    return stroke

def resampling(stroke, alpha):
    """
    A function to resample the stroke to remove velocity effects
    input: a list of data points to be resampled and the desired point density
    output: the resampled list of points
    """
    n = len(stroke)
    length = np.zeros(n)
    length[0] = 0
    #Compute the accumulated distance
    for i in range(1, n):
        length[i] = length[i-1] + math.sqrt(((stroke[i][0] - stroke[i-1][0])**2) + (stroke[i][1] - stroke[i-1][1])**2)

    #Compute new points based on the total accumulated length and desired density
    m = int(length[n-1]/alpha)+1
    newstroke = np.zeros((m, 2))
    newstroke[0][0] = stroke[0][0]
    newstroke[0][1] = stroke[0][1]
    j = 1
    for p in range(1,m-1):
        while length[j] < (p*alpha):
            j = j+1
        C = ((p*alpha)-length[j-1])/(length[j]-length[j-1])
        newstroke[p][0] = stroke[j-1][0]+(stroke[j][0]-stroke[j-1][0])*C
        newstroke[p][1] = stroke[j-1][1]+(stroke[j][1]-stroke[j-1][1])*C

    newstroke[m-1][0] = stroke[n-1][0]
    newstroke[m-1][1] = stroke[n-1][1]

    return newstroke

def preprocess(strokelist):
    """
    A function to perform data pre-processing
    """
    strokelist = normalization(strokelist, flip=True)
    for i, stroke in enumerate(strokelist):
        stroke = duplicatepointfiltering(stroke)
        stroke = smoothing(stroke)
        strokelist[i] = resampling(stroke, 0.05)
    strokelist = normalization(strokelist)
    return strokelist

def visualize(strokelist):
    """
    A function to visualize a combination of strokes
    """
    for stroke in strokelist:
        x1 = [pair[0] for pair in stroke]
        y1 = [pair[1] for pair in stroke]
        plt.plot(x1, y1)
        for i in range(0, len(x1)):
            plt.plot(x1[i], y1[i], 'ro')
    plt.show()
    plt.clf()

def magnitude(a):
    """
    A function to compute the magnitude of a vector
    """
    result = 0
    for dim in a:
        result += dim**2
    result = math.sqrt(result)
    return result

def dotdivide(a, b):
    """
    A function to compute the normalized dot product between two vectors
    """
    if magnitude(a) == 0 or magnitude(b) == 0:
        return 0
    result = 0
    for dimension in range(len(a)):
        result += (a[dimension] * b[dimension])
    result /= (magnitude(a)*magnitude(b))
    return min(max(result,-1),1)

def angularchange(stroke):
    """
    A function to compute the total running change in angle over a single stroke
    """
    if len(stroke) < 3:
        return 0
    anglechange = 0
    oldangle = np.arccos(dotdivide(stroke[1]-stroke[0], stroke[2]-stroke[1]))
    for i in range(3, len(stroke)):
        angle = np.arccos(dotdivide(stroke[i-1]-stroke[i-2], stroke[i]-stroke[i-1]))
        anglechange += abs(angle-oldangle)
        oldangle = angle
    return anglechange

def linelength(stroke):
    """
    A function to compute the euclidean distance traversed by a single stroke
    """
    totallength = 0
    count =0
    for i in range(1, len(stroke)):
        totallength += math.sqrt(((stroke[i][0] - stroke[i-1][0])**2) + (stroke[i][1] - stroke[i-1][1])**2)
        count = count +1
    return totallength

def sharppoints(stroke):
    """
    function to compute sharp points based on the changing slope value of points in stroke
    """
    n = len(stroke)
    count = 2 # Include first and last point

    if len(stroke) < 3:
        return count
    for i in range(1, len(stroke)-1):
        angle = np.arccos(dotdivide(stroke[i]-stroke[i-1], stroke[i+1]-stroke[i]))
        if (angle > 2.007) and (angle < 4.27):
            count += 1

    return count

def aspectratio(strokelist):
    """
    A function to compute the aspect ratio of a set of strokes
    """
    xmax = -float('inf')
    xmin = float('inf')
    ymax = -float('inf')
    ymin = float('inf')
    for stroke in strokelist:
        if max(stroke[:, 0]) > xmax:
            xmax = max(stroke[:, 0])
        if min(stroke[:, 0]) < xmin:
            xmin = min(stroke[:, 0])
        if max(stroke[:, 1]) > ymax:
            xmax = max(stroke[:, 1])
        if min(stroke[:, 1]) < ymin:
            xmin = min(stroke[:, 1])
    width = xmax - xmin
    height = ymax - ymin
    ratio = width/height
    return ratio

def mean(strokelist, coordinate):
    """
    Compute the mean value along a given axis
    """
    m = 0
    pointcount = 0
    for stroke in strokelist:
        for point in stroke:
            m += point[coordinate]
            pointcount += 1
    m /= pointcount
    return m

def covarianceXY(strokelist):
    """
    Computes the global covariance between x and y
    """
    data = np.vstack(strokelist)
    #If you have a single point, then x and y are essentially identical
    if len(data) == 1:
        return 1
    return np.cov(data, rowvar=0)[0, 1]

def linearcrossings(strokelist, axis):
    """
    Compute the horizontal crossings of a symbol
    """
    #Subdivide space into 5 regions
    maxlevel = max(np.ndarray.max(np.vstack(strokelist), 0)[axis], 1)
    minlevel = min(np.ndarray.min(np.vstack(strokelist), 0)[axis], 0)
    step5 = (maxlevel-minlevel)/5
    regions = np.arange(minlevel, maxlevel+step5-1e-8, step5) #Subtract small value to defend against rounding error
    if len(regions) != 6:
        print("Region Error: {}".format(len(regions)))

    #Subdivide each region into 10 threshold levels
    thresholds = []
    for i in range(1, len(regions)):
        step10 = (regions[i]-regions[i-1])/15 #Divide by 15 to get buffer from region edge
        thresholds.append(np.arange(regions[i-1]+3*step10, regions[i]-2*step10, step10))

    #Track the average number of crossings per region
    crossingcounts = [0 for x in range(len(thresholds))]
    #Track the average min coordinate at which crossings occur
    crossingmins = [0 for x in range(len(thresholds))]
    #Track the average max coordinate at which crossings occur
    crossingmaxs = [0 for x in range(len(thresholds))]
    #For every region...
    for r in range(0, len(thresholds)):
        #Consider every threshold in that region...
        for threshold in thresholds[r]:
            #Track the max and min crossing values
            tmin = float('inf')
            tmax = -float('inf')
            #If adjacent points are on opposite sides of the threshold...
            for stroke in strokelist:
                for p in range(1,len(stroke)):
                    # print(stroke[p][axis])
                    if ((stroke[p][axis] <= threshold) and (stroke[p-1][axis] > threshold)) or \
                        ((stroke[p][axis] >= threshold) and (stroke[p-1][axis] < threshold)):
                        #Increment crossing index
                        crossingcounts[r] += 1
                        #Estimate the coordinate at which the cross occurs
                        crosspoint = stroke[p][(axis+1)%2] - \
                                    (stroke[p][(axis+1)%2]-stroke[p-1][(axis+1)%2])*\
                                    abs(stroke[p][axis]-threshold)/abs(stroke[p][axis]-stroke[p-1][axis])
                        # print("crosspoint: {}".format(crosspoint))
                        if crosspoint < tmin:
                            tmin = crosspoint
                        if crosspoint > tmax:
                            tmax = crosspoint
            # print("tmin, tmax: {}, {}".format(tmin, tmax))
            if tmin != float('inf'):
                crossingmins[r] += tmin
            if tmax != -float('inf'):
                crossingmaxs[r] += tmax
    #Normalize data
    for r in range(0, len(thresholds)):
        if crossingcounts[r] > 0:
            crossingmins[r] /= crossingcounts[r]
            crossingmaxs[r] /= crossingcounts[r]
        else:
            crossingmins[r] = 0
            crossingmaxs[r] = 0
        crossingcounts[r] /= len(thresholds[r])

    results = []
    for i in range(len(crossingcounts)):
        results.append(crossingcounts[i])
        results.append(crossingmins[i])
        results.append(crossingmaxs[i])

    return results

def crossings(strokelist):
    """
    Compute the average horizontal and vertical crossing features of a symbol
    """
    result = []
    #Compute horizontal crossings (based on Y coordinate)
    result.extend(linearcrossings(strokelist, 1))
    #Compute vertical crossings (based on X coordinate)
    result.extend(linearcrossings(strokelist, 0))
    return result

def checkPoint(point,grid, w, h, dict1,dict2):
    """
    Checks if a point falls in a particular grid and computes memnbership
    :param point: x, y coordinates of the point in the stroke
    :param grid: the grid being tested
    :return: returns membership
    """
    if (dict1[grid[0]][0]>=  point[0] and point[0] <=dict1[grid[3]][0]) and (dict1[grid[0]][1] >=point[1] and point[1]>=dict1[grid[3]][1]) :
        for i in range(0,4):
            dict2[grid[i]] += ((w- abs(point[0]-dict1[grid[i]][0]))/w)*((h- abs(point[1]-dict1[grid[i]][1]))/h)
    return dict2



def histogramOfPoints(strokelist):
    """
    Computes a normalized fuzzy histogram of coordinate points of the symbol
    :param strokelist: strokelist of the trace symbol
    :return: the array containing the normalized 2D histogram of coordinate points
    """

    dict1 = {}
    dict2 = {}
    result = {}
    maxlevelX = max(np.ndarray.max(np.vstack(strokelist),0)[0],1)
    minlevelX = 0
    stepx = (maxlevelX-minlevelX)/4
    regionX = np.arange(minlevelX, maxlevelX+stepx-1e-8, stepx)

    minlevelY = 0.0
    maxlevelY = 1.0
    stepy = (maxlevelY-minlevelY)/4
    regionY = np.arange(minlevelY, maxlevelY+stepy-1e-8, stepy)

    grid1 = [1, 2, 6, 7]
    grid2 = [2, 3, 7, 8]
    grid3 = [3, 4, 8, 9]
    grid4 = [4, 5, 9, 10]
    grid5 = [6, 7, 11, 12]
    grid6 = [7, 8, 12, 13]
    grid7 = [8, 9, 13, 14]
    grid8 = [9, 10, 14, 15]

    grid9 = [11, 12, 16, 17]
    grid10 = [12, 13, 17, 18]
    grid11 = [13, 14,18, 19]
    grid12 = [14, 15, 19, 20]
    grid13 = [16, 17, 21, 22]
    grid14 = [17, 18, 22, 23]
    grid15 = [18, 19, 23, 24]
    grid16 = [19, 20, 24, 25]

    gridList = [grid1, grid2, grid3, grid4, grid5, grid6, grid7, grid8,grid9, grid10, grid11, grid12,grid13, grid14, grid15, grid16]

    i=1
    for x in regionX:
        for y in regionY:
            dict1[i] = [x, y] #dictionary to store the grid of corner points
            dict2[i] = 0.0 #dictionary storing membership values
            i = i+1


    for stroke in strokelist:
        for point in stroke:
           for grid in gridList:
               dict2 = checkPoint(point, grid, stepx, stepy, dict1,dict2)

    #Flatten the histogram and normalize it by the number of points
    numpoints = pointCount(strokelist)
    finalList = [x[1]/numpoints for x in sorted(dict2.items(), key=lambda t: t[0])]
    return finalList



def histogramOfOrientations(strokelist):
    """
    Computes a fuzzy orientation histogram based on the given strokelist
    """
    bins = {}

    maxX = max(np.ndarray.max(np.vstack(strokelist),0)[0],1)
    minX = 0
    stepx = (maxX-minX)/2
    regionX = np.arange(minX, maxX+stepx-1e-8, stepx)

    minY = 0.0
    maxY = 1.0
    stepy = (maxY-minY)/2
    regionY = np.arange(minY, maxY+stepy-1e-8, stepy)

    for x in regionX:
        for y in regionY:
            bins[(x,y)] = [0, 0, 0, 0]

    grid = {}
    grid[(3*maxX/4,0.75)] = [(maxX,0.5), (maxX,1), (maxX/2,1), (maxX/2,0.5)]
    grid[(maxX/4,0.75)] = [(maxX/2,0.5), (maxX/2,1), (0,1), (0,0.5)]
    grid[(maxX/4,0.25)] = [(maxX/2,0), (maxX/2,0.5), (0,0.5), (0,0)]
    grid[(3*maxX/4,0.25)] = [(maxX, 0), (maxX,0.5), (maxX/2,0.5), (maxX/2,0)]

    binCenters = [0, math.pi/4, 7*math.pi/4, math.pi/2, 3*math.pi/2]
    #Decide which quadrant each point belongs to
    for stroke in strokelist:
        #For every pair of points...
        for i in range(0,len(stroke)-1):
            midpoint = ((stroke[i][0]+stroke[i+1][0])/2, (stroke[i][1]+stroke[i+1][1])/2)

            #define a left-pointing line
            line = [0,0]
            if stroke[i][0] > stroke[i+1][0]:
                line[0] = stroke[i][0] - stroke[i+1][0]
                line[1] = stroke[i][1] - stroke[i+1][1]
            else:
                line[0] = stroke[i+1][0] - stroke[i][0]
                line[1] = stroke[i+1][1] - stroke[i][1]

            angle = np.arccos(dotdivide([1,0], line))
            if line[1]<0:
                angle = 2*math.pi-angle

            quadrant = (0,0)
            mindist = float('inf')
            for center in grid.keys():
                dist = math.sqrt(pow(center[0]-midpoint[0], 2)+pow(center[1]-midpoint[1], 2))
                if dist < mindist:
                    mindist = dist
                    quadrant = center

            #list of bin keys
            binlist = grid[quadrant]
            #Compute the membership for each bin
            for key in binlist:
                rawmembership = ((stepx- abs(midpoint[0]-key[0]))/stepx)*((stepy- abs(midpoint[1]-key[1]))/stepy)
                for a, bin in enumerate(bins[key]):
                    anglemembership = abs(angle - binCenters[a])
                    if a == 3:
                        anglemembership = min(anglemembership, abs(angle - binCenters[a+1]))
                    anglemembership = 7*math.pi/8 - anglemembership

                    #If you're one of the top 2 bins
                    if anglemembership >= 5*math.pi/8:
                        bins[key][a] += rawmembership*(anglemembership/(3*math.pi/2))
    binlist = [x[1] for x in sorted(sorted(bins.items(), key=lambda k: k[1]), key=lambda k: k[0])]
    binlist = [item for sublist in binlist for item in sublist]

    return binlist






def computefeatures(strokelist):
    """
    input: array of numpy array coordinates corresponding to a symbol
    output: feature vector for that symbol
    """

    #Pre-process the data
    # visualize(strokelist)
    strokelist = preprocess(strokelist)
    # visualize(strokelist)

    #Build feature array
    features = []
    features.append(pointCount(strokelist))
    features.append((pointCount(strokelist))/len(strokelist))
    traceprob=[]


    ### GLOBAL FEATURES ###

    #Number of traces
    features.append(len(strokelist))
    #Angular change
    totalangularchange = 0
    for stroke in strokelist:
        totalangularchange += angularchange(stroke)
    features.append(totalangularchange)
    features.append(totalangularchange/len(strokelist))


    #Line Length
    totallength = 0
    for stroke in strokelist:
        totallength += linelength(stroke)
    features.append(totallength)
    features.append(totallength/len(strokelist))


    #probability of stroke
    for stroke in strokelist:
        traceprob.append(len(stroke)/len(strokelist))

    #Sharp Point Count
    sharpcount = 0
    for stroke in strokelist:
        sharpcount += sharppoints(stroke)
    features.append(sharpcount)
    #Aspect Ratio
    features.append(aspectratio(strokelist))
    #Mean X
    features.append(mean(strokelist,0))
    #Mean Y
    features.append(mean(strokelist,1))
    #Covariance of X and Y
    features.append(covarianceXY(strokelist))

    ### CROSSING FEATURES ###
    features.extend(crossings(strokelist))

    ### FUZZY POINT HISTOGRAMS ###
    features.extend(histogramOfPoints(strokelist))

    ### FUZZY ORIENTATION HISTOGRAMS ###
    features.extend(histogramOfOrientations(strokelist))

    return features


def main(argv):
    """
    A function to parse the given input files and generate the appropriate
    data split
    """
    if not argv:
        usage()
        return
    if len(argv) != 2:
        usage()
        return

    #Find all .inkml files from the subdirectories of the input folder
    files = [os.path.join(root, name)
            for root, dirs, files in os.walk(argv[1])
            for name in files
            if name.endswith((".inkml"))]

    #Create empty feature repositories
    repo = argv[0]
    if repo != 'Train' and repo != 'Test':
        usage()
        return
    folderinit(repo+"_features")

    #Create a feature file for each input file
    for f in files:
        infile = open(f)
        #Identify all of the symbols in the document
        try:
            soup = BeautifulSoup(infile, 'html.parser')
        except UnicodeDecodeError: #File Corruption
            # print("Bad File: {}".format(infilename))
            #Attempt to load file by ignoring corrupted characters
            with codecs.open(f, "r", encoding='utf-8', errors='ignore') as fdata:
                soup = BeautifulSoup(fdata, 'html.parser')

        #Determine all tracegroups (removing the first since it is a group of groups)
        tracegroups = soup.find_all("tracegroup")
        tracegroups = tracegroups[1:]

        symbols = []
        totallen = 0
        probtrace = []
        traceDat = []
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
            features = computefeatures(tracedata)


            features = computefeatures(tracedata)
            # this =  max(GLOBALMAX, len(strokelist))

            '''
            features[0] = max(GLOBALMAX[0], len(tracedata))
            totallen = totallen + len(tracedata)
            #print('max length:', features[0])
            if(len(tracedata)>=6):
                print (f)
                #visualize(tracedata)
            '''


            #Determine the true symbol if known
            symbol = '\\unknown'
            if repo == 'Train':
                symbol = ''.join((group.find("annotation")).contents)

            #Append the combination to the files list
            symbols.append([symbol, features])
            traceDat = tracedata


        #Save the features to an output file
        outfile = open(os.path.join(repo+"_features", (f.split("/")[-1]).split(".")[0]+".features"),"w")
        for symbol in symbols:
            outfile.write("<symbol>\n")
            outfile.write("<character>{}</character>\n".format(symbol[0]))
            outfile.write("<features>")
            featstring = ''
            for feature in symbol[1]:
                featstring = featstring + "{},".format(feature)
            outfile.write(featstring[0:-1])
            outfile.write("</features>\n")
            outfile.write("</symbol>\n")
        outfile.close()
        infile.close()

    print (features[0])

if __name__ == "__main__":
    main(sys.argv[1:])

