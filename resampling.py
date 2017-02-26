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