dataSet = []
labels =[]

def createDataSet(fileName):

    with open(fileName) as ifile:
        for line in ifile:
            tokens = line.strip().split(',')
            dataSet.append(tokens)

    labels =['buying','maint','doors','persons','lug_boot','safety']
    return dataSet,labels