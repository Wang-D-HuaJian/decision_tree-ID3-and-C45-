from math import log
import operator

#######################   ID3算法
def calcShannonEnt(dataSet):
    """计算熵"""
    numEntries = len(dataSet)#显式声明一个变量保存实例总数
    labelCounts = {}
    for featVec in dataSet:
        currentLabel = featVec[-1]
        if currentLabel not in labelCounts.keys():
            labelCounts[currentLabel] = 0
        labelCounts[currentLabel] += 1
    shannonEnt = 0.0
    #计算香农熵
    for key in labelCounts:
        prob = float(labelCounts[key]/numEntries)
        shannonEnt -= prob * log(prob,2)
    return shannonEnt

def createDataSet():
     """鉴定数据集"""
     dataSet = [[1, 1, 'yes'],
                [1, 1, 'yes'],
                [1, 0, 'no'],
                [0, 1, 'no'],
                [0, 1, 'no']]
     labels = ['no surfacing', 'flippers']
     return dataSet, labels

def splitDataSet(dataSet, axis, value):#(数据集，划分数据集的特征（索引），需要返回的特征的值)
    """按照给定特征划分数据集"""
    retDataSet = []
    for featVec in dataSet:
        if featVec[axis] == value:#发现符合要求的值
            reducedFeatVec = featVec[:axis]#将索引axis之前的数据提出来，存入列表
            reducedFeatVec.extend(featVec[axis+1:])#将数据中索引axis之后的数据提出来，扩展到上面的列表中
            retDataSet.append(reducedFeatVec)####存入新建的列表中
    return retDataSet

def chooseBestFeatureToSplit(dataSet):
    """选择最好的数据集划分方式(ID3算法)"""
    numFeatures = len(dataSet[0]) - 1
    baseEntropy = calcShannonEnt(dataSet)
    #设定最好的信息增益和特征
    bestInfoGain = 0.0
    bestFeature = -1
    #创建唯一的分类标签
    for i in range(numFeatures):
        featList = [example[i] for example in dataSet]#每次循环都跟换一个特征作为标签
        uniqueVals = set(featList)#创建集合得到列表中唯一元素值
        #计算每种划分方式的信息熵
        newEntropy = 0.0
        for value in uniqueVals:
            subDataSet = splitDataSet(dataSet, i, value)
            prob = len(subDataSet)/float(len(dataSet))
            newEntropy += prob * calcShannonEnt(subDataSet)
        #计算最好的信息增益 （熵的减少或者是数据无序度的减少）
        InfoGain = baseEntropy - newEntropy
        if InfoGain > bestInfoGain:
            bestInfoGain = InfoGain
            bestFeature = i
#    print("最佳特征是： " + str(bestFeature) + "\n" + "最佳信息增益是： " + str(bestInfoGain))
    return bestFeature


def majorityCnt(classList):
    """多数表决决定叶子节点的分类"""
    classCount = {}
    for vote in classList:
        if vote not in classCount.keys():
            classCount[vote] = 0
        classCount[vote] += 1
    sortedClassCount = sorted(classCount.items(), key=operator.itemgetter(1), reverse=True)
    return sortedClassCount[0][0]

def createTree(dataSet, labels):
    """创建树的函数代码"""
    classList = [example[-1] for example in dataSet]
    if classList.count(classList[0]) == len(classList):#####第一个停止条件
        return classList[0]#########类别完全相同则停止继续划分
    #遍历完所有特征时返回出现次数最多的类别
    if len(dataSet[0]) == 1:
        return majorityCnt(classList)

    bestFeat = chooseBestFeatureToSplit(dataSet)
    bestFeatLable = labels[bestFeat]
    myTree = {bestFeatLable:{}}
    subLabels = labels[:]####Python语言中函数参数是列表类型的，参数是按照引用方式传递的。为了保护原始列表的内容，使用新变量来代替
    del (subLabels[bestFeat])

    #得到列表包含的所有属性值
    featValues = [example[bestFeat] for example in dataSet]
    uniqueVals = set(featValues)
    #遍历当前选择特征包含的所有属性值，在每个数据集划分上递归调用函数createTree（）
    for value in uniqueVals:
        myTree[bestFeatLable][value] = createTree(splitDataSet(dataSet, bestFeat, value), subLabels)
    return myTree

def classify(inputTree,featLabels,testVec):
    firstStr = list(inputTree.keys())[0]
    secondDict = inputTree[firstStr]
    featIndex = featLabels.index(firstStr)
    key = testVec[featIndex]
    valueOfFeat = secondDict[key]
    if isinstance(valueOfFeat, dict):
       classLabel = classify(valueOfFeat, featLabels, testVec)
    else:
       classLabel = valueOfFeat
    return classLabel

def storeTree(inputTree,filename):
    import pickle
    fw = open(filename, 'wb')
    pickle.dump(inputTree, fw)
    fw.close()

def grabTree(filename):
    import pickle
    fr = open(filename, 'rb')
    return pickle.load(fr)


####测试分类算法
# dataSet, labels = createDataSet()
# mytree = createTree(dataSet, labels)#创建决策树时，labels中的最优标签被删除了
# storeTree(mytree, 'classifierStorage.txt')
# decision_tree = grabTree('classifierStorage.txt')
# print(classify(decision_tree, labels, [1,1]))
# classify(mytree1, labels, [1,0])


############C4.5算法
def calcConditionalEntropy(dataSet, i, featList, uniqueVals):
    """计算条件熵"""
    conditionEnt = 0.0
    for value in uniqueVals:
        subDataSet = splitDataSet(dataSet, i, value)
        prob = len(subDataSet) / float(len(dataSet))
        conditionEnt += prob * calcShannonEnt(subDataSet)
    return conditionEnt

def calcInformationGain(dataSet, baseEntropy, i):
    """计算信息增益"""
    featList = [example[i] for example in dataSet]
    uniqueVals = set(featList)
    newEntropy = calcConditionalEntropy(dataSet, i, featList, uniqueVals)
    infoGain = baseEntropy - newEntropy
    return infoGain

def calcInformationGainRatio(dataSet, baseEntropy, i):
    """计算信息增益比"""
    return calcInformationGain(dataSet, baseEntropy, i) / baseEntropy

########C4.5算法选择最优属性划分数据函数
def chooseBestFeatureToSplitByC45(dataSet):
    numFeatures = len(dataSet[0]) - 1
    baseEntropy = calcShannonEnt(dataSet)
    bestInfoGainRate = 0.0
    bestFeature = -1
    for i in range(numFeatures):
        infoGainRate = calcInformationGainRatio(dataSet, baseEntropy, i)
        if infoGainRate > bestInfoGainRate:
            bestInfoGainRate = infoGainRate
            bestFeature = i
    return bestFeature

def createTreeC45(dataSet, labels):
    """创建树的函数代码"""
    classList = [example[-1] for example in dataSet]
    if classList.count(classList[0]) == len(classList):#####第一个停止条件
        return classList[0]#########类别完全相同则停止继续划分
    #遍历完所有特征时返回出现次数最多的类别
    if len(dataSet[0]) == 1:
        return majorityCnt(classList)

    bestFeat = chooseBestFeatureToSplitByC45(dataSet)
    bestFeatLable = labels[bestFeat]
    myTree = {bestFeatLable:{}}
    subLabels = labels[:]####Python语言中函数参数是列表类型的，参数是按照引用方式传递的。为了保护原始列表的内容，使用新变量来代替
    del (subLabels[bestFeat])

    #得到列表包含的所有属性值
    featValues = [example[bestFeat] for example in dataSet]
    uniqueVals = set(featValues)

    #遍历当前选择特征包含的所有属性值，在每个数据集划分上递归调用函数createTree（）
    for value in uniqueVals:
        myTree[bestFeatLable][value] = createTree(splitDataSet(dataSet, bestFeat, value), subLabels)
    return myTree