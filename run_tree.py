import sys
sys.path.append("F:/pythonTest")
from decision_tree.tree import *
from decision_tree.treePlotter import *

myDat, labels = createDataSet()
print(myDat)
myTree = retrieveTree(0)
print(myTree)
print(classify(myTree,labels,[1,0]))
print(classify(myTree,labels,[1,1]))

fr = open('lenses.txt')
lenses=[inst.strip().split('\t') for inst in fr.readlines()]
lensesLabels=['age', 'prescript', 'astigmatic', 'tearRate']
lenseeTree=createTree(lenses, lensesLabels)
lenseeTreeC45 = createTreeC45(lenses, lensesLabels)
createPlot(lenseeTree)
#createPlot(lenseeTreeC45)