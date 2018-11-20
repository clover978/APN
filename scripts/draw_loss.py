# coding: utf-8
import matplotlib.pyplot as plt
import sys

trainlines = []
testlines = []
with open('log.txt') as f:
    for line in f:
        if line.startswith('Epoch'):
            trainlines.append(line)
        elif line.startswith(' *'):
            testlines.append(line)

trainloss = []
trainacc = []
testacc = []
for line in trainlines:
    loss = line.split('Loss')[1].split('\t')[0]
    p = loss.index('(')
    loss = loss[:p]
    trainloss.append(float(loss))

    acc = line.split('Prec@1')[1].split('\t')[0]
    p = acc.index('(')
    acc = acc[:p]
    trainacc.append(float(acc))

for line in testlines:
    acc = line.split()[2]
    testacc.append(float(acc))

plt.plot(trainloss[::25])
plt.show()

plt.plot(trainacc[::25])
plt.show()

plt.plot(testacc)
plt.show()