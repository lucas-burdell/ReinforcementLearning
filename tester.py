import csv
from model import Model

model = Model(modelFilename="model-")
validation = csv.reader(open("validation.censusdata"), skipinitialspace=True, delimiter=',')
validationCheck = csv.reader(open("validationcheck.censusdata"), skipinitialspace=True, delimiter=',')
vTotal = 0
vCorrect = 0
vTotalPositives = 0
vTotalNegatives = 0
vFalsePositives = 0
vFalseNegatives = 0



for row in validation:
    vCheckRow = next(validationCheck)
    answer = False
    if vCheckRow:
        answer = vCheckRow[1].strip()
        if answer == ">50K.":
            answer = True
            vTotalPositives += 1
        else:
            answer = False
            vTotalNegatives += 1
    if row:
        vTotal += 1
        result = model.runSingle(row)
        if answer is result:
            vCorrect += 1
        elif answer is False:
            vFalseNegatives += 1
        else:
            vFalsePositives += 1
print("total: " + str(vTotal))
print("correct: " + str(vCorrect))
print("error rate: " + str((vTotal - vCorrect) / vTotal))
print("false negatives: " + str(vFalseNegatives))
print("false positives: " + str(vFalsePositives))
print("total negatives: " + str(vTotalNegatives))
print("total positives: " + str(vTotalPositives))