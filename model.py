from sys import argv
import csv
import math
import random
import datetime




# map input strings to numbers
# These lookup tables are for normalizing each row of data
workclass = {"Private": 0, "Self-emp-not-inc": 1, "Self-emp-inc": 2, "Federal-gov": 3, "Local-gov": 4, "State-gov": 5,
             "Without-pay": 6, "Never-worked": 7, "?": 8}
workclassList = list(range(len(workclass)))

education = {"Bachelors": 0, "Some-college": 1, "11th": 2, "HS-grad": 3, "Prof-school": 4, "Assoc-acdm": 5,
             "Assoc-voc": 6, "9th": 7, "7th-8th": 8, "12th": 9, "Masters": 10, "1st-4th": 11, "10th": 12,
             "Doctorate": 13, "5th-6th": 14, "Preschool": 15, "?": 16}

educationList = list(range(len(education)))

# ignoring education number; reminder that it's the third entry

occupation = {"Tech-support": 0, "Craft-repair": 1, "Other-service": 2, "Sales": 3, "Exec-managerial": 4,
              "Prof-specialty": 5, "Handlers-cleaners": 6, "Machine-op-inspct": 7, "Adm-clerical": 8, "Farming-fishing": 9,
              "Transport-moving": 10, "Priv-house-serv": 11, "Protective-serv": 12, "Armed-Forces": 13, "?": 14}
occupationList = list(range(len(occupation)))

relationship = {"Wife": 0, "Own-child": 1, "Husband": 2, "Not-in-family": 3, "Other-relative": 4, "Unmarried": 5, "?": 6}
relationshipList = list(range(len(relationship)))

sex = {"Female": 0, "Male": 1}

sexList = list(range(len(sex)))

country = {"United-States": 0, "Cambodia": 1, "England": 2, "Puerto-Rico": 3, "Canada": 4, "Germany": 5,
           "Outlying-US(Guam-USVI-etc)": 6, "India": 7, "Japan": 8, "Greece": 9, "South": 10, "China": 11, "Cuba": 12,
           "Iran": 13, "Honduras": 14, "Philippines": 15, "Italy": 16, "Poland": 17, "Jamaica": 18, "Vietnam": 19,
           "Mexico": 20, "Portugal": 21, "Ireland": 22, "France": 23, "Dominican-Republic": 24, "Laos": 25,
           "Ecuador": 26, "Taiwan": 27, "Haiti": 28, "Columbia": 29, "Hungary": 30, "Guatemala": 31, "Nicaragua": 32,
           "Scotland": 33, "Thailand": 34, "Yugoslavia": 35, "El-Salvador": 36, "Trinadad&Tobago": 37, "Peru": 38,
           "Hong": 39, "Holand-Netherland": 40, "?": 41}
countryList = list(range(len(country)))

# For the training set
earnings = {">50K.": 1, "<=50K.": 0}

# Map types
# Map for quick access of tables based on entry name
# Used mainly to keep track of entries in the neuron table; the key and index number
inputtypes = {"workclass": workclass, "education": education, "occupation": occupation, "relationship": relationship,
              "sex": sex, "country": country, "earnings": earnings}

MAX_WEIGHT = 3

# These lookups are for acting on normalized data-sets
# SKIP EDUCATION NUMBER

# FULL INPUTS USED
# rowInputTables = {0: workclass, 1: education, 3: occupation, 4: relationship, 5: sex, 6: country}
# rowInputLabels = {0: "workclass", 1: "education", 3: "occupation", 4: "relationship", 5: "sex", 6: "country"}
# rowLabelList = ["workclass", "education", "occupation", "relationship", "sex", "country"]
# rowTableList = [workclassList, educationList, occupationList, relationshipList, sexList, countryList]

# REMOVED SEX AND COUNTRY
rowInputTables = {0: workclass, 1: education, 3: occupation, 4: relationship}
rowInputLabels = {0: "workclass", 1: "education", 3: "occupation", 4: "relationship"}
rowLabelList = ["workclass", "education", "occupation", "relationship"]
rowTableList = [workclassList, educationList, occupationList, relationshipList]


earningPosition = 7
LEARNING_RATE = .5


def sigmoid(num):
    return 1 / (1 + math.pow(math.e, -num))


#NEURON CLASSES
class Neuron:

    def __init__(self, inputNeurons, outputNeuron):
        self.weight = 1
        self.value = 0
        self.inputNeurons = inputNeurons or []
        self.output = outputNeuron or None

    def addWeight(self, value):
        self.weight = min(MAX_WEIGHT, max(-MAX_WEIGHT, self.weight + value))
        return self.weight

    def activate(self):
        total = 0
        for neuron in self.inputNeurons:
            total = total + neuron.value * neuron.weight
        self.value = sigmoid(total)


class InputNeuron(Neuron):
    def __init__(self, inputValue, outputNeuron):
        super(InputNeuron, self).__init__(None, outputNeuron)
        self.inputValue = inputValue

    def doInputNeuron(self, inputData):
        if inputData:
            self.value = 1# + self.weight/5)
        else:
            self.value = 0
        #total = 0
        #for neuron in self.inputNeurons:
        #    total = total + neuron.value * neuron.weight
        #self.value = sigmoid(self.value + total)


class Bias(Neuron):

    def __init__(self, value, outputNeuron):
        super(Bias, self).__init__(None, outputNeuron)
        self.weight = 1
        self.value = value

def normalize(row, training=False):
    values = [None] * (len(row))

    for index, val in enumerate(row):
        i = 0

        if training is False:
            if index == 0:
                continue
            else:
                i = index - 1
        else:
            i = index
        val = val.strip()
        if i in rowInputTables:
            values[i] = rowInputTables[i][val]
        else:
            values[i] = val

    # get earnings value
    if training is True:
        values[earningPosition] = earnings[row[earningPosition]]
    return values

def getKeyFromValue(dict, value):
    # from http://stackoverflow.com/questions/8023306/get-key-by-value-in-dictionary
    return list(dict.keys())[list(dict.values()).index(value)]

class Model:
    def __init__(self, modelFilename=None):

        self.inputLayer = { }
        self.hiddenLayer = { }
        self.output = Neuron([], None)
        self.output.weight = 1 # set static weight

        # DIAGNOSIS VALUES
        self.corrects = 0
        self.total = 0
        self.positives = 0
        self.false_positives = 0
        self.false_negatives = 0
        self.count50 = 0
        self.correctOver50 = 0


        inputReader = None
        if modelFilename is not None:
            inputReader = csv.reader(open(modelFilename), skipinitialspace=True, delimiter=',')
        count = 0
        for i in range(len(rowLabelList)):
            name = rowLabelList[i]
            if name is "earnings":
                continue
            self.hiddenLayer[name] = Neuron([], self.output)
            self.output.inputNeurons.append(self.hiddenLayer[name])
            if inputReader is not None:
                line = next(inputReader)
                while not line:
                    line = next(inputReader)
                hiddenWeight = float(line[0].strip())
                self.hiddenLayer[name].weight = hiddenWeight
                #print(name + " is " + str(hiddenWeight))
            #print(rowTableList[i])
            for index in rowTableList[i]:
                newKey = name + "" + index.__str__()
                self.inputLayer[newKey] = InputNeuron(index, self.hiddenLayer[name])
                self.hiddenLayer[name].inputNeurons.append(self.inputLayer[newKey])
                if inputReader is not None:
                    line = next(inputReader)
                    while not line:
                        line = next(inputReader)
                    inputWeight = float(line[0].strip())
                    self.inputLayer[newKey].weight = inputWeight
                    #print(newKey + " is " + str(inputWeight))


    def save(self):
        now = datetime.datetime.now()
        outputfilename = "%d%d%d%d%d%d" % (now.year, now.month, now.day, now.hour, now.minute, now.second)

        modelwriter = open("model-" + outputfilename, "wt")
        modelcsv = csv.writer(modelwriter, skipinitialspace=True, delimiter=",")

        for i in range(len(rowLabelList)):
            name = rowLabelList[i]
            if name is "earnings":
                continue
            modelcsv.writerow([self.hiddenLayer[name].weight])

            for index in rowTableList[i]:
                newKey = name + "" + index.__str__()
                modelcsv.writerow([self.inputLayer[newKey].weight])
        modelwriter.close()

    def learn(self, oldValues, newValues, output, diagnostics):


        if diagnostics:
            # UPDATE DIAGNOSTICS VALUES

            self.total += 1
            if newValues[earningPosition] == 1:
                self.count50 += 1

            # check if output was expected value
            if output:
                self.positives += 1

                # check if output was expected value
            if output == newValues[earningPosition]:
                self.corrects += 1

                if output:
                    self.correctOver50 += 1
            else:
                if output > newValues[earningPosition]:
                    self.false_positives += 1
                else:
                    self.false_negatives += 1

        # END UPDATE DIAGNOSTICS VALUES

        if output == newValues[earningPosition]:
            # pass # remove return and always adjust weights. Interesting
            return

        multiplyBy = (1 if output == 1 else -1) * LEARNING_RATE
        HALF = 0.5
        FULL = 1

        # category changed and output stayed same
        if oldValues[earningPosition] == newValues[earningPosition]:
            for i, val in enumerate(newValues):
                if i in rowInputTables:
                    inputLabel = rowInputLabels[i]
                    if oldValues[i] != val:
                        self.hiddenLayer[inputLabel].addWeight(-HALF * multiplyBy)
                        self.inputLayer[inputLabel + "" + val.__str__()].addWeight(HALF * multiplyBy)
                    else:
                       #pass
                        self.hiddenLayer[inputLabel].addWeight(HALF * multiplyBy)
                        self.inputLayer[inputLabel + "" + val.__str__()].addWeight(FULL * multiplyBy)
        else:
            for i, val in enumerate(newValues):
                if i in rowInputTables:
                    inputLabel = rowInputLabels[i]
                    if oldValues[i] != val:
                        self.hiddenLayer[inputLabel].addWeight(FULL * -multiplyBy)
                        self.inputLayer[inputLabel + "" + val.__str__()].addWeight(FULL * multiplyBy)
                    else:
                        #pass
                        self.hiddenLayer[inputLabel].addWeight(-HALF * multiplyBy)
                        self.inputLayer[inputLabel + "" + val.__str__()].addWeight(-HALF * multiplyBy)

    def activate(self, values):
        for i, val in enumerate(values):
            if i in rowInputLabels:
                inputLabel = rowInputLabels[i]
                for key, value in rowInputTables[i].items():
                    self.inputLayer[inputLabel + "" + str(value)].doInputNeuron(val == value)
                    # print(val, value)
                    # print("firing " + (inputLabel + "" + str(value)))

        for label, neuron in self.hiddenLayer.items():
            neuron.activate()

        self.output.activate()
        return self.output.value

    def run(self, inputFilename, canLearn=False, diagnostics=False):
        inputReader = csv.reader(open(inputFilename), skipinitialspace=True, delimiter=',')

        oldRow = None

        for row in inputReader:
            if row:
                normal = normalize(row, training=canLearn)
                result = self.activate(normal)

                if canLearn is True:
                    if oldRow is not None:
                        self.learn(oldRow, normal, self.output.value > .5, diagnostics)
                    oldRow = normal

                print(result > 0.5 and 1 or 0)

        if diagnostics:
            # USEFUL OUTPUT
            print("total: " + str(self.total) + " correct: " + str(self.corrects) + " false negatives: " + str(
                self.false_negatives) + " false positives: " + str(self.false_positives) + "total positives: " + str(self.positives))
            print("error rate: " + str((self.total - self.corrects) / self.total))
            print("# Over 50K: " + str(self.count50))
            print("Correct # Over 50K: " + str(self.correctOver50))
            print("Over 50K correct ratio: " + str(self.correctOver50 / self.count50))

    def runSingle(self, rawInputRow, training=False):
        if rawInputRow:
            normal = normalize(rawInputRow, training=training)
            #print(rawInputRow, normal)
            result = self.activate(normal)
            print(result > 0.5 and 1 or 0)
            return result > 0.5
        else:
            return None

