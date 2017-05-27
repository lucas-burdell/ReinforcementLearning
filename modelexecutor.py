from model import Model
from sys import argv
script, modelfilename, datasetfilename = argv
model = Model(modelFilename=modelfilename)
model.run(datasetfilename, canLearn=False, diagnostics=False)
