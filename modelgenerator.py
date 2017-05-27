from sys import argv
from model import Model

script, inputfilename = argv
model = Model()
model.run(inputfilename, canLearn=True, diagnostics=False)
model.save()