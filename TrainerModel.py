import json

'''
Base class for trainers
'''

class TrainerModel(object):
	def __init__(self):
		pass

	def preprocess(self, l):
		raise NotImplementedError
	
	def group_labels(self, fname, field):
		f = open(fname)
		labels = {}
		for line in f:
			js = json.loads(line)
			try:
				labels[js[field]].append(js['votes']['useful'])
			except KeyError:
				labels[js[field]] = [js['votes']['useful']]
		return labels
	
	def build_examples(self, feats, labels):
		raise NotImplementedError
	
	def train(self):
		raise NotImplementedError
	
	def predict(self):
		raise NotImplementedError
	
