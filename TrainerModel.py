'''
Base class for trainers
'''

class TrainerModel(object):
	def __init__(self):
		pass

	def preprocess(self, l):
		raise NotImplementedError
	def build_example(self, feats, labels):
		raise NotImplementedError
	def train(self):
		raise NotImplementedError
	def predict(self):
		raise NotImplementedError
