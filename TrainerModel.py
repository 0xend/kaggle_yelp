'''
Base class for trainers
'''

class TrainerModel(object):
	def __init__(self):
		pass
	
	def preprocess(self):
		raise NotImplementedError
	def train(self):
		raise NotImplementedError
	def predict(self):
		raise NotImplementedError
