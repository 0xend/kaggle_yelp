clas TrainerModel(object):
	def preprocess(self):
		raise NotImplementedError
	def train(self):
		raise NotImplementedError
	def predict(self):
		raise NotImplementedError
