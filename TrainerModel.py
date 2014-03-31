'''
Base class for trainers
'''

import json
import sys
import constants as cons
from sklearn.cross_validation import KFold
from sklearn.grid_search import GridSearchCV
from sklearn.externals import joblib

class TrainerModel(object):
	def __init__(self):
		pass

	#implemented in child
	def preprocess(self, l):
		raise NotImplementedError
	
	#Use absolute dif between pred_i and y_i
	def get_error(self, pred, y):
		dif = 0
		total = len(pred)
		for i in range(0,len(pred)):
			p = round(int(pred[i]))
			dif +=  abs(round(int(pred[i]))-int(round(y[i])))	
		return dif/total
	
	#Use KFold to optimize hyper parameters.
	def _cross_validate_base(self, model, grid):
		cv = KFold(n=len(self.labels), n_folds=cons.N_FOLDS, indices=True)
		model = GridSearchCV(model, param_grid=grid, cv=cv)
		return model
	
	#implemented in child
	def _cross_validate(self, **extra):
		raise NotImplementedError
	
	#group labels for each example
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
	
	
	#implemented in child
	def build_examples(self, feats, labels):
		raise NotImplementedError


	#implemented in child
	def train(self):
		raise NotImplementedError
	
	#implemented in child
	def predict(self, data):
		raise NotImplementedError

	#saves the model to disk
	def save(self):
		name =self.__class__.__name__			
		_ = joblib.dump(self, 'models/%s.model' % name, compress=9)
