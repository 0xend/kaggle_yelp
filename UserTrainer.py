'''
Predictor based on User.
'''
import numpy as np
from TrainerModel import TrainerModel
from sklearn import svm
from sklearn.ensemble import RandomForestRegressor
from sklearn.cross_validation import KFold
from sklearn.feature_extraction import DictVectorizer

class UserTrainer(TrainerModel):
	def __init__(self):
		pass

	#keeps useful parameters
	def preprocess(self, l):
		res = {}
		user_id = l['user_id']
		del(l['user_id'])
		del(l['type'])
		del(l['name'])
		res[user_id] = l
		return res

	def get_error(self, pred, y):
		return super(UserTrainer, self).get_error(pred,y)

	#optimizes for number of estimators in the forest
	def _cross_validate(self, **extra):
		grid = dict(n_estimators=[10, 50, 100])
		return super(UserTrainer, self)._cross_validate_base(
			RandomForestRegressor(), grid)

	def group_labels(self, fname):
		return super(UserTrainer, self).group_labels(fname, 'user_id')

	#builds examples for trainer
	def build_examples(self, user, revs):
		feats = []
		labels = []
		ex = {}
		for id, votes in revs.items():
			feat = {}
			if id not in user:
				continue
			feat['review_count'] = user[id]['review_count']
			feat['average_stars'] = user[id]['review_count']
			for kind, count in user[id]['votes'].items():
				feat['votes-%s' % kind] = count
			feats.append(feat)
			labels.append(sum(votes)/len(votes))
		ex['feats'] = feats
		ex['labels'] = labels
		print len(ex['labels'])
		return ex

	#fits model using optimal hyper-parameters
	def train(self):	
		self.clf = self._cross_validate()
		self.clf.fit(self.feats, self.labels)

	#vectorizes data
	def prepare_data(self, x, y):
		self.dv = DictVectorizer(sparse=False)
		self.feats = self.dv.fit_transform(x)
		self.labels = np.array(y)
		

	#predicts Y given X
	def predict(self, data):
		data = self.dv.transform(data)
		pred = self.clf.predict(data)
		return pred			


