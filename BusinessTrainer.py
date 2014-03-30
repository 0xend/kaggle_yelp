'''
Predictor based on biz.
'''
import numpy as np
from TrainerModel import TrainerModel
from sklearn import svm
from sklearn.cross_validation import KFold
from sklearn.feature_extraction import DictVectorizer
from sklearn.svm import SVR
import logging

logging.basicConfig(level=logging.INFO,
                    format='%(asctime)s %(levelname)s %(message)s')

class BusinessTrainer(TrainerModel):
	def __init__(self):
		pass
	
	def preprocess(self, l):
		res = {}
		biz_id = l['business_id']
		del(l['business_id'])
		del(l['type'])
		res[biz_id] = l
		return res

	def get_error(self, pred, y):
		return super(BusinessTrainer, self).get_error(pred,y)

	def _cross_validate(self, **extra):
		C_range = 10.0 ** np.arange(-1, 5)
		gamma_range = 10.0 ** np.arange(-3, 1)
		grid = dict(gamma=gamma_range, C=C_range)
		return super(BusinessTrainer, self)._cross_validate_base(
			SVR(), grid)

	def group_labels(self, fname):
		return super(BusinessTrainer, self).group_labels(fname, 'business_id')

	def build_examples(self, biz, revs):
		feats = []
		labels = []
		ex = {}
		for id, votes in revs.items():
			if id not in biz:
				continue
			X = biz[id]
			feat = {'city' : X['city'], 'state' : X['state'], 
				'count' : X['review_count'], 'open' : X['open'],
				'stars' : X['stars']}
			for cat in X['categories']:
				feat['cat-%s' % cat] = True
			feats.append(feat)	
			labels.append(sum(votes)/len(votes))
		ex['feats'] = feats
		ex['labels'] = labels
		return ex


	def train(self):	
		iself.clf = self._cross_validate()
		#self.clf = SVR()
		self.clf.fit(self.feats, self.labels)

	def prepare_data(self, x, y):
		self.dv = DictVectorizer()
		self.feats = self.dv.fit_transform(x)
		self.labels = np.array(y)
		

	def predict(self, data):
		data = self.dv.transform(data)
		pred = self.clf.predict(data)
		return pred			

