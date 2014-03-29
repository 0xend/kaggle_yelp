'''
Predictor based on biz.
'''
import numpy as np
from TrainerModel import TrainerModel
from sklearn import svm
from sklearn.cross_validation import KFold
from sklearn.feature_extraction.text import HashingVectorizer

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
		return super(ReviewTrainer, self).get_error(pred,y)

	def _cross_validate(self, **extra):
		args = extra.items()
		values = [0.0001, 0.001, 0.01, 0.1, 1]
		return super(BusinessTrainer, self)._cross_validate_base(
			SVC, args, 'C', values)

	def group_labels(self, fname):
		return super(BusinessTrainer, self).group_labels(fname, 'business_id')

	def build_examples(self, biz, revs):
		feats = []
		labels = []
		ex = {}
		for id, votes in revs.items():
			X = biz[id]
			feats.append([
				X['city'], X['state'],
				X['review_count'], X['open']])
			labels.append(votes)
		ex['feats'] = feats
		ex['labels'] = labels
		return ex


	def train(self):
		print 'LINEAR'
		linear, linear_score = self._cross_validate(kernel='linear')	
		print 'RBF'
		rbf, rbf_score = self._cross_validate(kernel='rbf')
		print 'Poly'
		poly, poly_score = self._cross_validate(kernel='poly')
		clfs = [linear, rbf, poly]
		scores = [linear_score, rbf_score, poly_score]
		max_index = scores.index(max(scores))
		self.clf = clfs[max_index]

	def prepare_data(self, x, y):
		self.hv = HashingVectorizer(non_negative=True)
		self.feats = self.hv.transform(x)
		self.labels = np.array(y)
		

	def predict(self, data):
		pass
