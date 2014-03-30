'''
Predictor based on the review content.
'''

import numpy as np
from TrainerModel import TrainerModel
from nltk.corpus import stopwords
from sklearn.feature_extraction.text import HashingVectorizer
from sklearn.cross_validation import KFold
from sklearn.linear_model import Ridge
from sklearn.feature_selection import SelectKBest, chi2, f_classif

K_FEAT = 15000

class ReviewTrainer(TrainerModel):
	def __init__(self):
		pass

	def preprocess(self, l):
		res = {}
		sw = stopwords.words('english')
		clean = ' '.join([w for w in l['text'].split() if w not in sw])
		res[l['review_id']] = {'text' : clean, 'label' : l['votes']['useful']}
		return res

	def prepare_data(self, x, y):
		self.hv = HashingVectorizer(strip_accents='ascii', non_negative=True)
		self.feats = self.hv.transform(x)
		self.labels = np.array(y)
		
		self.ch2 = SelectKBest(chi2, k=K_FEAT)
		self.feats = self.ch2.fit_transform(self.feats, self.labels)
		
	def get_error(self, pred, y):
		return super(ReviewTrainer, self).get_error(pred,y)
	
	def _cross_validate(self):
		grid = dict(alpha=10.0 ** np.arange(-4,1))
		return super(ReviewTrainer, self)._cross_validate_base(
			Ridge(), grid) 
	
	def build_examples(self, data):
		print data
		feats = []
		labels = []
		ex = {}
		for k,v in data.items():
			feats.append(v['text'])
			labels.append(v['label'])
		ex['feats'] = feats
		ex['labels'] = labels
		return ex

	def train(self):
		self.clf = self._cross_validate()
		self.clf.fit(self.feats, self.labels)

	def predict(self, data):
		data = self.hv.transform(data)
		data = self.ch2.transform(data)
		pred = self.clf.predict(data)
		return pred			

