'''
Classifier based on the review content.
'''
import sys
from nltk.corpus import stopwords
from sklearn.feature_extraction.text import HashingVectorizer
from sklearn.cross_validation import KFold
from sklearn.linear_model import Ridge
from sklearn.feature_selection import SelectKBest, chi2, f_classif
import numpy as np

class ReviewTrainer(TrainerModel):

	def preprocess(self, l):
		sw = stopwords.words('english')
		clean = ' '.join([w for w in l['text'].split() if w not in sw])
		return {'text' : clean, 'votes' : l['votes']['useful']}
	
	def prepare_data(self, x, y):
		self.hv = HashingVectorizer(strip_accents='ascii', non_negative=True)
		self.feats = self.hv.transform(x)
		self.labels = np.array(y)
		
		self.ch2 = SelectKBest(chi2, k=15000)
		self.feats = self.ch2.fit_transform(self.feats, self.labels)
		
	def get_error(self, pred, y):
		dif = 0
		total = len(pred)
		for i in range(0,len(pred)):
			p = round(int(pred[i]))
			dif +=  abs(round(int(pred[i]))-int(y[i]))	
		return dif/total

	def _cross_validate(self):
		values = [0.001, 0.01, 0.1, 1, 10]
		best_score = sys.float_info.max
		best_clf = None
		kfold = KFold(n=len(self.labels), n_folds=10, indices=True)
		for v in values:
			clf = Ridge(alpha=v)
			values = []
			for train, test in kfold:
				clf.fit(self.feats[train], self.labels[train])
				pred = clf.predict(self.feats[test])
				values.append(self.get_error(pred, self.labels[test]))
			avg = sum(values)/len(values)
			if avg < best_score:
				best_clf = clf
				best_score = avg
		return best_clf

	def train(self):
		best = self._cross_validate()
		self.clf = Ridge(alpha=1)
		self.clf.fit(self.feats, self.labels)

	def predict(self, data):
		data = self.hv.transform(data)
		data = self.ch2.transform(data)
		pred = self.clf.predict(data)
		return pred			

