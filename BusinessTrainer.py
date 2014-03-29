from TrainerModel import TrainerModel

class BusinessTrainer(TrainerModel):
	def __init__(self):
		pass
	
	def preprocess(self, l):
		res = {}
		biz_id = l['business_id']
		del(l['business_id'])
		res[biz_id] = l
		return res

	def _cross_validate(self):
		pass
	
	def group_labels(self, fname):
		return super(BusinessTrainer, self).group_labels(fname, 'business_id')

	def build_examples(self, biz, revs):
		feats = []
		labels = []
		ex = {}
		for id, votes in revs.items():
			feats.append(biz[id])
			labels.append(votes)
		ex['feats'] = feats
		ex['labels'] = labels
		return ex


	def train(self):
		pass
	
	def predict(self, data):
		pass
