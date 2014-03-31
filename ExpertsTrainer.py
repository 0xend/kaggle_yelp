'''
Online learning: Learning with experts advice. 
Consists on a weighted majority. Initially, all weights
are equal for each model. After classifying each
example, weights are updated based on prediction 
(if right, weight stays the same; if wrong, at time t+1,
the ith model's weight is w_t+1,i = Bw_t,i, 0 < B < 1;
weights are normalized.)
'''
class ExpertsTrainer(object):
	def __init__(self, models):		
		pass

	def fit(self, examples):
		raise NotImplementedError
	
