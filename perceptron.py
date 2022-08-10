class Perceptron:
	"""
	A perceptron is a simple supervised learning algorithm
	that performs binary classification and is the basis
	for neural networks, serving as an artificial neuron.
	"""

	def __init__(self, weights = None):
		"""Initialize the perceptron with given weights (default is None)"""
		self.weights = weights

	# Problem 1

	def predict1(self, x):
		"""
		Predict a single input/output from the perceptron model.
		param x: A list or tuple giving the input vector
		returns: The predicted output (0 or 1), or None if not trained yet
		"""
		if self.weights is None:
			print("model has not been trained yet")
		else:
			X = [1] + list(x)
			activation = sum([i*j for (i, j) in zip(self.weights, X)])
			if activation > 0:
				return 1
			return 0
		pass

	# Problem 2

	def predict(self, X):
		"""
		Predict a list of input/outputs from the perceptron model.
		param X: An iterable containing lists/tuples of input vectors
		returns: A list containing the predicted outputs (each 0 or 1)
		"""
		return [self.predict1(x) for x in X]

	# Problem 3

	def update(self, x, y):
		"""
		Updates the perceptron weights from a single sample input/output pair.
		param x: A list or tuple giving the input vector
		param y: The observed output (0 or 1)
		returns: None
		"""
		X = [1] + list(x)
		if self.weights == None:
			self.weights = [0] * (len(list(x)) + 1)
		else:
			pass

		pred = self.predict1(x)
		if pred != y:
			correction = y-pred
			weights = []
			for i in range(len(self.weights)):
				weights.append(self.weights[i] + correction * X[i])
			self.weights = weights
		else:
			pass
		return self.weights

	# Problem 4

	def fit(self, X, Y, num_iter = 100):
		"""
		Updates the perceptron weights from a list of sample input/output pairs.
		param X: An iterable containing lists/tuples of input vectors
		param Y: An iterable containing the observed outputs (each 0 or 1)
		param num_iter: The number of training iterations over all samples
		returns: None
		"""
		i = 0
		while(i <= num_iter):
			for j in range(len(X)):
				self.update(X[j], Y[j])
			i += 1
		

	# Problem 5

	def score(self, X, Y):
		"""
		Calculates the prediction accuracy for a list of sample input/output pairs.
		param X: An iterable containing lists/tuples of input vectors
		param Y: An iterable containing the observed outputs (each 0 or 1)
		returns: The predictive accuracy (proportion of correct predictions)
		"""
		preds = self.predict(X)
		correct_preds = []
		for i in range(len(Y)):
			if preds[i] == Y[i]:
				correct_preds.append(preds[i])
		accuracy = len(correct_preds)/len(Y)
		return accuracy

### Tested by dummy data. Add/ import your own dataset to test or use "sonar dataset" to in file testing :). Enjoy the results.
### AI out beep boop beep 
# if __name__ == "__main__":
# 	X = [(-1, -1),
# 		(-5, -2.5),
# 		(-7.5, -7.5),
# 		(10, 7.5),
# 		(-2.5, 12.5),
# 		(5, 10),
# 		(5, 5)]
# 	Y = [0, 0, 0, 1, 0, 1, 1]
# 	model = Perceptron(weights=[-5, 1, 1])
# 	print(model.predict1(X[0])) # correct
# 	print(model.predict1(X[1])) # correct
# 	print(model.predict1(X[2])) # correct
# 	print(model.predict1(X[3])) # correct
# 	print(model.predict1(X[4])) # incorrect!
# 	print(Y)
# 	print(model.predict(X)) # all correct but X[4]
# 	print(model.score(X, Y)) # decent performance
# 	model.update(X[4], Y[4])
# 	print(model.weights)
# 	print(model.predict1(X[4])) # correct!
# 	print(model.predict(X))
# 	print(model.score(X, Y)) # poor overall though...
# 	model = Perceptron()
# 	model.fit(X, Y, num_iter=100)
# 	print(model.weights)
# 	print(model.predict(X)) # everything correct!
# 	print(model.score(X, Y)) # perfectly linearly separable
# 	X = []
# 	Y = []
# 	with open("sonar.csv") as f: # import sonar CSV dataset
# 		lines = f.readlines()
# 		for line in lines[1:]:
# 			csv = line.split(",")
# 			X.append([float(s) for s in csv[:60]])
# 			Y.append(1 if "Mine" in csv[60] else 0)
# 	model = Perceptron()
# 	print("fitting sonar data with 100 iterations...")
# 	model.fit(X, Y, num_iter=100) # accuracy is ~53% -- baseline
# 	print(model.score(X, Y))
# 	print("fitting sonar data with 1000 iterations...")
# 	model.fit(X, Y, num_iter=1000)  # accuracy is ~56% -- slightly better
# 	print(model.score(X, Y))
# 	print("fitting sonar data with 5000 iterations...")
# 	model.fit(X, Y, num_iter=5000) # accuracy is ~66% -- not terrible
# 	print(model.score(X, Y))
# 	print("fitting sonar data with 10000 iterations...")
# 	model.fit(X, Y, num_iter=10000) # accuracy is ~70% -- decent?
# 	print(model.score(X, Y))

