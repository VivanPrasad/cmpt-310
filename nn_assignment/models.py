import math
import nn


import util
###########################################################################
class NaiveBayesDigitClassificationModel(object):

    def __init__(self):
        self.conditionalProb = None
        self.prior = None
        self.features = None
        self.k = 1 # this is the smoothing parameter, ** use it in your train method **
        self.automaticTuning = True # Look at this flag to decide whether to choose k automatically ** use this in your train method **
        self.legalLabels = range(10)

    def train(self, dataset):
        # this is a list of all features in the training set.
        self.features = list(set([f for datum in dataset.trainingData for f in datum.keys()]))

        kgrid = [0.00005, 0.0001, 0.0005, 0.001, 0.005, 0.01, 0.02, 0.05, 0.5, 1, 5]
        self.trainAndTune(dataset, kgrid)

    def trainAndTune(self, dataset, kgrid):
        """
        Trains the classifier by collecting counts over the training data, and
        stores the Laplace smoothed estimates so that they can be used to classify.
        Evaluate each value of k in kgrid to choose the smoothing parameter
        that gives the best accuracy on the held-out validationData.

        trainingData and validationData are lists of feature Counters. The corresponding
        label lists contain the correct label for each datum.

        To get the list of all possible features or labels, use self.features and
        self.legalLabels.
        """
        trainingData = dataset.trainingData
        trainingLabels = dataset.trainingLabels
        validationData = dataset.validationData
        validationLabels = dataset.validationLabels

        bestAccuracyCount = -1  # best accuracy so far on validation set
        #Common training - get all counts from training data
        #We only do it once - save computation in tuning smoothing parameter
        commonPrior = util.Counter()  # Prior probability over labels
        commonConditionalProb = util.Counter()  #Conditional probability of feature feat being 1 indexed by (feat, label)
        commonCounts = util.Counter()  #how many time I have seen feature 'feat' with label 'y' whether inactive or active
        bestParams = (commonPrior, commonConditionalProb, kgrid[0])  # used for smoothing part  trying various Laplace factors kgrid

        for i in range(len(trainingData)):
            datum = trainingData[i]
            label = int(trainingLabels[i])
            "*** YOUR CODE HERE to complete populating commonPrior, commonCounts, and commonConditionalProb ***"
            commonPrior[label] += 1
            for feat in self.features:
                if datum[feat] == 1:
                    commonCounts[feat, label] += 1
                else:
                    commonCounts[feat, label] += 0
            for feat in self.features:
                if datum[feat] == 1:
                    commonConditionalProb[feat, label] += 1
                else:
                    commonConditionalProb[feat, label] += 0
            # end of populating commonPrior, commonCounts, and commonConditionalProb
            #util.raiseNotDefined()

        for k in kgrid:  # smoothing parameter tuning loop
            prior = util.Counter()
            conditionalProb = util.Counter()
            counts = util.Counter()

            # get counts from common training step
            for key, val in commonPrior.items():
                prior[key] += val
            for key, val in commonCounts.items():
                counts[key] += val
            for key, val in commonConditionalProb.items():
                conditionalProb[key] += val

            # smoothing:
            for label in self.legalLabels:
                for feat in self.features:
                    # Laplace smoothing
                    conditionalProb[feat, label] += k
                    prior[label] += k * len(self.features)

            # normalising:
            prior.normalize()
            # Normalize conditionalProb
            for label in self.legalLabels:
                total = sum(conditionalProb[feat, label] for feat in self.features)
                for feat in self.features:
                    conditionalProb[feat, label] /= total
            # end the normalisingg

            self.prior = prior
            self.conditionalProb = conditionalProb

            # evaluating performance on validation
            predictions = self.classify(validationData)
            accuracyCount = [predictions[i] == validationLabels[i] for i in range(len(validationLabels))].count(True)

            print("Performance on validation set for k=%f: (%.1f%%)" % (
            k, 100.0 * accuracyCount / len(validationLabels)))
            if accuracyCount > bestAccuracyCount:
                bestParams = (prior, conditionalProb, k)
                bestAccuracyCount = accuracyCount
            # end of automatic tuning loop

        self.prior, self.conditionalProb, self.k = bestParams
        print("Best Performance on validation set for k=%f: (%.1f%%)" % (
            self.k, 100.0 * bestAccuracyCount / len(validationLabels)))


    def classify(self, testData):
        """
        Classify the data based on the posterior distribution over labels.
        You shouldn't modify this method.
        """
        guesses = []
        self.posteriors = [] # Log posteriors are stored for later data analysis
        for datum in testData:
            ("***YOUR CODE HERE***  use calculateLogJointProbabilities() to compute posterior per datum  and use"
             "it to find best guess digit for datum and at the end accumulate in self.posteriors for later use")
            logJoint = self.calculateLogJointProbabilities(datum)
            bestLabel = None
            bestProb = float("-inf")
            for label in self.legalLabels:
                if logJoint[label] > bestProb:
                    bestProb = logJoint[label]
                    bestLabel = label
            guesses.append(bestLabel)
            self.posteriors.append(logJoint)
            #util.raiseNotDefined()

        return guesses

    def calculateLogJointProbabilities(self, datum):
        """
        Returns the log-joint distribution over legal labels and the datum.
        Each log-probability should be stored in the log-joint counter, e.g.
        logJoint[3] = <Estimate of log( P(Label = 3, datum) )>

        To get the list of all possible features or labels, use self.features and
        self.legalLabels.
        """
        logJoint = util.Counter()

        for label in self.legalLabels:
            "*** YOUR CODE HERE, to populate logJoint() list ***"
            # Calculate the log joint probability
            logJoint[label] = math.log(self.prior[label])
            for feat in self.features:
                if datum[feat] == 1:
                    logJoint[label] += math.log(self.conditionalProb[feat, label])
                else:
                    logJoint[label] += math.log(1 - self.conditionalProb[feat, label])
            # end of populating logJoint
            #util.raiseNotDefined()
        return logJoint

################################################################################3
class PerceptronModel(object):
    def __init__(self, dimensions):
        """
        Initialize a new Perceptron instance.

        A perceptron classifies data points as either belonging to a particular
        class (+1) or not (-1). `dimensions` is the dimensionality of the data.
        For example, dimensions=2 would mean that the perceptron must classify
        2D points.
        """
        self.w = nn.Parameter(1, dimensions)

    def get_weights(self):
        """
        Return a Parameter instance with the current weights of the perceptron.
        """
        return self.w

    def run(self, x):
        """
        Calculates the score assigned by the perceptron to a data point x.

        Inputs:
            x: a node with shape (1 x dimensions)
        Returns: a node containing a single number (the score)
        """
        "*** YOUR CODE HERE ***"
        return nn.DotProduct(self.w, x)

    def get_prediction(self, x):
        """
        Calculates the predicted class for a single data point `x`.

        Returns: 1 or -1
        """
        "*** YOUR CODE HERE ***"
        score = nn.as_scalar(self.run(x)) #single number
        return 1 if score >= 0 else -1

    def train(self, dataset):
        """
        Train the perceptron until convergence.
        """
        "*** YOUR CODE HERE ***"
        # Loop until convergence
        while True:
            miss = False
            for x, y in dataset.iterate_once(1):
                prediction = self.get_prediction(x)
                if prediction != nn.as_scalar(y):
                    miss = True
                    self.w.update(x, nn.as_scalar(y))
            if not miss:
                break

########################################################################33
class RegressionModel(object):
    """
    A neural network model for approximating a function that maps from real
    numbers to real numbers. The network should be sufficiently large to be able
    to approximate sin(x) on the interval [-2pi, 2pi] to reasonable precision.
    """
    def __init__(self):
        # Initialize your model parameters here
        self.batch_size = 100  # Increase batch size for faster convergence
        self.learning_rate = 0.05  # Higher learning rate for quicker updates
        self.hidden_size = 128  # Larger hidden size for better representation
        self.input_size = 1
        self.output_size = 1
        
        self.w1 = nn.Parameter(self.input_size, self.hidden_size)
        self.w2 = nn.Parameter(self.hidden_size, self.output_size)
        self.b1 = nn.Parameter(1, self.hidden_size)
        self.b2 = nn.Parameter(1, self.output_size)

    def run(self, x):
        """
        Runs the model for a batch of examples.
        Inputs:
            x: a node with shape (batch_size x 1)
        Returns:
            A node with shape (batch_size x 1) containing predicted y-values
        """
        # Forward pass through the network
        hidden = nn.ReLU(nn.AddBias(nn.Linear(x, self.w1), self.b1))
        output = nn.AddBias(nn.Linear(hidden, self.w2), self.b2)
        return output

    def get_loss(self, x, y):
        """
        Computes the loss for a batch of examples.
        Inputs:
            x: a node with shape (batch_size x 1)
            y: a node with shape (batch_size x 1), containing the true y-values
                to be used for training
        Returns: a loss node
        """
        # Forward pass to compute predictions
        predictions = self.run(x)
        # Compute the Mean Squared Error Loss
        loss = nn.SquareLoss(predictions, y)
        return loss

    def train(self, dataset):
        """
        Trains the model.
        """
        while True:
            for x, y in dataset.iterate_once(self.batch_size):
                loss = self.get_loss(x, y) # calculate
                # Compute the gradients
                gradients = nn.gradients(loss, [self.w1, self.w2, self.b1, self.b2])
                # Update the weights
                self.w1.update(gradients[0], -self.learning_rate)
                self.w2.update(gradients[1], -self.learning_rate)
                self.b1.update(gradients[2], -self.learning_rate)
                self.b2.update(gradients[3], -self.learning_rate)
            if nn.as_scalar(self.get_loss(nn.Constant(dataset.x), nn.Constant(dataset.y))) < 0.02: break
            #break out with loss < 0.02

##########################################################################
class DigitClassificationModel(object):
    """
    A second model for handwritten digit classification using the MNIST dataset.

    Each handwritten digit is a 28x28 pixel grayscale image, which is flattened
    into a 784-dimensional vector for the purposes of this model. Each entry in
    the vector is a floating point number between 0 and 1.

    The goal is to classify each digit into one of 10 classes (number 0 through 9).

    (See RegressionModel for more information about the APIs of different
    methods here. We recommend that you implement the RegressionModel before
    working on this part of the project.)
    """
    def __init__(self):
        # Initialize your model parameters here
        self.batch_size = 100
        self.learning_rate = 0.1  # Increased learning rate for faster convergence
        self.hidden_size = 256  # Increased hidden size for better representation
        self.input_size = 784
        self.output_size = 10
        self.w1 = nn.Parameter(self.input_size, self.hidden_size)
        self.w2 = nn.Parameter(self.hidden_size, self.output_size)
        self.b1 = nn.Parameter(1, self.hidden_size)
        self.b2 = nn.Parameter(1, self.output_size)

    def run(self, x):
        """
        Runs the model for a batch of examples.

        Your model should predict a node with shape (batch_size x 10),
        containing scores. Higher scores correspond to greater probability of
        the image belonging to a particular class.

        Inputs:
            x: a node with shape (batch_size x 784)
        Output:
            A node with shape (batch_size x 10) containing predicted scores
                (also called logits)
        """
        # Forward pass through the network
        hidden = nn.ReLU(nn.AddBias(nn.Linear(x, self.w1), self.b1))
        output = nn.AddBias(nn.Linear(hidden, self.w2), self.b2)
        return output

    def get_loss(self, x, y):
        """
        Computes the loss for a batch of examples.

        The correct labels `y` are represented as a node with shape
        (batch_size x 10). Each row is a one-hot vector encoding the correct
        digit class (0-9).

        Inputs:
            x: a node with shape (batch_size x 784)
            y: a node with shape (batch_size x 10)
        Returns: a loss node
        """
        # Forward pass to compute predictions
        predictions = self.run(x)
        # Compute the Softmax Loss
        loss = nn.SoftmaxLoss(predictions, y)
        return loss

    def train(self, dataset):
        """
        Trains the model.
        """
        # Loop for a fixed number of epochs
        while True:
            for x, y in dataset.iterate_once(self.batch_size):
                # Compute the loss
                loss = self.get_loss(x, y)
                # Compute the gradients
                gradients = nn.gradients(loss, [self.w1, self.w2, self.b1, self.b2])
                # Update the weights
                self.w1.update(gradients[0], -self.learning_rate)
                self.w2.update(gradients[1], -self.learning_rate)
                self.b1.update(gradients[2], -self.learning_rate)
                self.b2.update(gradients[3], -self.learning_rate)
            # Check the validation accuracy
            if dataset.get_validation_accuracy() >= 0.975: break

###################################################################################
class LanguageIDModel(object):
    """
    A model for language identification at a single-word granularity.

    (See RegressionModel for more information about the APIs of different
    methods here. We recommend that you implement the RegressionModel before
    working on this part of the project.)
    """
    def __init__(self):
        # Our dataset contains words from five different languages, and the
        # combined alphabets of the five languages contain a total of 47 unique
        # characters.
        # You can refer to self.num_chars or len(self.languages) in your code
        self.num_chars = 47
        self.languages = ["English", "Spanish", "Finnish", "Dutch", "Polish"]

        # Initialize your model parameters here
        "*** YOUR CODE HERE ***"
        self.batch_size = 100  # Increase batch size for faster convergence
        self.learning_rate = 0.06  # Higher learning rate for quicker updates
        self.hidden_size = 128  # Larger hidden size for better representation
        self.input_size = 47
        self.output_size = 5
        
        self.w1 = nn.Parameter(self.input_size, self.hidden_size)
        self.w2 = nn.Parameter(self.hidden_size, self.output_size)
        self.b1 = nn.Parameter(1, self.hidden_size)
        self.b2 = nn.Parameter(1, self.output_size)


    def run(self, xs):
        """
        Runs the model for a batch of examples.

        Although words have different lengths, our data processing guarantees
        that within a single batch, all words will be of the same length (L).

        Here `xs` will be a list of length L. Each element of `xs` will be a
        node with shape (batch_size x self.num_chars), where every row in the
        array is a one-hot vector encoding of a character. For example, if we
        have a batch of 8 three-letter words where the last word is "cat", then
        xs[1] will be a node that contains a 1 at position (7, 0). Here the
        index 7 reflects the fact that "cat" is the last word in the batch, and
        the index 0 reflects the fact that the letter "a" is the initial (0th)
        letter of our combined alphabet for this task.

        Your model should use a Recurrent Neural Network to summarize the list
        `xs` into a single node of shape (batch_size x hidden_size), for your
        choice of hidden_size. It should then calculate a node of shape
        (batch_size x 5) containing scores, where higher scores correspond to
        greater probability of the word originating from a particular language.

        Inputs:
            xs: a list with L elements (one per character), where each element
                is a node with shape (batch_size x self.num_chars)
        Returns:
            A node with shape (batch_size x 5) containing predicted scores
                (also called logits)
        """
        # Initialize the hidden state
        h = nn.ReLU(nn.AddBias(nn.Linear(xs[0], self.w1), self.b1))
        
        # Process each character in the sequence
        for x in xs[1:]:
            h = nn.ReLU(nn.Add(nn.AddBias(nn.Linear(x, self.w1), self.b1),h))
        
        # Compute the output logits
        output = nn.AddBias(nn.Linear(h, self.w2), self.b2)
        return output
        
        
    def get_loss(self, xs, y):
        """
        Computes the loss for a batch of examples.

        The correct labels `y` are represented as a node with shape
        (batch_size x 5). Each row is a one-hot vector encoding the correct
        language.

        Inputs:
            xs: a list with L elements (one per character), where each element
                is a node with shape (batch_size x self.num_chars)
            y: a node with shape (batch_size x 5)
        Returns: a loss node
        """
        "*** YOUR CODE HERE ***"
        # Forward pass to compute predictions
        predictions = self.run(xs)
        # Compute the Softmax Loss
        loss = nn.SoftmaxLoss(predictions, y)
        return loss
        

    def train(self, dataset):
        """
        Trains the model.
        """
        "*** YOUR CODE HERE ***"
        "*** hint: User get_validation_accuracy() to decide when to finish learning ***"
        while True:
            for xs, y in dataset.iterate_once(self.batch_size):
                loss = self.get_loss(xs, y)
                gradients = nn.gradients(loss, [self.w1, self.w2, self.b1, self.b2])
                self.w1.update(gradients[0], -self.learning_rate)
                self.w2.update(gradients[1], -self.learning_rate)
                self.b1.update(gradients[2], -self.learning_rate)
                self.b2.update(gradients[3], -self.learning_rate)
            if dataset.get_validation_accuracy() > 0.83: break