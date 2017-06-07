#!/usr/bin/env python
# -*- coding: utf-8 -*-

from data.mnist_seven import MNISTSeven
from model.stupid_recognizer import StupidRecognizer
from model.perceptron import Perceptron
from model.logistic_regression import LogisticRegression
from report.evaluator import Evaluator
import matplotlib.pyplot as plt

def main():
    data = MNISTSeven("../data/mnist_seven.csv", 3000, 1000, 1000)

    learningRates = (0.0005, 0.0006, 0.0007, 0.0008)
    logisticRegressionClassifier1 = LogisticRegression(data.trainingSet,
                                                      data.validationSet,
                                                      data.testSet,
                                                      learningRate=learningRates[0],
                                                      epochs=200)
    logisticRegressionClassifier2 = LogisticRegression(data.trainingSet,
                                                      data.validationSet,
                                                      data.testSet,
                                                      learningRate=learningRates[1],
                                                      epochs=200)
    logisticRegressionClassifier3 = LogisticRegression(data.trainingSet,
                                                      data.validationSet,
                                                      data.testSet,
                                                      learningRate=learningRates[2],
                                                      epochs=200)
    logisticRegressionClassifier4 = LogisticRegression(data.trainingSet,
                                                      data.validationSet,
                                                      data.testSet,
                                                      learningRate=learningRates[3],
                                                      epochs=200)


    # Train the classifiers
    print("=========================")
    print("Training..")

    print("\nLogistic Regression has been training..")
    errors1 = logisticRegressionClassifier1.train()
    errors2 = logisticRegressionClassifier2.train()
    errors3 = logisticRegressionClassifier3.train()
    errors4 = logisticRegressionClassifier4.train()
    print("Done..")



    # Do the recognizer
    # Explicitly specify the test set to be evaluated
    #stupidPred = myStupidClassifier.evaluate()
    #perceptronPred = myPerceptronClassifier.evaluate()
    logRegPred1 = logisticRegressionClassifier1.evaluate()
    logRegPred2 = logisticRegressionClassifier2.evaluate()
    logRegPred3 = logisticRegressionClassifier3.evaluate()
    logRegPred4 = logisticRegressionClassifier4.evaluate()

    # Report the result
    print("=========================")
    evaluator = Evaluator()

    print("Result of the Logistic Regression:")
    #evaluator.printComparison(data.testSet, logRegPred)
    evaluator.printAccuracy(data.testSet, logRegPred1)
    evaluator.printAccuracy(data.testSet, logRegPred2)
    evaluator.printAccuracy(data.testSet, logRegPred3)
    evaluator.printAccuracy(data.testSet, logRegPred4)

    plt.plot([i for i in range(200)], errors1)
    plt.plot([i for i in range(200)], errors2)
    plt.plot([i for i in range(200)], errors3)
    plt.plot([i for i in range(200)], errors4)
    plt.ylim([-10, 2])
    plt.axhline(y=0, color='k')

    plt.show()
    
    
if __name__ == '__main__':
    main()
