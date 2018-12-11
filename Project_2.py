import numpy as np
from operator import itemgetter
from collections import Counter
import time
import os

"""
Function for driving program
"""
def startProject():

    description1  = "Welcome to Carlos Gomez's Feature Selection Algorithm.\n"
    description1 += 'Type in the name of the file to test:'
    print(description1, end=' ')
    data_string = str(input())
    data_string_path = os.path.join(os.getcwd(), data_string)
    data = np.genfromtxt(data_string_path)

    alg_description  = "Type the number of the algorithm you want to run.\n"
    alg_description += "\t1)  Forward Selection\n"
    alg_description += "\t2)  Backward Elimination\n"
    alg_description += "\t3)  Bertie's Special Algorithm.\n"

    print(alg_description)
    
    alg_input = int(input())

    beg_prompt_1 = 'This dataset has {} features (not including the class attribute), with {} instances\n'.format(data.shape[1]-1, data.shape[0])
    print(beg_prompt_1)

    beg_prompt_2 = 'Running nearest neighbor with all {} features, using \"leaving-one-out\" evalutation, I get an accuracy of {}%\n'.format(data.shape[1]-1, leave_one_evaluation(data))
    print(beg_prompt_2)

    print('Beginning search\n')


    input_dict = {1:forward_selection, 2:backward_elimination, 3:special_algorithm}
    
    start = time.time()
    input_dict[alg_input](data)
    end = time.time()
    print("Time to finish: {}".format(end-start))


"""
Function for leave one out evaluation
"""
def leave_one_evaluation(data):
    predicted_values = []
    data_classes = [entry[0] for entry in data]
    for index in range(len(data)):
        test_data = data[index]
        test_data = test_data[1:]
        training_data = []
        if index == 0:
            training_data = data[1:]
        elif index == (len(data)-1):
            training_data = data[:index]
        else:
            training_data = np.concatenate((data[:index],data[index+1:]), axis=0)
        training_classifications = [i[0] for i in training_data]
        training_data = np.delete(training_data, 0, axis=1)
        predicted = nn_classifier(test_data, training_data, training_classifications, 2)
        predicted_values.append(predicted)
    
    accuracy = testAccuracy(predicted_values, data_classes)
    return accuracy
    


"""
Function to find distance between two vectors
"""
def LP_distance(x,y,p):
    totalDistance = 0
    for i,j in zip(x,y):
        newDistance = (abs(i-j)**p)
        totalDistance += newDistance
    return (totalDistance**(1/p))


"""
Function gets nearest neighbor for one tuple
"""
def get_k_neighbors(trainingData, testInstance, training_classifications, p):
    k = 1
    allDistances = []
    trainCnt = 0
    for dataEntry in trainingData:
        newDistance = LP_distance(dataEntry, testInstance, p)
        allDistances.append((newDistance, training_classifications[trainCnt]))
        trainCnt += 1
    allDistances = sorted(allDistances, key=itemgetter(0))
    
    kDistances = [val[1] for val in allDistances[:k]]
    return kDistances
    
"""
Function that implements Nearest Neighbor classifier 
"""
def nn_classifier(test_set, training_set, training_classifications, p):
    class_predictions = []
        
    neighbors = get_k_neighbors(training_set, test_set, training_classifications, p)
    countClass = list(Counter(neighbors).keys())
    class_predictions.append(countClass[0])
    
    return class_predictions

"""
Function to test accuracty of nearest neighbor classifier
"""
def testAccuracy(class_predictions, class_data):
    sameCount = 0
    wrong_count = 0
    for i,j in zip(class_predictions, class_data):
        if i == j:
            sameCount += 1
        """
        else:
            wrong_count += 1
            if wrong_count > 11:
                return 0
        """
    accuracy = (sameCount/len(class_data)) * 100
    return accuracy

"""
Function for forward selection feature search
"""
def forward_selection(data):
    current_features = []
    num_features = data.shape[1]
    best_accuracy = 0

    for i in range(1,num_features):
        #print("On the {}th level of the search tree".format(i))
        feature_add = []

        for k in range(1, num_features):
            if k in current_features:
                continue
            test_features = [0] + current_features + [k]
            accuracy = leave_one_evaluation(data[:,test_features])
            print("\t\tUsing feature(s) {features} accuracy is {accuracy}%".format(features=current_features+[k], accuracy=accuracy))

            if accuracy > best_accuracy:
                best_accuracy = accuracy
                feature_add = k

        if feature_add:
            current_features.append(feature_add)
        print("\nFeature {} was best, accuracy is {}\n".format(feature_add, best_accuracy))
    print("Finished search!! The best feature subset is {}, which has an accuracy of {}".format(current_features, best_accuracy))

    #return current_features
"""
Function for backward elimination feature search
"""
def backward_elimination(data):
    pass

"""
Function for special algorithm feature search
"""
def special_algorithm(data):
    pass

"""
Function for backward elimination feature search
"""

if __name__ == "__main__":

    startProject()
    