import sys
import numpy as np


# ********************************************************************************
# ****************************** FILE OPERATIONS *********************************
# ********************************************************************************
# Get the data from the file.
def get_data(data_file_name):
    # Open the file and split by newlines.
    with open(data_file_name) as data_file:
        all_data = data_file.readlines()
        all_data = [line.strip() for line in all_data]
    return all_data


# ********************************************************************************
# ****************************** Accuracy Check **********************************
# ********************************************************************************
# Test the accuracy of all the algorithms.
# Test the Perceptron.
def test_perceptron(x_train, y_train, w_perceptron):
    m = len(x_train)
    m_perceptron = 0
    w_perceptron = w_perceptron[0:w_perceptron.size - 1]
    for t in range(0, m):
        y_hat = np.argmax(np.dot(w_perceptron, x_train[t]))
        if y_train[t] != y_hat:
            m_perceptron += 1
    print("perceptron err =", float(m_perceptron)/m)


# Test the SVM.
def test_svm(x_train, y_train, w_svm):
    m = len(x_train)
    m_svm = 0
    w_svm = w_svm[0:w_svm.size - 1]
    for t in range(0, m):
        y_hat = np.argmax(np.dot(w_svm, x_train[t]))
        if y_train[t] != y_hat:
            m_svm += 1
    print("svm err =", float(m_svm)/m)


# Test the PA.
def test_pa(x_train, y_train, w_pa):
    m = len(x_train)
    m_pa = 0
    w_pa = w_pa[0:w_pa.size - 1]
    for t in range(0, m):
        y_hat = np.argmax(np.dot(w_pa, x_train[t]))
        if y_train[t] != y_hat:
            m_pa += 1
    print("pa err =", float(m_pa)/m)


# ********************************************************************************
# ****************************** NORMALIZATION ***********************************
# ********************************************************************************
# Min-Max normalization.
def normalize_min_max(input_array):
    # Convert the array to a numpy.
    normalized = np.array(input_array)
    # Calculate the denominator.
    denominator = (normalized[:, 1].max() - normalized[:, 1].min())
    # Handle case of zero-division error.
    if denominator == 0:
        return normalized
    # Normalize.
    normalized[:, 1] = (normalized[:, 1] - normalized[:, 1].min()) / denominator
    return normalized


# ********************************************************************************
# ****************************** PERCEPTRON **************************************
# ********************************************************************************
class Perceptron(object):
    def __init__(self):
        # The learning rate of the algorithm.
        self.learning_rate = 0.01
        # Epochs.
        self.thresh_hold = 100
        # The weight array, of size 3 and 8 because we have 3 classes and 8 features.
        self.arr_of_weights = np.zeros((3, 8))

    # Train the Perceptron algorithm using the training data.
    def train_func(self, training_inputs, labels):
        for t in range(self.thresh_hold):
            for x, y in zip(training_inputs, labels):
                # Get the parameters from the input.
                y = int(float(y))
                x = np.array(x).astype(float)
                new_y = int(np.argmax(np.dot(self.arr_of_weights, x)))
                # Update.
                if y != new_y:
                    eta = self.learning_rate * x
                    self.arr_of_weights[y] += eta
                    self.arr_of_weights[new_y] -= eta

    # Predict the classification using the input data vector and the calculated weights.
    def predict_func(self, data1_of_inputs):
        x = np.array(data1_of_inputs).astype(float)
        # Predict.
        prediction = np.argmax(np.dot(self.arr_of_weights, x))
        return prediction


# ********************************************************************************
# ****************************** SVM - (Support Vector Machine) ******************
# ********************************************************************************
class SVM(object):
    def __init__(self):
        # The learning rate.
        self.learning_rate = 0.01
        # Epochs.
        self.thresh_hold = 400
        # The weight array, of size 3 and 8 because we have 3 classes and 8 features.
        self.arr_of_weights = np.zeros((3, 8))
        # The lambda, 1/epochs therefore the regularization number reduces as the number of epochs increases.
        self.lambada = 1/self.thresh_hold

    # Train the SVM algorithm using the training data.
    def train_func(self, training_inputs, labels):
        for t in range(self.thresh_hold):
            for x, y in zip(training_inputs, labels):
                y = int(float(y))
                x = np.array(x).astype(float)
                new_y = int(np.argmax(np.dot(self.arr_of_weights, x)))
                alpha_lambada = 1 - self.learning_rate * self.lambada
                # Update.
                if y != new_y:
                    alpha = self.learning_rate * x
                    for i in range(len(self.arr_of_weights)):
                        if i == y:
                            self.arr_of_weights[y] *= alpha_lambada
                            self.arr_of_weights[y] += alpha
                        elif i == new_y:
                            self.arr_of_weights[new_y] *= alpha_lambada
                            self.arr_of_weights[new_y] -= alpha
                        else:
                            self.arr_of_weights[i] *= alpha_lambada
                else:
                    for i in range(len(self.arr_of_weights)):
                        if i != y:
                            self.arr_of_weights[i] *= alpha_lambada

    # Predict the classification using the input data vector and the calculated weights.
    def predict_func(self, data_of_inputs):
        x = np.array(data_of_inputs).astype(float)
        # Predict.
        prediction = np.argmax(np.dot(self.arr_of_weights, x))
        return prediction


# ********************************************************************************
# ****************************** PA - Passive Aggressive *************************
# ********************************************************************************
class PassiveAggressive:
    def __init__(self):
        # The learning rate.
        self.learning_rate = 0.01
        # Epochs.
        self.thresh_hold = 400
        # The weight array, of size 3 and 8 because we have 3 classes and 8 features.
        self.arr_of_weights = np.zeros((3, 8))

    # Train the PA algorithm using the training data.
    def train_func(self, train_data_set, labels):
        for t in range(self.thresh_hold):
            for x, y in zip(train_data_set, labels):
                y = int(float(y))
                x = np.array(x).astype(float)
                new_y = int(np.argmax(np.dot(self.arr_of_weights, x)))
                # Update.
                if y != new_y:
                    a = max(0, (1 - (np.dot(self.arr_of_weights[int(y)], x))
                                + (np.dot(self.arr_of_weights[int(new_y)], x))))
                    b = 2 * (np.linalg.norm(x) ** 2)
                    c = a / b
                    d = c * x
                    self.arr_of_weights[y] += d
                    self.arr_of_weights[new_y] -= d

    # Predict the classification using the input data vector and the calculated weights.
    def predict_func(self, test_set):
        x = np.array(test_set).astype(float)
        # Predict.
        prediction = np.argmax(np.dot(self.arr_of_weights, x))
        return prediction
# *******************************************************************************
# ****************************************************************************
# ****************************************************************************


# Main implementation.
def main():
    # ****************************************************************************
    # ****************************** SETUP **************************************
    # ****************************************************************************
    # Get training and testing files from command line.
    train_x = sys.argv[1]
    train_y = sys.argv[2]
    test_x = sys.argv[3]
    # Get the data from each file in the form of string lists.
    data1 = get_data(train_x)
    data2 = get_data(train_y)
    test_data = get_data(test_x)
    training_inputs = [0] * len(data1)
    testing_inputs = [0] * len(test_data)
    # ****************************************************************************
    # ************************ TRAINING DATA *************************************
    # ****************************************************************************
    # Replace each M,F,I input in the training example with a binary representation
    # with 'One Hot Encoding'.
    for i in range(len(data1)):
        new_list = list(data1[i])
        if data1[i][0] == 'M':
            new_list.pop(0)
            new_list = ["0.25"] + new_list
        elif data1[i][0] == 'F':
            new_list.pop(0)
            new_list = ["0.5"] + new_list
        elif data1[i][0] == 'I':
            new_list.pop(0)
            new_list = ["0.75"] + new_list
        new_list = "".join(new_list)
        training_inputs[i] = new_list
    # ****************************************************************************
    # ************************ TESTING DATA **************************************
    # ****************************************************************************
    # Replace each M,F,I input in the testing examples with a numerical representation
    for i in range(len(test_data)):
        test_list = list(test_data[i])
        if test_data[i][0] == 'M':
            test_list.pop(0)
            test_list = ["0.25"] + test_list
        elif test_data[i][0] == 'F':
            test_list.pop(0)
            test_list = ["0.5"] + test_list
        elif test_data[i][0] == 'I':
            test_list.pop(0)
            test_list = ["0.75"] + test_list
        test_list = "".join(test_list)
        testing_inputs[i] = test_list
    # ****************************************************************************
    # ************** CONVERT TRAINING BACK TO FLOATS *****************************
    # ****************************************************************************
    arr_floats = [0] * len(training_inputs)
    training_arr = []
    # Convert back to float.
    for i in range(len(training_inputs)):
        str_i = training_inputs[i]
        x = str_i.split(',')
        for r in range(len(x)):
            x[r] = float(x[r])
        arr_floats[i] = x
        training_arr.append(np.array(arr_floats[i]))
    # Convert the labels to integers.
    arr_ints = [0] * len(data2)
    labels_arr = []
    for i in range(len(data2)):
        str_i = data2[i]
        x = str_i.split(',')
        for r in range(len(x)):
            x[r] = float(x[r])
        arr_ints[i] = x
        labels_arr.append(np.array(arr_ints[i]))
    # ****************************************************************************
    # ***************** CONVERT TESTING BACK TO FLOATS ***************************
    # ****************************************************************************
    test_floats = [0] * len(testing_inputs)
    testing_arr = []
    for i in range(len(testing_inputs)):
        str_i = testing_inputs[i]
        x = str_i.split(',')
        for r in range(len(x)):
            x[r] = float(x[r])
        test_floats[i] = x
        testing_arr.append(np.array(test_floats[i]))
    # ****************************************************************************
    # ************************** TESTING & TRAINING PHASE ************************
    # ****************************************************************************
    # First normalize the data sets.
    training_arr = normalize_min_max(training_arr)
    test_floats = normalize_min_max(test_floats)
    # Create the perceptron class to use the perceptron algorithm.
    perceptron = Perceptron()
    # Train the perceptron using our arguments from the train_x and train_y files.
    perceptron.train_func(training_arr, labels_arr)
    # Check perceptron accuracy.
    # test_perceptron(training_arr,labels_arr, perceptron.arr_of_weights)
    # Create the SVM class to use the SVM algorithm.
    svm = SVM()
    # Train the SVM using our arguments from the train_x and train_y files.
    svm.train_func(training_arr, labels_arr)
    # Check the SVM accuracy.
    # test_svm(training_arr,labels_arr, svm.arr_of_weights)
    # Create the PA.
    pa = PassiveAggressive()
    # Train the PA algorithm.
    pa.train_func(training_arr, labels_arr)
    # Check the PA accuracy.
    # test_pa(training_arr, labels_arr, pa.arr_of_weights)
    # Use algorithm for each input in the test_x file we converted to floats earlier.
    for i in range(len(test_floats)):
        # Get the arguments for the ML arguments from the file.
        inputs = np.array(test_floats[i])
        # Label from train_y
        # y = labels_arr[i]
        # Perceptron, SVM and PA results.
        res = "perceptron: {}, svm: {}, pa: {}".format(perceptron.predict_func(inputs), svm.predict_func(inputs),
                                                       pa.predict_func(inputs))
        print(res)
    # ****************************************************************************
    # ****************************************************************************
    # ****************************************************************************


if __name__ == "__main__":
    main()
