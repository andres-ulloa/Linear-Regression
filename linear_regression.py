import math
import numpy as np
import matplotlib.pyplot as plt 
from model import model


def load_dataset(path_list):
    dataset = list()
    for path in path_list:
        dataset.append(np.genfromtxt(path))
    
    return dataset

#plots a two axis list containing np arrays

def plot_dataset(dataset, labelX, labelY):
    plt.scatter(dataset[0], dataset[1])
    plt.xlabel(labelX)
    plt.ylabel(labelY)
    plt.show()


def compute_hypothesis(order, weights, feature_vector):

    dependent_variable = 0
    feature_vector_index = 0
    power_index = 0

    for i in range(0, len(weights)):
        
        if i > 0:
            power_index += 1 
            dependent_variable = dependent_variable + weights[i] * pow(feature_vector[feature_vector_index], power_index)
        else:
            dependent_variable += weights[i]

        if power_index == order and i + 1 < len(weights):
            
            feature_vector_index += 1
            power_index = 0  
 
    return dependent_variable


#runs batch gradient descent (every single data point is considered for each weight update)
def run_gradient_descent(dataset, learning_rate = 0.5, hypothesis_order = 3, num_features = 1, num_epochs = 2500):
    
    weights = np.random.uniform(low = -0.1, high = 0.1, size = (hypothesis_order * num_features + 1))
    error_registry = list()
    x_axis = dataset[0]
    y_axis = dataset[1]
    epsilon = 0.001
    print('\nNum epochs = ', num_epochs)
    print('\n\nPreparing Batch Gradient descent...')
    print("\nAlpha = ",learning_rate)
    print('\nWeights = ', weights)
    #input("\n\nPress Enter to continue...")
   

    for j in range(0, num_epochs):
        
        gradient_vector = np.zeros(len(weights), dtype = float)
        error = 0
        for h in range(0, len(x_axis)):
            
            feature_vector = list()

            for i in range(0, num_features):
                feature_vector.append((dataset[i])[h])
               

            label = (dataset[len(dataset) - 1])[h]
            cost = (label - compute_hypothesis(hypothesis_order, weights, feature_vector))
            
            power_index = 0
            feature_vector_index = 0  

            for x in range(0, len(weights)):

                gradient_vector[x] +=  cost * pow(feature_vector[feature_vector_index], power_index)
                """print('FVI = ', feature_vector_index, 'PWI = ', power_index, 'pow = ', pow(feature_vector[feature_vector_index], power_index))
                print('\nGV = ', gradient_vector)
                print('\nCost = ',cost)"""
                if power_index  == hypothesis_order and x + 1 < len(weights):

                    power_index = 0
                    feature_vector_index += 1 

                power_index += 1
              
                
            error += cost

        for x in range(0, len(weights)):
            weights[x] += (learning_rate * gradient_vector[x])
       
        error_registry.append(error)
        if j == 0:
            print('Weights after 1 iteration = \n', weights)
            
        

    regression_model = model(weights, hypothesis_order, error_registry)
    print('Done.')
    print('\n\n\nFunction is now "epsilon exhausted"')
    print('Optimization is over.')
 
    return regression_model


def plot_error_curve(error_registry):
    plt.plot(error_registry, linewidth = 3)
    plt.show()


def plot_model_function(model, num_examples, dataset, labelX, labelY):
    
    model_samples_y = np.zeros((num_examples), np.float)
    model_samples_x = np.zeros((num_examples), np.float)
    model_samples = list()
    print('Model weights = ', model.weights)
    for i in range(1, num_examples + 1):

        feature_vector = list()
        x_coord_value =  (i - 1)/(101 - 1)
        feature_vector.append(x_coord_value)
        model_samples_y[i - 1] =  model.sample_model(feature_vector)
        model_samples_x[i - 1] = x_coord_value
    
    plt.xlabel(labelX)
    plt.ylabel(labelY)
    plt.plot(model_samples_x, model_samples_y ,linewidth = 2.0, color = 'C2')
    plt.scatter(dataset[0], dataset[1])
    plt.show()


def main():

    file_path_list = list()
    file_path_list.append('ex2x.csv')
    file_path_list.append('ex2y.csv')
    dataset = load_dataset(file_path_list)
    
    plot_dataset(dataset, 'Age', 'Height')
    model = run_gradient_descent(dataset, 0.001, 1, 1, 500)
    plot_error_curve(model.error_registry)
    plot_model_function(model, 1000, dataset, 'Age', 'Height')
    boy1 = np.array([3.5])
    boy2 = np.array([7])
    
    print('3.5 yrs =  ', model.sample_model(boy1))
    print('7 yrs = ', model.sample_model(boy2))
    

if __name__ == '__main__':
    main()