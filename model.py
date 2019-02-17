import math
import numpy as np

class model:
    
    def __init__ (self, weights, order, error_registry):

        self.weights = weights
        self.order = order
        self.error_registry = error_registry

    def sample_model(self, feature_vector):

        dependent_variable = 0
        power_index = 0
        feature_vector_index = 0

        for i in range(0, len(self.weights)):

            if i > 0:
                power_index += 1
                dependent_variable += self.weights[i] * pow(feature_vector[feature_vector_index], power_index)
            else:
                dependent_variable += self.weights[i]

            if power_index == self.order and i + 1 < len(self.weights):
                feature_vector_index += 1
                power_index = 0  
                
        return dependent_variable