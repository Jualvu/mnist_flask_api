import math

class CategoricalCrossentropyLoss:

    @staticmethod
    def calculate(y_pred, y_true):
        """
            y_pred (array)
            y_true (array)
        """

        #clip predictions to avoid log(0)
        y_pred_clipped = []
        for val in y_pred:
            if val < 1e-7:
                y_pred_clipped.append(1e-7)
            elif val > (1-1e-7):
                y_pred_clipped.append(1-1e-7)
            else:
                y_pred_clipped.append(val)

        #to calculate crossentropy loss we multiply each prediction val with its real y result
        # because y_true looks like this (0,0,0,0,0,0,0,0,0,1), only the correct value will persist in the end
        true_class_index = y_true.index(1) #get the index of the correct class
        correct_confidences = y_pred_clipped[true_class_index] * y_true[true_class_index] # append the predicted value

        loss = -math.log(correct_confidences)

        return loss