import numpy as np

class CategoricalCrossentropyLoss:

    """
        forward calculates the losses of every prediction in the output predictions array
    """
    # def forward(y_prediction, y_true):
    #     """
    #         quick note on clipped values:
    #         we clip the values to normalize the outputs and avoid overflow
    #         clip(array, min_val, max_val) so each value that in the range min - max is kept
    #         if a value exceeds one of the limits is going to be assign the limit value it exceeds
    #     """
    #     samples = len(y_prediction)
    #     y_prediction_clipped = np.clip(y_prediction, 1e-7, 1-1e-7)

    #     # print("\nfrom loss")
    #     # print(y_true.shape)
    #     # print(y_prediction.shape)

    #     # # ask for shape of y_true to manage both possible prediction formats
    #     # if len(y_true.shape) == 1:
    #     #     correct_confidences = y_prediction_clipped[range(samples), y_true]
    #     #     """
    #     #          when len(y_true.shape) == 1

    #     #             y_true = [1, 0, 0]
                
    #     #         the array represent the correct class on each example

    #     #         in this case, we need to index the correct class on each example --> y_prediction_clipped[range(samples), y_true]
    #     #         we basically index the value of the correct class on the y_prediction matrix to get what the probability is
                
    #     #     """

    #     if len(y_true.shape) == 2:
    #         correct_confidences = np.sum(y_prediction_clipped *  y_true, axis = 1)
    #         """
    #             when len(y_true.shape) == 2

    #                 y_true = [
    #                             [1, 0, 0],
    #                             [0, 1, 0],
    #                             [0, 0, 1]
    #                          ]
                
    #             the index where val==1 represent the correct class

    #             in this case, because its a matrix, a matrix sum on each row (axis=1) can be done 
    #         """

    #     #apply final negative log
    #     negative_log = -np.log(correct_confidences)

    #     return np.mean(negative_log) # get the mean of all losses


    def forward(y_prediction, y_true):
        """
        y_pred: numpy array of shape (n_samples, n_classes) — probabilities (from softmax)
        y_true: either:
            - 1D array of shape (n_samples,) with integer labels
            - or 2D array of shape (n_samples, n_classes) with one-hot encoded labels
        """
        # Clip predictions to avoid log(0)
        y_pred_clipped = np.clip(y_prediction, 1e-7, 1 - 1e-7)
        samples = y_prediction.shape[0]

        # If labels are one-hot encoded
        if len(y_true.shape) == 2:
            correct_confidences = np.sum(y_pred_clipped * y_true, axis=1)

        # If labels are integers (e.g., [3, 0, 4])
        elif len(y_true.shape) == 1:
            correct_confidences = y_pred_clipped[np.arange(samples), y_true]

        # Calculate the negative log likelihood
        negative_log_likelihoods = -np.log(correct_confidences)

        # Return the average loss
        return np.mean(negative_log_likelihoods)