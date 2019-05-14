import numpy as np

"""
The WeightMask class creates a array in the same shape of the neural network's
weights.Initially all values are set to "1". The method prune_weight will set
a given weight to "0". The method prune bias will set a given bias to "0".

apply_mask will multiply a model's weights with the mask and returns an updated model.
"""

class WeightMask():
    def __init__(self, model):
        self.mask =[]
        weights = model.get_weights()
        for layer in weights:
            layer_weights = np.ones(layer.shape)
            self.mask.append(layer_weights)

    def prune_weight(self, layer_index, node, connection):
        """set specific entry in the weight_mask zero to prune"""
        self.mask[layer_index][node, connection] = 0

    def get_mask(self):
        """returns the pruned mask for debugging reasons"""
        return self.mask

    def prune_bias(self, layer_index, node):
        """set specific bias in layer number "layer_index", at node number "node"."""
        self.mask[layer_index][node] = 0

    def prune_parameter(self, layer_index, prune_index):
        """prune parameter independent of being weight or bias"""
        if len(prune_index) == 2:
            self.prune_weight(layer_index, *prune_index)
        else:
            self.prune_bias(layer_index, prune_index)


    def apply_mask(self, model):
        """multiply mask with weights of model an returning model with new weights"""
        weights = model.get_weights()
        for i, layer in enumerate(weights):
            layer *= self.mask[i]
        model.set_weights(weights)
        return model

    def propagate_pruning(self, lower_layer_index, higher_layer_index):
        for i,entry in enumerate(self.mask[higher_layer_index]):
            if entry==0:
                self.mask[lower_layer_index][i,:] = 0
