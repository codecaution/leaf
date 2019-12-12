import numpy as np

# gradient_list: type  list
# gradient_list[0]: type  numpy.ndarray
class SignUpdate:

    def GradientCompress(self, gradient_list):
        compressed_gradient_list = []
        element_num = 0
        for gradient in gradient_list:
            element_num += np.size(gradient)
            compressed_gradient_list.append(np.sign(gradient))

        return compressed_gradient_list, element_num * 32,  element_num
    
    def MajorityVote(self, gradient_list):
        compressed_gradient_list = []
        for gradient in gradient_list:
            compressed_gradient_list.append(np.sign(gradient))

        return compressed_gradient_list

class SigumUpdate:
    def __init__(self, belta):
        self.momentum = None
        self.belta = belta

    def GradientCompress(self, gradient_list):
        compressed_gradient_list = []
        element_num = 0
        if self.momentum is None:
            for i, gradient in enumerate(gradient_list):
                element_num += np.size(gradient)
                self.momentum[i] = (1 - self.belta) * gradient
                compressed_gradient_list.append(np.sign(self.momentum[i]))
        else:
            for i, gradient in enumerate(gradient_list):
                element_num += np.size(gradient)
                self.momentum[i] = self.belta * self.momentum + (1 - self.belta) * gradient
                compressed_gradient_list.append(np.sign(self.momentum[i]))

        return compressed_gradient_list, element_num * 32, element_num
    
    def MajorityVote(self, gradient_list):
        compressed_gradient_list = []
        for gradient in gradient_list:
            compressed_gradient_list.append(np.sign(gradient))

        return compressed_gradient_list


SignSGDUpdate = SignUpdate()
SigumSGDUpdate = SigumUpdate(0.9)

