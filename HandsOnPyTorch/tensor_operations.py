import torch
import numpy as np
import matplotlib.pyplot as plt


def multiply_tensors(*args):
    res_t = 0
    for arg in args:
        if torch.is_tensor(arg):
            if not torch.is_tensor(res_t):
                res_t = arg
            else:
                if arg.size() == res_t.size():
                    res_t = res_t * arg
    return res_t


def substract_tensors(*args):
    res_t = 0
    for arg in args:
        if torch.is_tensor(arg):
            if not torch.is_tensor(res_t):
                res_t = arg
            else:
                if arg.size() == res_t.size():
                    res_t = res_t - arg
    return res_t


def add_tensors(*args):
    res_t = 0
    for arg in args:
        if torch.is_tensor(arg):
            if not torch.is_tensor(res_t):
                res_t = arg
            else:
                if arg.size() == res_t.size():
                    res_t = res_t + arg
    return res_t


if __name__ == "__main__":
    #tensor1 = torch.tensor([1.0, 2.0, True])
    #tensor2 = torch.tensor([-1.0, -5.0, False])

    #result_add = add_tensors(tensor1, tensor2, "blah blah")
    #print(result_add)
    #result_subst = substract_tensors(tensor1, tensor2)
    #print(result_subst)

    #tensor3 = torch.rand(2, 3) * 9 + 1  # random from 0 to 1 normalized to 1 to 10
    #tensor4 = torch.randint(1, 11, (2, 3))  # integers from 1 to 10
    #print(f"Tensor 3: {tensor3}")
    #print(f"Tensor 4: {tensor4}")

    #ceil: the closest integer greater than or equal to each element
    #print(f"Ceil tensor 3: {torch.ceil(tensor3)}")
    #print(f"Ceil tensor 4: {torch.ceil(tensor4)}")

    #floor: the closest integer lesser than or equal to each element
    #print(f"Floor tensor 3: {torch.floor(tensor3)}")
    #print(f"Floor tensor 4: {torch.floor(tensor4)}")

    #clamp: all tensor values clamped in the provided range in clamp
    #print(f"Tensor 3 clamp [5:7]: {torch.clamp(tensor3, 5, 7)}")
    #print(f"Tensor 4 clamp [5:7]: {torch.clamp(tensor4, 5, 7)}")

    #mean: average for tensor
    #tensor5 = torch.tensor([1, 2, 3, 4, 5, 6], dtype=torch.float32)
    #print(f"Tensor 5 mean: {tensor5.mean()}")
    #print(f"Tensor 5 median: {tensor5.median()}")

    #numpi and trigonomical functions
    #tensor6 = torch.tensor([-np.pi, -np.pi / 2, -1, 0, 1, np.pi / 2, np.pi])
    #print(f"Tensor 6 sin: {tensor6.sin()}")
    #print(f"Tensor 6 cos: {tensor6.cos()}")

    #list of evenly spaced numbers in a range
    #pi_tensor = torch.linspace(-np.pi / 2, np.pi / 2, steps=1000)
    #pi_spased_sin = pi_tensor.sin()
    #pi_spased_cos = pi_tensor.cos()
    #print(pi_spased_sin[0:5])
    #print(pi_spased_cos[0:5])

    #building a graph
    #plt.plot(pi_spased_sin, label="Sin")
    #plt.plot(pi_spased_cos, label="Cos")
    #plt.show()

    #tensor7 = torch.tensor([[[2, 4, 3], [5, 7, 6], [1, 1, 1]],
    #                        [[1, 1, 1], [1, 1, 1], [1, 1, 1]]])
    #tensor8 = torch.tensor([[[1, 4, 1], [1, 1, 2], [0, 1, 0]],
    #                        [[0, 0, 0], [0, 0, 0], [0, 0, 0]]])
    #mult_tensor = multiply_tensors(tensor7, tensor8)
    #print(f"Multiplication 1 result: {mult_tensor}")
    #print(f"Multiplied shape {mult_tensor.shape}")
    #rint(f"Multiplied dimensions {mult_tensor.ndim}")

    tensor_ones = torch.ones(4, 3, 2, dtype=torch.int32)
    tensor_rand = torch.floor(torch.rand(3, 2)*10+1).to(torch.int32)
    tensor9 = tensor_ones * tensor_rand
    print(f"Ones tensor: {tensor_ones}")
    print(f"Random tensor: {tensor_rand}")
    print(f"Broadcast tensor: {tensor9}\n")

    tensor_rand = torch.floor(torch.rand(3, 1)*10+1).to(torch.int32)
    tensor10 = tensor_ones * tensor_rand
    print(f"Random tensor: {tensor_rand}")
    print(f"Broadcast tensor: {tensor10}")

    tensor_rand = torch.floor(torch.rand(1, 2)*10+1).to(torch.int32)
    tensor11 = tensor_ones * tensor_rand
    print(f"Random tensor: {tensor_rand}")
    print(f"Broadcast tensor: {tensor11}")
