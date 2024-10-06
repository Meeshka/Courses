import torch

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
    tensor1 = torch.tensor([1.0, 2.0, True])
    tensor2 = torch.tensor([-1.0, -5.0, False])

    result_add = add_tensors(tensor1, tensor2, "blah blah")
    #print(result_add)
    result_subst = substract_tensors(tensor1, tensor2)
    #print(result_subst)

    tensor3 = torch.rand(2, 3) * 9 + 1 # random from 0 to 1 normalized to 1 to 10
    tensor4 = torch.randint(1,11,(2,3)) # integers from 1 to 10
    print(f"Tensor 3: {tensor3}")
    print(f"Tensor 4: {tensor4}")

    #ceil: the closest integer greater than or equal to each element
    print(f"Ceil tensor 3: {torch.ceil(tensor3)}")
    print(f"Ceil tensor 4: {torch.ceil(tensor4)}")

    #floor: the closest integer lesser than or equal to each element
    print(f"Floor tensor 3: {torch.floor(tensor3)}")
    print(f"Floor tensor 4: {torch.floor(tensor4)}")

    #clamp: all tensor values clamped in the provided range in clamp
    print(f"Tensor 3 clamp [5:7]: {torch.clamp(tensor3, 5, 7)}")
    print(f"Tensor 4 clamp [5:7]: {torch.clamp(tensor4, 5, 7)}")
