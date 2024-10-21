import torch

def ex1():
    t0 = torch.tensor(1000)  # 0-D tensor
    t1 = torch.tensor([9, 8, 7, 6])  # 1-D tensor
    t2 = torch.tensor([[1, 2, 3], [7, 5, 3]])  # 2-D tensor

    print(t0)
    print(t1)
    print(t2)

def ex2():
    return torch.ones((2, 3, 2), dtype=torch.int16)

if __name__ == "__main__":
    #ex1()
    a = ex2()
    print(a)
    print(a.size())
    print(a.shape)
    print(a.ndimension())
    print(a.ndim)
