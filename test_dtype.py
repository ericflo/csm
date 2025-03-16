import torch

# Demo: tensors have dtype attribute
tensor = torch.zeros(10, 10)
print(f'Tensor dtype: {tensor.dtype}')
print('This works because tensors have a dtype attribute')

# Demo: strings don't have dtype attribute
string_val = 'test'
try:
    print(string_val.dtype)
except AttributeError as e:
    print(f'Expected error: {e}. This is the error we fixed.')

print("Our fix ensures we check if an attribute is a string before trying to access dtype")
print("The fix should prevent 'str' object has no attribute 'dtype' errors")