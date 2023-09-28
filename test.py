array1 = [1, 2, 3, 4]
array2 = [5, 6, 7, 8]
array3 = [0, 2, 4, 6]

result = [list(t) for t in zip(array1, array2, array3)]

print(result)
