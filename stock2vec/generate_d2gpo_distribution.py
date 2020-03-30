import scipy.stats as stats
import sys
import numpy as np
import tqdm
from sklearn.utils.extmath import softmax
import h5py

mode = sys.argv[1]
assert mode in ['gaussian', 'linear', 'cosine']

def scatter(a, dim, index, b): # a inplace
    expanded_index = tuple([index if dim==i else np.arange(a.shape[i]).reshape([-1 if i==j else 1 for j in range(a.ndim)]) for i in range(a.ndim)])
    a[expanded_index] = b

if mode == 'gaussian':
    std = float(sys.argv[2])
    offset = int(sys.argv[3])
    mean = 0
    sample_width = int(sys.argv[4])
    softmax_position = sys.argv[5]
    softmax_temperature = float(sys.argv[6])
    order_path = sys.argv[7]
    output_path = sys.argv[8]

    distribution_func = stats.norm(mean, std)

elif mode == 'linear':
    k = float(sys.argv[2])
    assert k < 0
    b = 1.0
    offset = 0
    sample_width = int(sys.argv[3])
    softmax_position = sys.argv[4]
    softmax_temperature = float(sys.argv[5])
    order_path = sys.argv[6]
    output_path = sys.argv[7]

    assert (-b / k) >= (offset + sample_width)

elif mode == 'cosine':
    max_width = int(sys.argv[2])
    offset = int(sys.argv[3])
    sample_width = int(sys.argv[4])
    softmax_position = sys.argv[5]
    softmax_temperature = float(sys.argv[6])
    order_path = sys.argv[7]
    output_path = sys.argv[8]

    assert max_width >= (offset + sample_width)


assert softmax_position in ['presoftmax', 'postsoftmax']

# load the order information
with open(order_path, 'r', encoding='utf-8') as fin:
    data = fin.readlines()
data = [[int(item) for item in line.strip().split()] for line in data if len(line.strip())>0]

assert len(data) == len(data[0])

if sample_width == 0:
    sample_width = len(data)

x = np.arange(sample_width) + offset

if mode == 'gaussian':
    y_sample = distribution_func.pdf(x)
elif mode == 'linear':
    y_sample = k * x + b
else:
    y_sample = np.cos(np.pi / 2 * x / max_width)

if softmax_position == 'presoftmax':
    y_sample = y_sample / softmax_temperature 
    y_sample = softmax(np.expand_dims(y_sample,0)).squeeze(0)

y = np.zeros(len(data))

y[:sample_width] = y_sample

print(y[:sample_width])

label_weights = np.zeros((len(data), len(data)), dtype=np.float32)

for idx in tqdm.tqdm(range(len(data))):
    sort_index = np.array(data[idx])
    resort_index = np.zeros(len(data), dtype=np.int)
    natural_index = np.arange(len(data))
    scatter(resort_index, 0, sort_index, natural_index)
    weight = y[resort_index]
    label_weights[idx] = weight

f = h5py.File(output_path,'w')

f.create_dataset('weights', data=label_weights)

f.close()
