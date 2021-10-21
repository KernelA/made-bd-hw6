import numpy as np

def generate_data(size: int, ceoffs, shift: float) -> np.ndarray:
    data = np.random.rand(size, ceoffs.shape[0])
    target = data @ coefficient + shift
    target += np.random.normal(0, 0.35, size=data.shape[0])
    return np.hstack((data, target.reshape(-1, 1)))

num_coeff = 6
train_size = 500
test_size = 100

shift = 10
coefficient = np.random.rand(num_coeff)

train = generate_data(train_size, coefficient, shift)

feature_header = [f"feature_{i}" for i in range(train.shape[1] - 1)] 
train_header = feature_header + ["target"]
np.savetxt("data/train.csv", train, delimiter=",", header=",".join(train_header), comments="")

test = generate_data(test_size, coefficient, shift)

np.savetxt("data/test.csv", test[:, :-1], delimiter=",", header=",".join(feature_header), comments="")

header = ",".join([f"alpha_{i}" for i in range(len(coefficient))] + ["shift"])

coefficient = np.hstack((coefficient, [shift])).reshape(1, -1)

np.savetxt("data/true_coeff.csv", coefficient, delimiter=",", header=header, comments="")
