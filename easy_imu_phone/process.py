import numpy as np

x = np.load('imu_x.npy')
y = np.load('imu_y.npy')

print(len(x))

lens = len(x)
train_len = lens * 4 // 5

train_x = x[:train_len]
test_x = x[train_len:]
train_y = y[:train_len]
test_y = y[train_len:]

np.save('x_train.npy', train_x)
np.save('y_train.npy', train_y)
np.save('x_test.npy', test_x)
np.save('y_test.npy', test_y)