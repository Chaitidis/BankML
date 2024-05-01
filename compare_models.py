import matplotlib
import matplotlib.pyplot as plt

# Load evaluation metrics from both scripts
test_accuracy_nn = 0.90  # Replace with actual test accuracy from NN.py
test_accuracy_nn_l2 = 0.92  # Replace with actual test accuracy from NN_L2.py
with open('test_accuracy_nn.txt', 'r') as f:
    test_accuracy_nn = float(f.read())

with open('test_accuracy_nn_l2.txt', 'r') as f:
    test_accuracy_nn_l2 = float(f.read())

# Plotting the results
models = ['NN.py', 'NN_L2.py']
accuracies = [test_accuracy_nn, test_accuracy_nn_l2]

# Plotting the results
models = ['NN.py', 'NN_L2.py']
accuracies = [test_accuracy_nn, test_accuracy_nn_l2]

plt.bar(models, accuracies, color=['blue', 'orange'])
plt.xlabel('Model')
plt.ylabel('Test Accuracy')
plt.title('Comparison of Test Accuracy between NN.py and NN_L2.py')
plt.ylim(0, 1)  # Set the y-axis limit
plt.show()
