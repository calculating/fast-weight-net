import tensorflow as tf
import time
import matplotlib.pyplot as plt

# Define the range of number of operations to test
num_ops_range = range(10, 6000, 200)

# Create a function that performs 4 parallel operations
def parallel_operations(num_ops):
    inputs = [tf.random.normal(shape=[num_ops, num_ops]) for _ in range(4)]
    outputs = []
    for i in range(4):
        outputs.append(tf.linalg.matmul(inputs[i], inputs[(i+1)%4]))
    return tf.reduce_sum(outputs)

# Measure the execution time for each number of operations in the range
times = []
for num_ops in num_ops_range:
    start_time = time.time()
    result = parallel_operations(num_ops)
    elapsed_time = time.time() - start_time
    times.append(elapsed_time)

# Create a plot of the number of operations vs time graph
plt.plot(num_ops_range, times)
plt.xlabel('Number of Operations')
plt.ylabel('Time (seconds)')
plt.title('Parallel TensorFlow Operations')
#plt.ylim(top=7)  
# set the maximum of the chart to 2x the max functional value
plt.ylim(top=2*max(times))

print(num_ops_range, times)
plt.plot(num_ops_range, times, color='red')  # set the plot color to red
plt.legend("Tf")
plt.show()
