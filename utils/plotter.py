import matplotlib.pyplot as plt

x_axes = [100, 1000, 10000, 20000, 50000]
times = {
    'sequential': [0, 24, 1738, 6800, 42577],
    'parallel': [75, 7, 48, 136, 296]
}
speedups = {
    '': [0, 3.42857, 36.2083, 50, 143.841],
}
colors = ['red', 'blue', 'green', 'yellow', 'orange', 'purple']

plt.figure(figsize=(10, 8))

for key, speedup, index in zip(speedups.keys(), speedups.values(), range(len(speedups))):
    plt.plot(x_axes, speedup, label=key, color=colors[index])

plt.title('Speedups')
plt.gcf().canvas.set_window_title('speedups')
plt.xlabel('Words length')
plt.xticks(x_axes)
plt.ylabel('speedup')
plt.legend()

plt.show()


plt.figure(figsize=(10, 8))

for key, time, index in zip(times.keys(), times.values(), range(len(times))):
    plt.plot(x_axes, time, label=key, color=colors[index])

plt.title('Execution times')
plt.gcf().canvas.set_window_title('times')
plt.xlabel('Number of words')
plt.xticks(x_axes)
plt.ylabel('time (ms)')
plt.legend()

plt.show()
