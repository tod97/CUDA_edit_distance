import matplotlib.pyplot as plt

x_axes = [10000, 100000, 500000, 1000000, 2000000]
times = {
    'sequential': [941, 7328, 41362, 76416, 125961],
    'parallel': [176, 1490, 11869, 25822, 48243]
}
speedups = {
    '': [5.34659, 4.91812, 3.48488, 2.95934, 2.61097],
}
colors = ['red', 'blue', 'green', 'yellow', 'orange', 'purple']

plt.figure(figsize=(10, 8))

for key, speedup, index in zip(speedups.keys(), speedups.values(), range(len(speedups))):
    plt.plot(x_axes, speedup, label=key, color=colors[index])

plt.title('Speedups')
plt.gcf().canvas.set_window_title('speedups')
plt.xlabel('Number of words')
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
