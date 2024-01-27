import matplotlib.pyplot as plt

x_axes = [100, 1000, 10000, 20000, 50000]
times = {
    '32 block size': [77, 7, 52, 146, 432],
    '64 block size': [0, 7, 39, 79, 268],
    '128 block size': [0, 5, 47, 126, 277],
    '256 block size': [0, 5, 38, 76, 212],
    '512 block size': [0, 5, 38, 77, 215],
    '1024 block size': [0, 5, 38, 77, 240]
}
speedups = {
    '32 block size': [0, 3.42857, 32.4423, 45.9247, 97.8009],
    '64 block size': [0, 3.42857, 43.2564, 84.8734, 157.649],
    '128 block size': [0, 4.6, 36.4681, 54.2143, 155.13],
    '256 block size': [0, 4.6, 45.1053, 89.8816, 202.693],
    '512 block size': [0, 4.6, 45.1053, 88.7143, 199.865],
    '1024 block size': [0, 4.6, 45.1053, 88.7143, 179.046]
}
colors = ['red', 'blue', 'green', 'orange', 'purple', 'yellow']

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
plt.xlabel('Words length')
plt.xticks(x_axes)
plt.ylabel('time (ms)')
plt.legend()

plt.show()
