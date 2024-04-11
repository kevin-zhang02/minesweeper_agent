import matplotlib.pyplot as plt


def plot_stats(name, y_label, stats, window=10):
    plt.figure(figsize=(16, 4))
    plt.suptitle(name)
    plt.subplot(1, 2, 1)
    xline = range(
        0,
        len(stats.episode_lengths) - len(stats.episode_lengths) % window,
        window
    )
    plt.plot(xline, smooth(stats.episode_lengths, window=window))
    plt.ylabel("Episode Length")
    plt.xlabel("Episode Count")
    plt.subplot(1, 2, 2)
    plt.plot(xline, smooth(stats.episode_rewards, window=window))
    plt.ylabel(y_label)
    plt.xlabel("Episode Count")

    plt.show()


def smooth(x, window=10):
    return x[:window * (len(x) // window)].reshape(len(x) // window,
                                                   window).mean(axis=1)
