import matplotlib.pyplot as plt


class HistoryVisualizer:
    @staticmethod
    def vis(history, name):
        plt.title(f"{name.upper()}")
        plt.xlabel('epochs')
        plt.ylabel(f"{name.lower()}")
        value = history.history.get(name)
        val_value = history.history.get(f"val_{name}", None)
        epochs = range(1, len(value) + 1)
        plt.plot(epochs, value, 'b-', label=f'training {name}')
        if val_value is not None:
            plt.plot(epochs, val_value, 'r:', label=f'validation {name}')
        plt.legend(loc='upper center', bbox_to_anchor=(0.05, 1.2), fontsize=10, ncol=1)

    @staticmethod
    def plot_history(history):
        key_value = list(set([i.split("val_")[-1] for i in list(history.history.keys())]))
        plt.figure(figsize=(12, 4))
        for idx, key in enumerate(key_value):
            plt.subplot(1, len(key_value), idx + 1)
            HistoryVisualizer.vis(history, key)
        plt.tight_layout()
        plt.show()
