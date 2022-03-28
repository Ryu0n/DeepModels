from src.trainer.model_trainer import ModelTrainer
from visualization.history_visualizer import HistoryVisualizer

if __name__ == '__main__':
    trainer = ModelTrainer()
    model, history = trainer.train('AlexNetRefactoredLight', 'mnist', batch_size=64, epochs=10)
    HistoryVisualizer.plot_history(history)
