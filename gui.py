import tkinter as tk
from tkinter import ttk
from threading import Thread
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg

from from_scratch import train_numpy
from from_pytorch import train_torch

class TrainingApp:
    def __init__(self, root):
        self.root = root
        self.root.title("Wizualizacja procesu uczenia")

        # Setup GUI
        self.button = ttk.Button(root, text="Start treningu", command=self.start_training)
        self.button.pack(pady=10)

        self.fig, self.ax = plt.subplots(1, 2, figsize=(10, 4))
        self.canvas = FigureCanvasTkAgg(self.fig, master=root)
        self.canvas.get_tk_widget().pack()

        self.status = tk.StringVar()
        self.status.set("Gotowy do treningu")
        tk.Label(root, textvariable=self.status).pack(pady=5)

    def start_training(self):
        self.button.config(state=tk.DISABLED)
        self.status.set("Trening w toku...")

        thread = Thread(target=self.train)
        thread.start()

    def train(self):
        # Trenuj sieci
        losses_np, mses_np = train_numpy()
        losses_torch, mses_torch = train_torch()

        # Aktualizacja wykresów
        self.ax[0].clear()
        self.ax[0].plot(losses_np, label='Loss (NumPy)')
        self.ax[0].plot(mses_np, label='MSE (NumPy)')
        self.ax[0].set_title('Sieć NumPy')
        self.ax[0].legend()

        self.ax[1].clear()
        self.ax[1].plot(losses_torch, label='Loss (PyTorch)')
        self.ax[1].plot(mses_torch, label='MSE (PyTorch)')
        self.ax[1].set_title('Sieć PyTorch')
        self.ax[1].legend()

        self.canvas.draw()
        self.status.set("Trening zakończony!")
        self.button.config(state=tk.NORMAL)

if __name__ == "__main__":
    root = tk.Tk()
    app = TrainingApp(root)
    root.mainloop()
