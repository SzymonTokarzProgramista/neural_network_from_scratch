# train_gui.py
import tkinter as tk
from tkinter import ttk
from threading import Thread
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg

from from_scratch import train_numpy_live
from from_pytorch import train_torch_live

class TrainingApp:
    def __init__(self, root):
        self.root = root
        self.root.title("Wizualizacja procesu uczenia - Live")

        self.fig, self.axs = plt.subplots(1, 3, figsize=(15, 4))
        self.canvas = FigureCanvasTkAgg(self.fig, master=root)
        self.canvas.get_tk_widget().pack()

        self.status = tk.StringVar(value="Gotowy do treningu")
        tk.Label(root, textvariable=self.status).pack(pady=5)

        self.button = ttk.Button(root, text="Start treningu", command=self.start_training)
        self.button.pack(pady=10)

        self.losses_np = []
        self.losses_torch = []
        self.mses_np = []
        self.mses_torch = []
        self.accs_np = []
        self.accs_torch = []

    def start_training(self):
        self.button.config(state=tk.DISABLED)
        self.status.set("Trening w toku...")

        self.losses_np.clear()
        self.losses_torch.clear()
        self.mses_np.clear()
        self.mses_torch.clear()
        self.accs_np.clear()
        self.accs_torch.clear()

        Thread(target=self.train_models).start()

    def train_models(self):
        epochs = 10

        for epoch in range(epochs):
            loss_np, mse_np, acc_np = train_numpy_live(epoch)
            loss_torch, mse_torch, acc_torch = train_torch_live(epoch)

            self.losses_np.append(loss_np)
            self.mses_np.append(mse_np)
            self.accs_np.append(acc_np)

            self.losses_torch.append(loss_torch)
            self.mses_torch.append(mse_torch)
            self.accs_torch.append(acc_torch)

            self.root.after(0, self.update_plot)

        self.root.after(0, lambda: self.status.set("Trening zako\u0144czony!"))
        self.root.after(0, lambda: self.button.config(state=tk.NORMAL))

    def update_plot(self):
        ax1, ax2, ax3 = self.axs

        ax1.clear()
        ax1.plot(self.losses_np, label='Loss NumPy', linestyle='--')
        ax1.plot(self.losses_torch, label='Loss PyTorch', linestyle='-')
        ax1.set_title("Loss")
        ax1.set_xlabel("Epoka")
        ax1.set_ylabel("Loss")
        ax1.legend()
        ax1.grid(True)

        ax2.clear()
        ax2.plot(self.mses_np, label='MSE NumPy', linestyle='--')
        ax2.plot(self.mses_torch, label='MSE PyTorch', linestyle='-')
        ax2.set_title("MSE")
        ax2.set_xlabel("Epoka")
        ax2.set_ylabel("MSE")
        ax2.legend()
        ax2.grid(True)

        ax3.clear()
        ax3.plot(self.accs_np, label='Accuracy NumPy', linestyle='--')
        ax3.plot(self.accs_torch, label='Accuracy PyTorch', linestyle='-')
        ax3.set_title("Dok\u0142adno\u015b\u0107")
        ax3.set_xlabel("Epoka")
        ax3.set_ylabel("Accuracy")
        ax3.legend()
        ax3.grid(True)

        self.canvas.draw_idle()

if __name__ == "__main__":
    root = tk.Tk()
    app = TrainingApp(root)
    root.mainloop()