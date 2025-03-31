# train_gui.py
import tkinter as tk
from tkinter import ttk
from threading import Thread
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg

from from_scratch import train_numpy_live, set_numpy_hyperparams
from from_pytorch import train_torch_live, set_torch_hyperparams

class TrainingApp:
    def __init__(self, root):
        self.root = root
        self.root.title("Wizualizacja procesu uczenia - Live")

        self.fig, self.axs = plt.subplots(1, 3, figsize=(15, 4))
        self.canvas = FigureCanvasTkAgg(self.fig, master=root)
        self.canvas.get_tk_widget().pack()

        self.controls_frame = tk.Frame(root)
        self.controls_frame.pack(pady=5)

        self._build_controls()

        self.status = tk.StringVar(value="Gotowy do treningu")
        tk.Label(root, textvariable=self.status).pack(pady=5)

        self.losses_np = []
        self.losses_torch = []
        self.mses_np = []
        self.mses_torch = []
        self.accs_np = []
        self.accs_torch = []

    def _build_controls(self):
        tk.Label(self.controls_frame, text="Liczba epok:").grid(row=0, column=0)
        self.epochs_var = tk.IntVar(value=10)
        tk.Entry(self.controls_frame, textvariable=self.epochs_var, width=5).grid(row=0, column=1)

        tk.Label(self.controls_frame, text="Learning rate:").grid(row=0, column=2)
        self.lr_var = tk.DoubleVar(value=0.1)
        tk.Entry(self.controls_frame, textvariable=self.lr_var, width=5).grid(row=0, column=3)

        tk.Label(self.controls_frame, text="Batch size:").grid(row=0, column=4)
        self.batch_var = tk.IntVar(value=64)
        tk.Entry(self.controls_frame, textvariable=self.batch_var, width=5).grid(row=0, column=5)

        tk.Label(self.controls_frame, text="Dropout:").grid(row=0, column=6)
        self.dropout_var = tk.DoubleVar(value=0.3)
        tk.Entry(self.controls_frame, textvariable=self.dropout_var, width=5).grid(row=0, column=7)

        tk.Label(self.controls_frame, text="Aktywacja:").grid(row=0, column=8)
        self.act_var = tk.StringVar(value="sigmoid")
        ttk.Combobox(self.controls_frame, textvariable=self.act_var, values=["sigmoid", "tanh", "relu"], width=7).grid(row=0, column=9)

        self.button = ttk.Button(self.controls_frame, text="Start treningu", command=self.start_training)
        self.button.grid(row=0, column=10, padx=10)

    def start_training(self):
        self.button.config(state=tk.DISABLED)
        self.status.set("Trening w toku...")

        self.losses_np.clear()
        self.losses_torch.clear()
        self.mses_np.clear()
        self.mses_torch.clear()
        self.accs_np.clear()
        self.accs_torch.clear()

        # Przekazanie hiperparametrów
        params = {
            'epochs': self.epochs_var.get(),
            'lr': self.lr_var.get(),
            'batch_size': self.batch_var.get(),
            'dropout': self.dropout_var.get(),
            'activation': self.act_var.get(),
        }
        set_numpy_hyperparams(params)
        set_torch_hyperparams(params)

        Thread(target=self.train_models, args=(params['epochs'],)).start()

    def train_models(self, epochs):
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