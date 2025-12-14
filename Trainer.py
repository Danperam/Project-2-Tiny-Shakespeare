import os
import time
import torch
import matplotlib.pyplot as plt
from tqdm import tqdm


class Trainer:
    def __init__(
        self,
        model,
        optimizer,
        data_loader,
        device=None,
        max_iters=5000,
        eval_interval=100,
        eval_iters=200,
    ):
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.model = model.to(self.device)
        self.optimizer = optimizer
        self.data_loader = data_loader
        self.max_iters = max_iters
        self.eval_interval = eval_interval
        self.eval_iters = eval_iters
        self.train_losses = []
        self.val_losses = []
        self.training_time = 0.0

    @torch.no_grad()
    def _estimate_loss(self):
        self.model.eval()
        out = {}
        for split in ["train", "val"]:
            losses = torch.zeros(self.eval_iters)
            for k in range(self.eval_iters):
                x, y = self.data_loader.get_batch(split)
                x, y = x.to(self.device), y.to(self.device)
                _, loss = self.model(x, y)
                losses[k] = loss.item()
            out[split] = losses.mean().item()
        self.model.train()
        return out

    def fit(self):
        start_time = time.time()
        for step in tqdm(range(self.max_iters), desc="Training", ncols=80):
            if step % self.eval_interval == 0 or step == self.max_iters - 1:
                losses = self._estimate_loss()
                self.train_losses.append(losses["train"])
                self.val_losses.append(losses["val"])
                #print(
                #    f"step {step+1}: train loss {losses['train']:.4f}, val loss {losses['val']:.4f}"
                #)
            x, y = self.data_loader.get_batch("train")
            x, y = x.to(self.device), y.to(self.device)
            _, loss = self.model(x, y)
            self.optimizer.zero_grad(set_to_none=True)
            loss.backward()
            self.optimizer.step()
        self.training_time = time.time() - start_time
        return self.train_losses, self.val_losses

    def plot_learning_curve(self, path="learning_curves", title="Learning curve"):
        if not self.train_losses or not self.val_losses:
            print("No loss history to plot.")
            return None
        os.makedirs(path, exist_ok=True)
        fig, ax = plt.subplots(figsize=(6, 4))
        ax.plot(
            range(1, len(self.train_losses) + 1), self.train_losses, label="Training", color="firebrick"
        )
        ax.plot(range(1, len(self.val_losses) + 1), self.val_losses, label="Validation", color="darkslategrey")
        ax.set_xlabel("Epoch")
        ax.set_ylabel("Loss")
        ax.set_title(title)
        ax.legend()
        out_file = os.path.join(path, "learning_curve.png")
        
        plt.savefig(out_file, dpi=300, bbox_inches="tight")
        plt.close(fig)
        
