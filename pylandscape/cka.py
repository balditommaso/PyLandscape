import numpy as np
import torch
from torch import nn
from typing import Dict, List, Optional, Tuple


class CKA:
    """
    Minibatch CKA between all pairs of leaf-module activations across two models.

    Memory strategy
    ---------------
    * Gram vectors are moved to CPU immediately after being computed on GPU,
      so only one gram vector at a time lives on the GPU.
    * Accumulator tensors (hsic_acc, hsic_self*) are kept on CPU throughout;
      only the per-batch dot-products hit the GPU briefly.
    * torch.cuda.empty_cache() is called once per batch, not per layer.
    """

    def __init__(
        self,
        device: Optional[torch.device] = None,
        dtype: torch.dtype = torch.float32,
    ):
        self.device = torch.device("cpu") if device is None else device
        self.dtype = dtype

    # ------------------------------------------------------------------
    # Gram matrix helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _is_leaf(module: nn.Module) -> bool:
        return len(list(module.children())) == 0

    def _u_centered_gram_vector(self, x: torch.Tensor) -> torch.Tensor:
        """
        Compute the u-centred gram vector for a batch of activations.

        Args:
            x: Tensor of shape (n, ...) — n examples, arbitrary trailing dims.

        Returns:
            Flattened gram vector of shape (n*n,) on CPU, dtype=self.dtype.
            Returns a zero vector when n <= 2.
        """
        n = x.shape[0]
        if n <= 2:
            return torch.zeros(n * n, dtype=self.dtype)

        # Flatten spatial / channel dims and compute gram on the model's device
        x_flat = x.reshape(n, -1).to(device=self.device, dtype=self.dtype)
        gram = x_flat @ x_flat.t()          # (n, n)

        # Zero diagonal
        gram.fill_diagonal_(0.0)

        # Row-means excluding diagonal (Szekely & Rizzo U-statistic)
        row_sum = gram.sum(dim=1)           # (n,)
        means = row_sum / float(n - 2)
        means = means - means.sum() / (2.0 * float(n - 1))

        # Center rows and columns, then zero diagonal again
        gram = gram - means[:, None] - means[None, :]
        gram.fill_diagonal_(0.0)

        # Move to CPU immediately to free GPU memory
        return gram.reshape(-1).cpu()

    # ------------------------------------------------------------------
    # Hook management
    # ------------------------------------------------------------------

    def _register_hooks(
        self, model: nn.Module
    ) -> Tuple[List[str], List, Dict[str, torch.Tensor]]:
        """Register forward hooks on all leaf modules."""
        activations: Dict[str, torch.Tensor] = {}
        handles = []
        names = []

        def make_hook(name: str):
            def hook(module, inp, out):
                # Detach immediately; keep on whatever device the model uses
                act = out[0].detach() if isinstance(out, tuple) else out.detach()
                activations[name] = act
            return hook

        for name, module in model.named_modules():
            if self._is_leaf(module):
                handles.append(module.register_forward_hook(make_hook(name)))
                names.append(name)

        return names, handles, activations

    @staticmethod
    def _remove_hooks(handles: List) -> None:
        for h in handles:
            try:
                h.remove()
            except Exception:
                pass

    # ------------------------------------------------------------------
    # Per-batch gram accumulation
    # ------------------------------------------------------------------

    def _grams_for_names(
        self,
        names: List[str],
        activations: Dict[str, torch.Tensor],
        expected_batch_size: int,
    ) -> Tuple[List[int], List[torch.Tensor]]:
        """
        Compute u-centred gram vectors for every name that has an activation
        whose first dimension equals `expected_batch_size`.

        Layers are silently skipped when:
          - No activation was recorded (hook never fired this forward pass).
          - The activation is a scalar (dim == 0).
          - The activation's first dimension != expected_batch_size.
            This filters out embedding tables, RNN hidden states, layers
            called multiple times per forward pass, etc., all of which would
            produce gram vectors of the wrong length and break torch.stack.

        Returns:
            present_idx : indices into `names` for which a gram was computed
            grams       : corresponding gram vectors (all on CPU, same length)
        """
        present_idx = []
        grams = []
        expected_gram_len = expected_batch_size * expected_batch_size

        for idx, name in enumerate(names):
            act = activations.get(name)
            if act is None or act.dim() == 0:
                continue
            if act.shape[0] != expected_batch_size:
                # First dim is not the batch axis — skip this layer
                continue
            try:
                g = self._u_centered_gram_vector(act)
            except RuntimeError as exc:
                if "out of memory" in str(exc).lower():
                    torch.cuda.empty_cache()
                    g = self._u_centered_gram_vector(act.cpu())
                else:
                    raise
            # Sanity-check: gram must be exactly batch*batch elements
            if g.shape[0] != expected_gram_len:
                continue
            present_idx.append(idx)
            grams.append(g)
        return present_idx, grams

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def compare_models(
        self,
        model1: nn.Module,
        model2: Optional[nn.Module],
        dataloader: torch.utils.data.DataLoader,
        num_batches: int = 10,
    ) -> np.ndarray:
        """
        Compute the CKA similarity matrix between every pair of leaf layers
        from model1 (rows) and model2 (columns).

        Args:
            model1      : First model.
            model2      : Second model. Pass None to compare model1 with itself.
            dataloader  : Yields (inputs, *extras); only inputs are forwarded.
            num_batches : How many batches to accumulate over.

        Returns:
            cka_matrix: np.ndarray of shape (n_layers1, n_layers2), values in [0, 1].
        """
        same_model = model2 is None or model2 is model1
        model1.to(self.device).eval()
        if not same_model:
            model2.to(self.device).eval()
        else:
            model2 = model1

        names1, handles1, acts1 = self._register_hooks(model1)
        if same_model:
            names2, handles2, acts2 = names1, handles1, acts1
        else:
            names2, handles2, acts2 = self._register_hooks(model2)

        n1, n2 = len(names1), len(names2)

        # Accumulators stay on CPU — they are updated with scalar adds
        hsic_acc   = torch.zeros((n1, n2),  dtype=self.dtype)   # cross
        hsic_self1 = torch.zeros((n1,),     dtype=self.dtype)   # model1 self
        hsic_self2 = torch.zeros((n2,),     dtype=self.dtype)   # model2 self

        batches_done = 0
        with torch.no_grad():
            for batch, *_ in dataloader:
                batch = batch.to(self.device)

                acts1.clear()
                if not same_model:
                    acts2.clear()

                model1(batch)
                if not same_model:
                    model2(batch)

                batch_size = batch.shape[0]
                idx1, grams1 = self._grams_for_names(names1, acts1, batch_size)
                if same_model:
                    idx2, grams2 = idx1, grams1
                else:
                    idx2, grams2 = self._grams_for_names(names2, acts2, batch_size)

                if not idx1 or not idx2:
                    continue

                # Stack on CPU and do a single matmul — no GPU needed here
                G1 = torch.stack(grams1, dim=0)   # (L1p, P)
                G2 = torch.stack(grams2, dim=0)   # (L2p, P)

                block = G1 @ G2.t()               # (L1p, L2p)  — CPU matmul
                for a, ii in enumerate(idx1):
                    for b, jj in enumerate(idx2):
                        hsic_acc[ii, jj] += block[a, b]

                self_sq1 = (G1 * G1).sum(dim=1)   # (L1p,)
                for a, ii in enumerate(idx1):
                    hsic_self1[ii] += self_sq1[a]

                if same_model:
                    # Reuse self_sq1 for model2
                    for b, jj in enumerate(idx2):
                        hsic_self2[jj] += self_sq1[b]
                else:
                    self_sq2 = (G2 * G2).sum(dim=1)
                    for b, jj in enumerate(idx2):
                        hsic_self2[jj] += self_sq2[b]

                batches_done += 1
                # Free any intermediate GPU tensors accumulated this batch
                if self.device.type == "cuda":
                    torch.cuda.empty_cache()

                if batches_done >= num_batches:
                    break

        self._remove_hooks(handles1)
        if not same_model:
            self._remove_hooks(handles2)

        # CKA = HSIC_xy / sqrt(HSIC_xx * HSIC_yy)
        denom = torch.sqrt(
            hsic_self1[:, None] * hsic_self2[None, :]
        ).clamp(min=1e-12)
        cka_matrix = (hsic_acc / denom).numpy()
        return np.clip(cka_matrix, 0.0, 1.0)

    def compare_outputs(
        self,
        model1: nn.Module,
        model2: Optional[nn.Module],
        dataloader: torch.utils.data.DataLoader,
        num_batches: int = 10,
    ) -> float:
        """
        CKA similarity between the final outputs of model1 and model2.

        Returns:
            Scalar in [0, 1].
        """
        same_model = model2 is None or model2 is model1
        model1.to(self.device).eval()
        if not same_model:
            model2.to(self.device).eval()
        else:
            model2 = model1

        # Accumulate gram vectors incrementally on CPU rather than
        # concatenating all outputs and computing one giant gram.
        hsic_xy = torch.tensor(0.0, dtype=self.dtype)
        hsic_xx = torch.tensor(0.0, dtype=self.dtype)
        hsic_yy = torch.tensor(0.0, dtype=self.dtype)

        with torch.no_grad():
            for i, (batch, *_) in enumerate(dataloader, start=1):
                batch = batch.to(self.device)
                out1 = model1(batch).detach()
                out2 = model2(batch).detach() if not same_model else out1

                gx = self._u_centered_gram_vector(out1)   # CPU
                gy = self._u_centered_gram_vector(out2)   # CPU

                hsic_xy += torch.dot(gx, gy)
                hsic_xx += torch.dot(gx, gx)
                hsic_yy += torch.dot(gy, gy)

                if self.device.type == "cuda":
                    torch.cuda.empty_cache()

                if i >= num_batches:
                    break

        denom = torch.sqrt(hsic_xx * hsic_yy).clamp(min=1e-12)
        sim = float((hsic_xy / denom).item())
        return float(np.clip(sim, 0.0, 1.0)) if not np.isnan(sim) else 0.0