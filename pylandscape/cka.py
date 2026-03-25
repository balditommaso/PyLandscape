import torch
import numpy as np
from .metric import Metric
from torch import nn, tensor
from typing import List, Dict, Tuple, Optional


class CKA(Metric):
    def __init__(self, device: Optional[torch.device] = None, dtype: torch.dtype = torch.float32, name: str = "cka"):
        super().__init__(name)
        self.device = torch.device('cpu') if device is None else device
        self.dtype = dtype


    @staticmethod
    def _is_leaf(module: torch.nn.Module) -> bool:
        return len(list(module.children())) == 0


    def _register_hooks(self, model: nn.Module) -> Tuple[List[str], List[torch.utils.hooks.RemovableHandle], Dict[str, tensor]]:
        """Register forward hooks on leaf modules. Returns (names, handles, activations_dict)."""
        activations = {}
        handles = []
        names = []

        def make_hook(name):
            def hook(module, inp, out):
                activations[name] = out[0].detach() if isinstance(out, tuple) else out.detach()
            return hook

        for name, module in model.named_modules():
            if self._is_leaf(module):
                handle = module.register_forward_hook(make_hook(name))
                handles.append(handle)
                names.append(name)

        return names, handles, activations


    def _remove_handles(self, handles: List[torch.utils.hooks.RemovableHandle]) -> None:
        for h in handles:
            try:
                h.remove()
            except Exception:
                pass


    def _u_centered_gram_vector(self, x: torch.Tensor) -> torch.Tensor:
        n = x.shape[0]
        x2 = x.reshape(n, -1)
        gram = x2 @ x2.t()  # [n, n]
        if n <= 2:
            return torch.zeros((n * n,), device=self.device, dtype=self.dtype)

        # zero diagonal
        gram = gram.clone()
        gram.fill_diagonal_(0.0)
        gram = gram.to(device=self.device, dtype=self.dtype)

        # means excluding diagonal: sum / (n - 2)
        row_sum = gram.sum(dim=1)  # shape [n]
        means = row_sum / float(n - 2)

        # subtract global correction
        means = means - means.sum() / (2.0 * (n - 1))

        # center
        gram = gram - means[:, None] - means[None, :]

        # zero diagonal again
        gram.fill_diagonal_(0.0)

        return gram.reshape(-1)
    


    def _collect_layer_grams_from_hooks(self, names: List[str], activations: Dict[str, tensor]) -> List[tensor]:
        grams = []
        for n in names:
            tensor_act = activations.get(n)
            if tensor_act is None:
                grams.append(None)
            else:
                grams.append(self._u_centered_gram_vector(tensor_act))
        return grams


    def compare_models(
        self,
        model1: nn.Module,
        model2: Optional[nn.Module],
        dataloader: torch.utils.data.DataLoader,
        num_batches: int = 10,
    ) -> np.ndarray:

        device = self.device
        model1.to(device).eval()
        if model2 is not None:
            model2.to(device).eval()
        else:
            model2 = model1

        # register hooks
        names1, handles1, acts1 = self._register_hooks(model1)
        names2, handles2, acts2 = self._register_hooks(model2)

        n1 = len(names1)
        n2 = len(names2)

        hsic_acc = torch.zeros((n1, n2), device=device, dtype=self.dtype)
        hsic_self1 = torch.zeros((n1,), device=device, dtype=self.dtype)
        hsic_self2 = torch.zeros((n2,), device=device, dtype=self.dtype)

        batches_done = 0
        with torch.no_grad():
            for i, (batch, *rest) in enumerate(dataloader, start=1):
                batch = batch.to(device)
                # clear activations dicts to avoid stale entries
                acts1.clear()
                acts2.clear()

                model1(batch)
                if model1 is not model2:
                    model2(batch)

                # collect grams in consistent order; skip layers missing for this forward
                grams1 = self._collect_layer_grams_from_hooks(names1, acts1)
                grams2 = self._collect_layer_grams_from_hooks(names2, acts2)

                # convert lists to stacked tensors; when a layer had no activation this iteration, we skip it
                # But shapes must align: each gram is (n*n,)
                present1_idx = [idx for idx, g in enumerate(grams1) if g is not None]
                present2_idx = [idx for idx, g in enumerate(grams2) if g is not None]

                if len(present1_idx) == 0 or len(present2_idx) == 0:
                    # nothing to do this batch
                    continue

                stacked1 = torch.stack([grams1[i] for i in present1_idx], dim=0)  # [L1p, P]
                stacked2 = torch.stack([grams2[i] for i in present2_idx], dim=0)  # [L2p, P]

                # accumulate cross HSIC for present layers
                # we need to add into hsic_acc at the correct row/col positions
                block = stacked1 @ stacked2.t()  # [L1p, L2p]
                for a, ii in enumerate(present1_idx):
                    for b, jj in enumerate(present2_idx):
                        hsic_acc[ii, jj] += block[a, b]

                # accumulate self terms for normalization
                self_sq1 = torch.einsum('ij,ij->i', stacked1, stacked1)  # [L1p]
                for a, ii in enumerate(present1_idx):
                    hsic_self1[ii] += self_sq1[a]

                self_sq2 = torch.einsum('ij,ij->i', stacked2, stacked2)  # [L2p]
                for b, jj in enumerate(present2_idx):
                    hsic_self2[jj] += self_sq2[b]

                batches_done += 1
                if batches_done >= num_batches:
                    break

        # remove hooks
        self._remove_handles(handles1)
        # if different model, remove second handles; if same model handles2==handles1, they were already removed
        if model1 is not model2:
            self._remove_handles(handles2)

        # finalize normalization: CKA = HSIC / sqrt(HSIC_xx * HSIC_yy)
        # For layers with zero denom, clamp to small positive to avoid div0
        denom = torch.sqrt(hsic_self1[:, None] * hsic_self2[None, :]).clamp(min=1e-12)
        cka_matrix = (hsic_acc / denom).cpu().numpy()
        cka_matrix = np.clip(cka_matrix, 0.0, 1.0)
        return cka_matrix


    def compare_outputs(
        self, 
        model1: nn.Module, 
        model2: Optional[nn.Module],
        dataloader: torch.utils.data.DataLoader, 
        num_batches: int = 10
    ) -> float:
        """
        Convenience: compute CKA between model outputs (final forward outputs).
        Returns scalar similarity.
        """

        device = self.device
        model1.to(device).eval()
        if model2 is not None:
            model2.to(device).eval()
        else:
            model2 = model1

        outputs1 = []
        outputs2 = []
        with torch.no_grad():
            for i, (batch, *rest) in enumerate(dataloader, start=1):
                batch = batch.to(device)
                out1 = model1(batch)
                out2 = model2(batch) if model1 is not model2 else out1
                outputs1.append(out1.detach())
                outputs2.append(out2.detach())
                if i >= num_batches:
                    break

        X = torch.cat(outputs1, dim=0)
        Y = torch.cat(outputs2, dim=0)
        gx = self._u_centered_gram_vector(X)
        gy = self._u_centered_gram_vector(Y)
        hsic = torch.dot(gx, gy)
        nx = torch.linalg.norm(gx)
        ny = torch.linalg.norm(gy)
        sim = (hsic / (nx * ny)).item() if nx > 0 and ny > 0 else 0.0
        if np.isnan(sim):
            sim = 0.0
        return float(sim)
