import torch
from einops import einsum, rearrange, reduce

from analog.analysis import AnalysisBase
from analog.analysis.utils import reconstruct_grad, do_decompose, rescaled_dot_product


class InfluenceFunction(AnalysisBase):
    def parse_config(self):
        return

    @torch.no_grad()
    def precondition(self, src, mode="svd"):
        preconditioned = {}
        for module_name in src.keys():
            src_log = src[module_name].to("cpu")
            if mode == "svd":
                hessian_svd = self.hessian_handler.get_hessian_svd_state(module_name)
                rotated_grad = einsum(
                    hessian_svd["backward"][1].t(),
                    src_log,
                    hessian_svd["forward"][1],
                    "a b, batch b c, c d -> batch a d",
                )
                scale = torch.outer(hessian_svd["backward"][0], hessian_svd["forward"][0])
                prec_rotated_grad = rotated_grad / (scale + 0.1)
                preconditioned[module_name] = einsum(
                    hessian_svd["backward"][1],
                    prec_rotated_grad,
                    hessian_svd["forward"][1].t(),
                    "a b, batch b c, c d -> batch a d",
                )
            else:
                hessian_inv = self.hessian_handler.get_hessian_inverse_state(
                    module_name
                )
                preconditioned[module_name] = einsum(
                    hessian_inv["backward"],
                    src_log,
                    hessian_inv["forward"],
                    "a b, batch b c, c d -> batch a d",
                )
        return preconditioned

    @torch.no_grad()
    def compute_influence(self, src, tgt, preconditioned=False):
        if not preconditioned:
            src = self.precondition(src)

        total_influence = 0.0
        for module_name in src.keys():
            src_log, tgt_log = src[module_name], tgt[module_name]
            assert src_log.shape[1:] == tgt_log.shape[1:]
            src_log_expanded = rearrange(src_log, "n ... -> n 1 ...")
            tgt_log_expanded = rearrange(tgt_log, "m ... -> 1 m ...")
            module_influence = reduce(
                src_log_expanded * tgt_log_expanded, "n m a b -> n m", "sum"
            )
            total_influence += module_influence.squeeze()
        return total_influence

    def compute_self_influence(self, src):
        src_pc = self.precondition(src)
        total_influence = 0.0
        for module_name in src_pc.keys():
            src_pc_log, src_log = src_pc[module_name], src[module_name]
            module_influence = reduce(src_pc_log * src_log, "n a b -> n", "sum")
            total_influence += module_influence.squeeze()
        return total_influence

    def compute_influence_all(self, src, loader):
        if_scores = []
        src = self.precondition(src)
        for tgt_ids, tgt in loader:
            if_scores.append(self.compute_influence(src, tgt, preconditioned=True))
        return torch.cat(if_scores, dim=-1)
