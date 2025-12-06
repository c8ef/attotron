import os

import torch
import torch.distributed as dist
import torch.nn.functional as F

from attotron import pgm


def update_rope_for_context_parallel(cos, sin):
    seq_len = cos.size(0)
    cp_rank, cp_world_size = pgm.pgm.cp_rank, pgm.pgm.cp_world_size
    assert seq_len % cp_world_size == 0
    size_per_partition = seq_len // cp_world_size
    start_idx, end_idx = cp_rank * size_per_partition, (cp_rank + 1) * size_per_partition
    return cos[start_idx:end_idx], sin[start_idx:end_idx]


class ContextComm:
    def __init__(self):
        self._pending_operations = []
        self._active_requests = None
        self.rank = pgm.pgm.cp_rank
        self.world_size = pgm.pgm.cp_world_size
        self.send_rank = pgm.pgm.cp_send_rank
        self.recv_rank = pgm.pgm.cp_recv_rank

    def send_recv(self, send_tensor, recv_tensor=None):
        result_tensor = torch.zeros_like(send_tensor) if recv_tensor is None else recv_tensor

        send_op = dist.P2POp(dist.isend, send_tensor, self.send_rank, group=pgm.pgm.cp_group)
        recv_op = dist.P2POp(dist.irecv, result_tensor, self.recv_rank, group=pgm.pgm.cp_group)
        self._pending_operations.extend([send_op, recv_op])

        return result_tensor

    def commit(self):
        if self._active_requests is not None:
            raise RuntimeError("commit called twice")
        self._active_requests = dist.batch_isend_irecv(self._pending_operations)

    def wait(self):
        if self._active_requests is None:
            raise RuntimeError("Wait called before commit")
        for req in self._active_requests:
            req.wait()
        torch.cuda.synchronize()
        self._active_requests = None
        self._pending_operations = []


def ring_attention_forward(q, k, v, sm_scale, is_causal):
    b, h, s, _ = q.shape
    S = torch.matmul(q, k.transpose(-2, -1)) * sm_scale
    if is_causal:
        causal_mask = torch.tril(torch.ones(s, s, device=q.device, dtype=torch.bool), diagonal=1)
        causal_mask = causal_mask.unsqueeze(0).unsqueeze(1).expand(b, h, s, s)
        S.masked_fill_(causal_mask, float("-inf"))
    S_max = torch.max(S, dim=-1, keepdim=True)[0]
    exp_S = torch.exp(S - S_max)
    sum_exp = torch.sum(exp_S, dim=-1, keepdim=True)
    log_sum_exp = torch.log(sum_exp) + S_max
    P = exp_S / sum_exp
    O = torch.matmul(P, v)
    return O, log_sum_exp.squeeze(-1)


def ring_attention_backward(dO, Q, K, V, O, softmax_lse, sm_scale, is_causal):
    b, h, s, _ = Q.shape
    S = torch.matmul(Q, K.transpose(-2, -1)) * sm_scale
    if is_causal:
        causal_mask = torch.tril(torch.ones(s, s, device=Q.device, dtype=torch.bool), diagonal=1)
        causal_mask = causal_mask.unsqueeze(0).unsqueeze(1).expand(b, h, s, s)
        S.masked_fill_(causal_mask, float("-inf"))
    P = torch.exp(S - softmax_lse.unsqueeze(-1))

    dV = torch.matmul(P.transpose(-2, -1), dO)
    dP = torch.matmul(dO, V.transpose(-2, -1))
    D = torch.sum(dO * O, dim=-1, keepdim=True)
    dS = P * (dP - D)
    dQ = torch.matmul(dS, K) * sm_scale
    dK = torch.matmul(dS.transpose(-2, -1), Q) * sm_scale
    return dQ, dK, dV


def update_out_and_lse(out, lse, block_out, block_lse):
    def _update(current_out, current_lse):
        current_out = current_out - F.sigmoid(block_lse - current_lse) * (current_out - block_out)
        current_lse = current_lse - F.logsigmoid(current_lse - block_lse)
        return current_out, current_lse

    block_out = block_out.to(torch.float32)
    block_lse = block_lse.unsqueeze(-1)

    if out is None:
        return block_out, block_lse

    out, lse = _update(out, lse)
    return out, lse


class RingAttention(torch.autograd.Function):
    @staticmethod
    def forward(ctx, q, k, v, sm_scale, is_causal):
        comm = ContextComm()
        k_og = k.clone()
        v_og = v.clone()
        out, lse = None, None
        next_k, next_v = None, None

        for step in range(comm.world_size):
            if step + 1 != comm.world_size:
                next_k = comm.send_recv(k)
                next_v = comm.send_recv(v)
                comm.commit()

            if not is_causal or step <= comm.rank:
                block_out, block_lse = ring_attention_forward(
                    q, k, v, sm_scale, is_causal and step == 0
                )
                out, lse = update_out_and_lse(out, lse, block_out, block_lse)

            if step + 1 != comm.world_size:
                comm.wait()
                k = next_k
                v = next_v

        out = out.to(q.dtype)
        ctx.save_for_backward(q, k_og, v_og, out, lse.squeeze(-1))
        ctx.sm_scale = sm_scale
        ctx.is_causal = is_causal
        return out

    @staticmethod
    def backward(ctx, dout, *args):
        q, k, v, out, softmax_lse = ctx.saved_tensors
        sm_scale = ctx.sm_scale
        is_causal = ctx.is_causal

        kv_comm = ContextComm()
        d_kv_comm = ContextComm()
        dq, dk, dv = None, None, None
        next_k, next_v = None, None
        next_dk, next_dv = None, None

        block_dq_buffer = torch.empty(q.shape, dtype=q.dtype, device=q.device)
        block_dk_buffer = torch.empty(k.shape, dtype=k.dtype, device=k.device)
        block_dv_buffer = torch.empty(v.shape, dtype=v.dtype, device=v.device)

        for step in range(kv_comm.world_size):
            if step + 1 != kv_comm.world_size:
                next_k = kv_comm.send_recv(k)
                next_v = kv_comm.send_recv(v)
                kv_comm.commit()

            if step <= kv_comm.rank or not is_causal:
                bwd_causal = is_causal and step == 0
                block_dq_buffer, block_dk_buffer, block_dv_buffer = ring_attention_backward(
                    dout, q, k, v, out, softmax_lse, sm_scale, bwd_causal
                )

                if dq is None:
                    dq = block_dq_buffer.to(torch.float32)
                    dk = block_dk_buffer.to(torch.float32)
                    dv = block_dv_buffer.to(torch.float32)
                else:
                    dq += block_dq_buffer
                    d_kv_comm.wait()
                    dk = block_dk_buffer + next_dk
                    dv = block_dv_buffer + next_dv
            elif step != 0:
                d_kv_comm.wait()
                dk = next_dk
                dv = next_dv

            if step + 1 != kv_comm.world_size:
                kv_comm.wait()
                k = next_k
                v = next_v

            next_dk = d_kv_comm.send_recv(dk)
            next_dv = d_kv_comm.send_recv(dv)
            d_kv_comm.commit()
        d_kv_comm.wait()
        return dq, next_dk, next_dv, None, None


def apply_context_parallel(model):
    os.environ["CONTEXT_PARALLEL"] = "1" if pgm.pgm.cp_world_size > 1 else "0"
    return model


def ring_attention(q, k, v, sm_scale, is_causal):
    return RingAttention.apply(q, k, v, sm_scale, is_causal)
