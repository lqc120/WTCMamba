import triton
import triton.language as tl
import torch
import torch.nn.functional as F


@triton.jit
def belloch_scan_kernel(
        A_ptr, X_ptr,
        B: tl.constexpr, D: tl.constexpr, L: tl.constexpr, N: tl.constexpr,
        stride_A_b, stride_A_d, stride_A_l, stride_A_n,
        stride_X_b, stride_X_d, stride_X_l, stride_X_n,
        BLOCK_SIZE: tl.constexpr,
        LOG2_BLOCK_SIZE: tl.constexpr,
):
    b = tl.program_id(0)
    d = tl.program_id(1)
    n = tl.program_id(2)

    out_of_bounds = (b >= B) | (d >= D) | (n >= N)
    if out_of_bounds:
        return

    offs_l = tl.arange(0, BLOCK_SIZE)
    mask_l = offs_l < L

    A_base = b * stride_A_b + d * stride_A_d + n * stride_A_n
    X_base = b * stride_X_b + d * stride_X_d + n * stride_X_n

    A = tl.load(A_ptr + A_base + offs_l * stride_A_l, mask=mask_l, other=1.0)
    X = tl.load(X_ptr + X_base + offs_l * stride_X_l, mask=mask_l, other=0.0)

    for k in tl.static_range(1, LOG2_BLOCK_SIZE + 1):
        t = 1 << k
        all_i = tl.arange(0, BLOCK_SIZE)

        mask_step = (all_i >= (t - 1)) & ((all_i - (t - 1)) % t == 0)
        mask_i = mask_step & (all_i < BLOCK_SIZE) & ((all_i - (t // 2)) >= 0) & mask_l

        j = all_i - (t // 2)

        A_j = tl.load(A_ptr + A_base + j * stride_A_l, mask=mask_i, other=1.0)
        X_j = tl.load(X_ptr + X_base + j * stride_X_l, mask=mask_i, other=0.0)
        A_i = tl.load(A_ptr + A_base + all_i * stride_A_l, mask=mask_i, other=1.0)
        X_i = tl.load(X_ptr + X_base + all_i * stride_X_l, mask=mask_i, other=0.0)

        new_A_i = A_i * A_j
        new_X_i = X_i + A_i * X_j

        tl.store(A_ptr + A_base + all_i * stride_A_l, new_A_i, mask=mask_i)
        tl.store(X_ptr + X_base + all_i * stride_X_l, new_X_i, mask=mask_i)

    last_idx = BLOCK_SIZE - 1
    mask_last = (last_idx < L) & (~out_of_bounds)
    last_X_val = tl.load(X_ptr + X_base + last_idx * stride_X_l, mask=mask_last, other=0.0)
    tl.store(A_ptr + A_base + last_idx * stride_A_l, 1.0, mask=mask_last)
    tl.store(X_ptr + X_base + last_idx * stride_X_l, last_X_val, mask=mask_last)

    for k in tl.static_range(1, LOG2_BLOCK_SIZE):
        k_rev = LOG2_BLOCK_SIZE - k
        t = 1 << k_rev
        all_i = tl.arange(0, BLOCK_SIZE)

        mask_step = (all_i >= (t - 1)) & ((all_i - (t - 1)) % t == 0)
        mask_i = mask_step & (all_i < BLOCK_SIZE) & ((all_i - (t // 2)) >= 0) & mask_l

        j = all_i - (t // 2)

        A_i = tl.load(A_ptr + A_base + all_i * stride_A_l, mask=mask_i, other=1.0)
        X_i = tl.load(X_ptr + X_base + all_i * stride_X_l, mask=mask_i, other=0.0)
        A_j = tl.load(A_ptr + A_base + j * stride_A_l, mask=mask_i, other=1.0)
        X_j = tl.load(X_ptr + X_base + j * stride_X_l, mask=mask_i, other=0.0)

        new_A_j = A_i * A_j
        new_X_j = X_i * A_j + X_j

        tl.store(A_ptr + A_base + j * stride_A_l, new_A_j, mask=mask_i)
        tl.store(X_ptr + X_base + j * stride_X_l, new_X_j, mask=mask_i)

    tl.store(X_ptr + X_base + offs_l * stride_X_l, X, mask=mask_l)


def triton_pscan(A, X):
    assert A.is_cuda and X.is_cuda, "Triton PScan仅支持CUDA张量"
    B = int(A.shape[0])
    D = int(A.shape[1])
    L = int(A.shape[2])
    N = int(A.shape[3])

    BLOCK_SIZE = min(triton.next_power_of_2(L), 256)
    LOG2_BLOCK_SIZE = BLOCK_SIZE.bit_length() - 1
    assert (1 << LOG2_BLOCK_SIZE) == BLOCK_SIZE, "BLOCK_SIZE必须是2的幂"

    grid = (B, D, N)
    belloch_scan_kernel[grid](
        A, X,
        B, D, L, N,
        A.stride(0), A.stride(1), A.stride(2), A.stride(3),
        X.stride(0), X.stride(1), X.stride(2), X.stride(3),
        BLOCK_SIZE=BLOCK_SIZE,
        LOG2_BLOCK_SIZE=LOG2_BLOCK_SIZE,
    )
    return X[:, :, :L, :]


def selective_scan_parallel(u, delta, A, B, C, D=None, z=None, delta_bias=None, delta_softplus=False):
    dtype = u.dtype
    BATCH, ED, LEN = delta.shape
    N = A.shape[1]
    G = B.shape[1]
    assert ED % G == 0, f"ED({ED}) must be divisible by G({G})"
    ED_PER_GROUP = ED // G

    delta = delta.clone()
    if delta_bias is not None:
        delta = delta + delta_bias[None, :, None].to(dtype)
    if delta_softplus:
        delta = F.softplus(delta)

    is_complex = A.is_complex()
    if is_complex:
        B = torch.view_as_complex(B.float().reshape(*B.shape[:-1], -1, 2)).to(dtype)
        C = torch.view_as_complex(C.float().reshape(*C.shape[:-1], -1, 2)).to(dtype)
    else:
        B = B.to(dtype)
        C = C.to(dtype)

    u_t = u.transpose(1, 2)
    delta_t = delta.transpose(1, 2)
    B_t = B.permute(0, 3, 1, 2)
    C_t = C.permute(0, 3, 1, 2)

    delta_A = torch.einsum('bl d, d n -> bl d n', delta_t, A)
    delta_A = torch.clamp(delta_A, min=-15.0, max=15.0)
    deltaA = torch.exp(delta_A)

    delta_reshaped = delta_t.reshape(BATCH, LEN, G, ED_PER_GROUP)
    deltaB = delta_reshaped.unsqueeze(-1) * B_t.unsqueeze(3)
    u_reshaped = u_t.reshape(BATCH, LEN, G, ED_PER_GROUP)
    BX = deltaB * u_reshaped.unsqueeze(-1)
    BX = BX.reshape(BATCH, LEN, ED, N)

    deltaA = deltaA.transpose(1, 2)
    BX = BX.transpose(1, 2)
    hs = triton_pscan(deltaA, BX)
    hs = hs.transpose(1, 2)

    hs_reshaped = hs.reshape(BATCH, LEN, G, ED_PER_GROUP, N)
    y = torch.einsum('blgdn,blgn->blgd', hs_reshaped, C_t)
    y = y.reshape(BATCH, LEN, ED)

    if D is not None:
        y = y + D[None, None, :].to(dtype) * u_t
    if z is not None:
        z_t = z.transpose(1, 2)
        y = y * F.silu(z_t)

    y = y.transpose(1, 2).to(dtype)
    if is_complex:
        y = y.real * 2

    return y


class SelectiveScanTriton(torch.nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, u, delta, A, B, C, D=None, z=None, delta_bias=None, delta_softplus=False):
        return selective_scan_parallel(u, delta, A, B, C, D, z, delta_bias, delta_softplus)


def selective_scan_ref(u, delta, A, B, C, D=None, z=None, delta_bias=None, delta_softplus=False,
                       return_last_state=False):
    dtype_in = u.dtype
    u = u.float()
    delta = delta.float()
    if delta_bias is not None:
        delta = delta + delta_bias[..., None].float()
    if delta_softplus:
        delta = F.softplus(delta)
    batch, dim, dstate = u.shape[0], A.shape[0], A.shape[1]
    is_variable_B = B.dim() >= 3
    is_variable_C = C.dim() >= 3
    if A.is_complex():
        if is_variable_B:
            B = torch.view_as_complex(B.float().reshape(*B.shape[:-1], -1, 2))
        if is_variable_C:
            C = torch.view_as_complex(C.float().reshape(*C.shape[:-1], -1, 2))
    else:
        B = B.float()
        C = C.float()
    x = A.new_zeros((batch, dim, dstate))
    ys = []
    deltaA = torch.exp(torch.einsum('bdl,dn->bdln', delta, A))
    if not is_variable_B:
        deltaB_u = torch.einsum('bdl,dn,bdl->bdln', delta, B, u)
    else:
        if B.dim() == 3:
            deltaB_u = torch.einsum('bdl,bnl,bdl->bdln', delta, B, u)
        else:
            B = torch.repeat_interleave(B, dim // B.shape[1], dim=1)
            deltaB_u = torch.einsum('bdl,bdnl,bdl->bdln', delta, B, u)
    if is_variable_C and C.dim() == 4:
        C = torch.repeat_interleave(C, dim // C.shape[1], dim=1)
    last_state = None
    for i in range(u.shape[2]):
        x = deltaA[:, :, i] * x + deltaB_u[:, :, i]
        if not is_variable_C:
            y = torch.einsum('bdn,dn->bd', x, C)
        else:
            if C.dim() == 3:
                y = torch.einsum('bdn,bn->bd', x, C[:, :, i])
            else:
                y = torch.einsum('bdn,bdn->bd', x, C[:, :, :, i])
        if i == u.shape[2] - 1:
            last_state = x
        if y.is_complex():
            y = y.real * 2
        ys.append(y)
    y = torch.stack(ys, dim=2)
    out = y if D is None else y + u * D.reshape(-1, 1)
    if z is not None:
        out = out * F.silu(z)
    out = out.to(dtype=dtype_in)
    return out if not return_last_state else (out, last_state)
