import torch
import torch.distributed as dist
import os

# Indexes
X_MIN, X_MAX = 0, -1
Y_MIN, Y_MAX = 0, -1
Z_MIN, Z_MAX = 0, -1

def init_process(backend='nccl'):
    """
    Initialize NCCL backend for GPU communication
    """
    try:
        rank = int(os.environ["RANK"])
        world_size = int(os.environ["WORLD_SIZE"])
        local_rank = int(os.environ["LOCAL_RANK"])
    except KeyError:
        print("Erro: Torchrun variables (RANK, WORLD_SIZE) not found.")
        exit(1)

    device_id = torch.device(f"cuda:{local_rank}")
    torch.cuda.set_device(device_id)
    # torch.cuda.set_device(local_rank) # Redundante se set_device(device_id) já foi chamado

    if not dist.is_initialized():
        print(f"[Rank {rank}] Initialize backend {backend}...", flush=True)
        dist.init_process_group(
            backend=backend,
            rank=rank,
            world_size=world_size,
            device_id=device_id
        )

    dist.barrier()
    return rank, world_size, local_rank

def gather_all_data(local_tensor):
    rank = dist.get_rank()
    world_size = dist.get_world_size()

    if(world_size == 1):
        return local_tensor.cpu()

    tensor_gpu = local_tensor.contiguous()

    if rank == 0:
        gathered_list = [torch.empty_like(tensor_gpu) for _ in range(world_size)]
    else:
        gathered_list = None

    dist.gather(tensor_gpu, gather_list=gathered_list, dst=0)

    if rank == 0:
        # Proteção contra erros de topologia no gather
        if world_size < 4 or world_size % 4 != 0:
             # Fallback simples se a topologia não for 2x2
             return torch.cat(gathered_list, dim=2).cpu()

        num_z_slices = world_size // 4
        z_slices = []
        for i in range(num_z_slices):
            base = i * 4
            tl = gathered_list[base]     # Rank 0, 4...
            tr = gathered_list[base + 1] # Rank 1, 5...
            bl = gathered_list[base + 2] # Rank 2, 6...
            br = gathered_list[base + 3] # Rank 3, 7...

            row_top = torch.cat([tl, tr], dim=4)
            row_bot = torch.cat([bl, br], dim=4)

            z_slice = torch.cat([row_top, row_bot], dim=3)

            z_slices.append(z_slice)

        full_tensor = torch.cat(z_slices, dim=2)
        return full_tensor.cpu()

    return None

def halo_exchange(tensor, neighbors):
    """
    Asynchronous halo exchange with contiguous buffers.
    """
    reqs = []

    # Buffers para recebimento (precisam ser mantidos vivos até o wait())
    # Usamos um dicionário para facilitar a cópia de volta
    recv_buffs = {}

    # Buffers de envio (mantidos vivos pelo escopo da função até o wait)
    send_buffs = []

    # ==========================================
    # 1. X-Axis (Left / Right)
    # ==========================================
    # O PROBLEMA ERA AQUI: O slice em X não é contíguo.
    # Precisamos receber num buffer contíguo e copiar depois.

    if neighbors['left'] != -1:
        # Prepara Buffer de Recebimento
        rb_l = torch.empty_like(tensor[..., X_MIN]).contiguous()
        reqs.append(dist.irecv(rb_l, src=neighbors['left']))
        recv_buffs['left'] = rb_l

        # Envio
        sb_l = tensor[..., X_MIN + 1].contiguous()
        send_buffs.append(sb_l) # Mantém referência
        reqs.append(dist.isend(sb_l, dst=neighbors['left']))

    if neighbors['right'] != -1:
        # Prepara Buffer de Recebimento
        rb_r = torch.empty_like(tensor[..., X_MAX]).contiguous()
        reqs.append(dist.irecv(rb_r, src=neighbors['right']))
        recv_buffs['right'] = rb_r

        # Envio
        sb_r = tensor[..., X_MAX - 1].contiguous()
        send_buffs.append(sb_r)
        reqs.append(dist.isend(sb_r, dst=neighbors['right']))

    # ==========================================
    # 2. Y-Axis (Top / Bottom)
    # ==========================================
    # Slices em Y e Z geralmente são contíguos (dependendo do layout),
    # mas usar buffers padroniza e evita erros.

    if neighbors['top'] != -1:
        rb_t = torch.empty_like(tensor[..., Y_MIN, :]).contiguous()
        reqs.append(dist.irecv(rb_t, src=neighbors['top']))
        recv_buffs['top'] = rb_t

        sb_t = tensor[..., Y_MIN + 1, :].contiguous()
        send_buffs.append(sb_t)
        reqs.append(dist.isend(sb_t, dst=neighbors['top']))

    if neighbors['bottom'] != -1:
        rb_b = torch.empty_like(tensor[..., Y_MAX, :]).contiguous()
        reqs.append(dist.irecv(rb_b, src=neighbors['bottom']))
        recv_buffs['bottom'] = rb_b

        sb_b = tensor[..., Y_MAX - 1, :].contiguous()
        send_buffs.append(sb_b)
        reqs.append(dist.isend(sb_b, dst=neighbors['bottom']))

    # ==========================================
    # 3. Z-Axis - Inter nodes
    # ==========================================
    if neighbors['back'] != -1:
        rb_back = torch.empty_like(tensor[..., Z_MIN, :, :]).contiguous()
        reqs.append(dist.irecv(rb_back, src=neighbors['back']))
        recv_buffs['back'] = rb_back

        sb_back = tensor[..., Z_MIN + 1, :, :].contiguous()
        send_buffs.append(sb_back)
        reqs.append(dist.isend(sb_back, dst=neighbors['back']))

    if neighbors['front'] != -1:
        rb_front = torch.empty_like(tensor[..., Z_MAX, :, :]).contiguous()
        reqs.append(dist.irecv(rb_front, src=neighbors['front']))
        recv_buffs['front'] = rb_front

        sb_front = tensor[..., Z_MAX - 1, :, :].contiguous()
        send_buffs.append(sb_front)
        reqs.append(dist.isend(sb_front, dst=neighbors['front']))

    # Wait for all communications to complete
    for req in reqs:
        req.wait()

    # ==========================================
    # Copiar dados dos buffers de volta para o tensor (Halo Update)
    # ==========================================

    # X Axis
    if 'left' in recv_buffs:
        tensor[..., X_MIN] = recv_buffs['left']
    if 'right' in recv_buffs:
        tensor[..., X_MAX] = recv_buffs['right']

    # Y Axis
    if 'top' in recv_buffs:
        tensor[..., Y_MIN, :] = recv_buffs['top']
    if 'bottom' in recv_buffs:
        tensor[..., Y_MAX, :] = recv_buffs['bottom']

    # Z Axis
    if 'back' in recv_buffs:
        tensor[..., Z_MIN, :, :] = recv_buffs['back']
    if 'front' in recv_buffs:
        tensor[..., Z_MAX, :, :] = recv_buffs['front']

    return tensor
