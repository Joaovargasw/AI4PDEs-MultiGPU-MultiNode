import torch
import torch.distributed as dist
from torch.multiprocessing import Process, spawn
import os
#  +++++++++++++++++++++++++++++++++++++++++++++++++
#  +++++++++++++++++++++++++++++++++++++++++++++++++
#  +++++++++++++++++++++++++++++++++++++++++++++++++
def init_process(rank, size, backend='nccl'):
    os.environ['MASTER_ADDR'] = '127.0.0.1'
    os.environ['MASTER_PORT'] = '29501'
    dist.init_process_group(backend, rank=rank, world_size=size)

def split_and_rearrange_tensor(tensor):
    tensor = tensor.squeeze(dim=0).squeeze(dim=0)
    zert, rows, cols = tensor.size()
    top_left = tensor[: ,:rows // 2, :cols // 2]
    top_right = tensor[: ,:rows // 2, cols // 2:]
    bottom_left = tensor[: ,rows // 2:, :cols // 2]
    bottom_right = tensor[: ,rows // 2:, cols // 2:]
    return [top_left.unsqueeze(dim=0).unsqueeze(dim=0), top_right.unsqueeze(dim=0).unsqueeze(dim=0),\
             bottom_left.unsqueeze(dim=0).unsqueeze(dim=0), bottom_right.unsqueeze(dim=0).unsqueeze(dim=0)]

def structured_halo_update_3D(rank_id, input_data):
    input_data = input_data.squeeze(dim=0).squeeze(dim=0)
    if rank_id == 0:
        send_req  = dist.isend(tensor=input_data[: ,1:-1,-2].contiguous(), dst=1)
        recv_data = torch.zeros_like(input_data[: ,1:-1,-1].contiguous())
        recv_req  = dist.irecv(tensor=recv_data.contiguous(), src=1)
        send_req.wait()
        recv_req.wait()
        input_data[: ,1:-1, -1] = recv_data

        send_req1  = dist.isend(tensor=input_data[: ,-2, 1:-1].contiguous(), dst=2)
        recv_data1 = torch.zeros_like(input_data[: ,-2, 1:-1].contiguous())
        recv_req1  = dist.irecv(tensor=recv_data1.contiguous(), src=2)
        send_req1.wait()
        recv_req1.wait()
        input_data[: ,-1, 1:-1] = recv_data1

        send_req2  = dist.isend(tensor=input_data[: ,-2,-2].contiguous(), dst=3)
        recv_data2 = torch.zeros_like(input_data[: ,-2,-2].contiguous())
        recv_req2  = dist.irecv(tensor=recv_data2.contiguous(), src=3)
        send_req2.wait()
        recv_req2.wait()
        input_data[: ,-1, -1] = recv_data2

    elif rank_id == 1:
        recv_data = torch.zeros_like(input_data[: ,1:-1, 0].contiguous())
        recv_req  = dist.irecv(tensor=recv_data.contiguous(), src=0)
        send_req  = dist.isend(tensor=input_data[: ,1:-1, 1].contiguous(), dst=0)
        send_req.wait()
        recv_req.wait()
        input_data[: ,1:-1, 0] = recv_data


        send_req1  = dist.isend(tensor=input_data[: ,-2, 1:-1].contiguous(), dst=3)
        recv_data1 = torch.zeros_like(input_data[: ,-1, 1:-1].contiguous())
        recv_req1  = dist.irecv(tensor=recv_data1.contiguous(), src=3)
        send_req1.wait()
        recv_req1.wait()
        input_data[: ,-1, 1:-1] = recv_data1

        recv_data2  = torch.zeros_like(input_data[: ,-1,0].contiguous())
        recv_req2   = dist.irecv(tensor=recv_data2.contiguous(), src=2)
        send_req2   = dist.isend(tensor=input_data[: ,-2, 1].contiguous(), dst=2)
        send_req2.wait()
        recv_req2.wait()
        input_data[: ,-1,0] = recv_data2

    elif rank_id == 2:
        recv_data = torch.zeros_like(input_data[: ,0, 1:-1].contiguous())
        recv_req  = dist.irecv(tensor=recv_data.contiguous(), src=0)
        send_req  = dist.isend(tensor=input_data[: ,1, 1:-1].contiguous(), dst=0)
        send_req.wait()
        recv_req.wait()
        input_data[: ,0, 1:-1] = recv_data

        send_req1  = dist.isend(tensor=input_data[: ,1:-1, -2].contiguous(), dst=3)
        recv_data1 = torch.zeros_like(input_data[: ,1:-1, -2].contiguous())
        recv_req1  = dist.irecv(tensor=recv_data1.contiguous(), src=3)
        send_req1.wait()
        recv_req1.wait()
        input_data[: ,1:-1,-1] = recv_data1

        send_req2   = dist.isend(tensor=input_data[: ,1,-2].contiguous(), dst=1)
        recv_data2  = torch.zeros_like(input_data[: ,0, -1].contiguous())
        recv_req2   = dist.irecv(tensor=recv_data2.contiguous(), src=1)
        send_req2.wait()
        recv_req2.wait()
        input_data[: ,0, -1] = recv_data2

    elif rank_id == 3:
        recv_data = torch.zeros_like(input_data[: ,0,1:-1].contiguous())
        recv_req  = dist.irecv(tensor=recv_data.contiguous(), src=1)
        send_req  = dist.isend(tensor=input_data[: ,1,1:-1].contiguous(), dst=1)
        send_req.wait()
        recv_req.wait()
        input_data[:,0,1:-1] = recv_data

        recv_data1 = torch.zeros_like(input_data[: ,1:-1,0].contiguous())
        recv_req1  = dist.irecv(tensor=recv_data1.contiguous(), src=2)
        send_req1  = dist.isend(tensor=input_data[: ,1:-1,1].contiguous(), dst=2)
        send_req1.wait()
        recv_req1.wait()
        input_data[: ,1:-1,0] = recv_data1

        recv_data2 = torch.zeros_like(input_data[: ,0, 0].contiguous())
        recv_req2  = dist.irecv(tensor=recv_data2.contiguous(), src=0)
        send_req2  = dist.isend(tensor=input_data[: ,1, 1].contiguous(), dst=0)
        send_req2.wait()
        recv_req2.wait()
        input_data[: ,0, 0] = recv_data2
    return input_data.unsqueeze(dim=0).unsqueeze(dim=0)

# def gather_all_data_3D(rank,data,concatenated_data_on_cpu):
#     data = data.squeeze(dim=0).squeeze(dim=0)
#     concatenated_data_on_cpu = concatenated_data_on_cpu.squeeze(dim=0).squeeze(dim=0)
#     nz = concatenated_data_on_cpu.shape[0]
#     ny = concatenated_data_on_cpu.shape[1]
#     nx = concatenated_data_on_cpu.shape[2]

#     if rank == 0:
#         top_left = data
#         concatenated_data_on_cpu[: ,:ny // 2, :nx // 2] = top_left.cpu()
#     elif rank == 1:
#         top_right = data
#         concatenated_data_on_cpu[: ,:ny // 2, nx // 2:] = top_right.cpu()
#     elif rank == 2:
#         bottom_left = data
#         concatenated_data_on_cpu[: ,ny // 2:, :nx // 2] = bottom_left.cpu()
#     elif rank == 3:
#         bottom_right = data
#         concatenated_data_on_cpu[: ,ny // 2:, nx // 2:] = bottom_right.cpu()
#     return concatenated_data_on_cpu.unsqueeze(dim=0).unsqueeze(dim=0)


# Em exchange_3D_complete.py

def gather_all_data_3D(rank, local_tensor, ignored_cpu_buffer=None):
    """
    Coleta dados de uma decomposição 2x2 (Y-X) para o Rank 0.
    Ignora o buffer de CPU antigo e cria um novo tensor completo.
    """
    # 1. Garante que o tensor está na GPU e contíguo
    tensor_gpu = local_tensor.contiguous()

    # 2. Prepara a lista de recebimento no Rank 0
    world_size = dist.get_world_size()
    if rank == 0:
        gather_list = [torch.empty_like(tensor_gpu) for _ in range(world_size)]
    else:
        gather_list = None

    # 3. Executa a comunicação (Sincronização implícita)
    dist.gather(tensor_gpu, gather_list=gather_list, dst=0)

    # 4. Reconstrói o domínio global no Rank 0
    if rank == 0:
        # A topologia é:
        # Rank 0 (Top-Left), Rank 1 (Top-Right)
        # Rank 2 (Bot-Left), Rank 3 (Bot-Right)

        # Pega as peças (ainda na GPU)
        top_left = gather_list[0]
        top_right = gather_list[1]
        bot_left = gather_list[2]
        bot_right = gather_list[3]

        # Concatena em X (Dimensão 4)
        top = torch.cat((top_left, top_right), dim=4)
        bot = torch.cat((bot_left, bot_right), dim=4)

        # Concatena em Y (Dimensão 3)
        full_domain = torch.cat((top, bot), dim=3)

        return full_domain # Retorna tensor na GPU

    return local_tensor # Ranks > 0 retornam o local apenas para manter o fluxo
