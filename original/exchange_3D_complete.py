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

def gather_all_data_3D(rank,data,concatenated_data_on_cpu):
    data = data.squeeze(dim=0).squeeze(dim=0)
    concatenated_data_on_cpu = concatenated_data_on_cpu.squeeze(dim=0).squeeze(dim=0)
    nz = concatenated_data_on_cpu.shape[0]
    ny = concatenated_data_on_cpu.shape[1]
    nx = concatenated_data_on_cpu.shape[2]

    if rank == 0:
        top_left = data
        concatenated_data_on_cpu[: ,:ny // 2, :nx // 2] = top_left.cpu()
    elif rank == 1:
        top_right = data
        concatenated_data_on_cpu[: ,:ny // 2, nx // 2:] = top_right.cpu()
    elif rank == 2:
        bottom_left = data
        concatenated_data_on_cpu[: ,ny // 2:, :nx // 2] = bottom_left.cpu()
    elif rank == 3:
        bottom_right = data
        concatenated_data_on_cpu[: ,ny // 2:, nx // 2:] = bottom_right.cpu()
    return concatenated_data_on_cpu.unsqueeze(dim=0).unsqueeze(dim=0)