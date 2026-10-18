"""Small independent reproductions separating OpInfo findings from harness limits."""
import argparse,json,subprocess
from pathlib import Path
import torch
import torch_vulkan as vk

p=argparse.ArgumentParser(description=__doc__);p.add_argument('--out',type=Path,required=True);a=p.parse_args()
torch.set_num_threads(4)

def tensor(shape,device):return torch.arange(torch.tensor(shape).prod().item(),dtype=torch.float32).reshape(shape).to(device)/7

def addmm(device,beta=.2):
    bias=tensor((2,3),device)+1
    return torch.addmm(bias,tensor((2,2),device),tensor((2,3),device),alpha=.6,beta=beta)

def partial_view(device):
    base=torch.arange(20.,device='cpu').to(device)
    return base[5:15].as_strided((2,2),(1,2),storage_offset=0)

def upstream_copy_reference(device):
    base=torch.arange(20.,device='cpu').to(device)
    view=base[5:15]
    # Exactly the reference conversion used by upstream test_compare_cpu.
    return view.to('cpu').as_strided((2,2),(1,2),storage_offset=0)

cases={
 'addmm_scaled_matrix_bias':lambda d:addmm(d),
 'addmm_beta_one_control':lambda d:addmm(d,1.),
 'empty_inner_mm':lambda d:torch.empty(5,0,device=d)@torch.empty(0,10,device=d),
 'transpose_vector':lambda d:torch.arange(5.,device='cpu').to(d).t(),
 'empty_uniform':lambda d:torch.empty((2,0,3,5,0,1),device=d).uniform_(),
 'bool_rng':lambda d:torch.randint(0,1,(8,),dtype=torch.bool,device=d),
 'float_integer_rng':lambda d:torch.empty(8,device=d).random_(0,1),
 'advanced_indexing':lambda d:torch.arange(12.,device='cpu').reshape(3,4).to(d)[torch.tensor([2,0],device=d)],
 'tensor_equality':lambda d:torch.arange(5.,device='cpu').to(d)==torch.arange(5.,device='cpu').to(d),
 'partial_view_storage_offset':partial_view,
 'upstream_partial_view_cpu_reference':upstream_copy_reference,
}
report={'backend_commit':subprocess.check_output(['git','rev-parse','HEAD'],text=True).strip(),'device':vk.device_name(),'torch':torch.__version__,'cases':{}}
for name,fn in cases.items():
    row={}
    try:expected=fn('cpu');row['cpu_status']='passed'
    except Exception as e:row.update(cpu_status='error',cpu_error=str(e));report['cases'][name]=row;continue
    vk.reset_execution_stats()
    try:
        got=fn('vulkan');row['execution']=vk.execution_stats();actual=got.cpu()
        row['actual']=actual.tolist();row['expected']=expected.tolist()
        torch.testing.assert_close(actual,expected,rtol=1e-4,atol=1e-5)
        row['vulkan_status']='passed'
    except AssertionError as e:row.update(vulkan_status='mismatch',error=str(e))
    except Exception as e:row.update(vulkan_status='error',error=str(e))
    report['cases'][name]=row
    print(name,row['vulkan_status'],row.get('error','').split('\n')[0],flush=True)
a.out.parent.mkdir(parents=True,exist_ok=True);a.out.write_text(json.dumps(report,indent=2)+'\n')
