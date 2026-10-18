"""Inductor external-kernel backend for the Vulkan dispatcher.

Dynamo and AOTAutograd capture forward/backward; Inductor schedules external
ATen calls into native Vulkan kernels. This first integration does not generate
new fused SPIR-V. CPU and other accelerator lowerings retain their own codegen.
"""
import contextlib
import functools
import torch
from torch._dynamo.device_interface import DeviceInterface,register_interface_for_device
from torch._inductor.codegen.common import register_backend_for_device,register_device_op_overrides,DeviceOpOverrides
from torch._inductor.codegen.wrapper import PythonWrapperCodegen
from torch._inductor.scheduler import BaseScheduling
from torch._inductor import lowering
from torch.utils._pytree import tree_leaves
import torch_vulkan as vk

class VulkanInterface(DeviceInterface):
    device=staticmethod(lambda d:contextlib.nullcontext())
    Event=vk.Event
    Stream=vk.Stream
    current_device=staticmethod(vk.current_device)
    set_device=staticmethod(vk.set_device)
    device_count=staticmethod(vk.device_count)
    is_available=staticmethod(vk.is_available)
    stream=staticmethod(vk.stream)
    current_stream=staticmethod(vk.current_stream)
    set_stream=staticmethod(vk.set_stream)
    synchronize=staticmethod(vk.synchronize)
    exchange_device=staticmethod(lambda d:(vk.set_device(d),0)[1])
    maybe_exchange_device=exchange_device
    get_raw_stream=staticmethod(lambda device_idx:vk.current_stream(device_idx).stream_id)
    get_compute_capability=staticmethod(lambda device=None:'vulkan-fp32')
    is_bf16_supported=staticmethod(lambda including_emulation=False:False)
    _set_stream_by_id=staticmethod(lambda stream_id,device_index,device_type:vk._C._set_stream_id(stream_id))
    class Worker:
        set_device=staticmethod(vk.set_device)
        current_device=staticmethod(lambda:0)
        get_device_properties=staticmethod(lambda device=None:None)

class VulkanScheduling(BaseScheduling):
    def can_fuse_vertical(self,*args):return False
    def can_fuse_horizontal(self,*args):return False
    def group_fn(self,sizes):return tuple(tuple(x) for x in sizes)
    def flush(self):pass
    def codegen_sync(self):
        from torch._inductor.virtualized import V
        V.graph.wrapper_code.writeline('torch.vulkan.synchronize()')
    def codegen_node(self,node):
        raise NotImplementedError('Vulkan Inductor expected an external device kernel')

class VulkanOverrides(DeviceOpOverrides):
    def import_get_raw_stream_as(self,name):return f'from torch_vulkan.inductor import get_raw_stream as {name}'
    def set_device(self,index):return f'torch.vulkan.set_device({index})'
    def synchronize(self):return 'torch.vulkan.synchronize()'
    def device_guard(self,index):return '__import__("contextlib").nullcontext()'
    def current_stream(self):return 'torch.vulkan.current_stream()'
    def stream_handle(self,name):return f'{name}.stream_id'

def get_raw_stream(index):return vk.current_stream(index).stream_id

register_interface_for_device('vulkan',VulkanInterface)
register_interface_for_device('vulkan:0',VulkanInterface)
from torch._inductor.custom_graph_pass import CustomGraphModulePass,get_hash_for_files

class VulkanGraphPass(CustomGraphModulePass):
    def uuid(self):
        return get_hash_for_files((__file__,))
    def __call__(self,gm):
        # Generated CPU/Triton seed nodes have no purpose for external Vulkan
        # kernels. Restore native ATen RNG factories on this device only.
        prims=torch.ops.prims
        random_ops=(prims.inductor_random.default,prims.inductor_randint.default)
        seed_ops=(prims.inductor_seed.default,prims.inductor_seeds.default,prims.inductor_lookup_seed.default)
        for node in list(gm.graph.nodes):
            value=node.meta.get('val')
            if node.op!='call_function' or not isinstance(value,torch.Tensor) or value.device.type!='vulkan':continue
            if node.target==random_ops[0]:
                size,seed,mode=node.args
                node.target=getattr(torch.ops.aten,mode).default;node.args=(size,);node.kwargs={'device':value.device,'dtype':torch.float32}
            elif node.target==random_ops[1]:
                low,high,size,seed=node.args
                node.target=torch.ops.aten.randint.low;node.args=(low,high,size);node.kwargs={'device':value.device,'dtype':torch.int64}
        for node in reversed(list(gm.graph.nodes)):
            value=node.meta.get('val')
            if node.op=='call_function' and node.target in seed_ops and not node.users and isinstance(value,torch.Tensor) and value.device.type=='vulkan':gm.graph.erase_node(node)
        gm.graph.lint();gm.recompile()

register_backend_for_device('vulkan',VulkanScheduling,PythonWrapperCodegen,device_custom_pass=VulkanGraphPass())
register_device_op_overrides('vulkan',VulkanOverrides())

def _vulkan_args(args,kwargs):
    for value in tree_leaves((args,kwargs)):
        if isinstance(value,torch.device) and value.type=='vulkan':return True
        if hasattr(value,'get_device'):
            device=value.get_device()
            if device is not None and device.type=='vulkan':return True
    return False

def _device_lowering(op,original):
    @functools.wraps(original)
    def dispatch(*args,**kwargs):
        if _vulkan_args(args,kwargs):
            return lowering.fallback_handler(op,add_to_fallback_set=False)(*args,**kwargs)
        return original(*args,**kwargs)
    return dispatch

# Device-selective adapters use Inductor's supported external-kernel IR. They do
# not change global compile flags, suppress errors, or redirect torch.compile.
for _op,_fn in list(lowering.lowerings.items()):
    if isinstance(_op,torch._ops.OpOverload):
        lowering.lowerings[_op]=_device_lowering(_op,_fn)

