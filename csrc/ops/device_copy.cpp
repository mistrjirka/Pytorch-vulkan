#include "device_copy.h"
#include "native_utils.h"
#include <ATen/ExpandUtils.h>
#include <ATen/MemoryOverlap.h>
namespace torch_vulkan::ops {
using namespace native;
bool device_copy_supported(at::ScalarType t){return t==at::kFloat||t==at::kInt||t==at::kLong||t==at::kBool;}
static uint32_t type_code(at::ScalarType t){return t==at::kBool?0:t==at::kFloat?1:t==at::kInt?2:3;}
bool copy_on_device(at::Tensor& out,const at::Tensor& input){
 if(out.device().type()!=c10::DeviceType::PrivateUse1||input.device()!=out.device()||!device_copy_supported(out.scalar_type())||!device_copy_supported(input.scalar_type()))return false;
 auto src=input.expand(out.sizes());
 if(!out.numel())return true;
 TORCH_CHECK(can_pack_view(out)&&can_pack_view(src),"device copy exceeds Vulkan indexing/storage limits");
 at::assert_no_internal_overlap(out);at::assert_no_partial_overlap(out,src);
 // One invocation owns an entire destination word. Invert non-overlapping
 // destination strides to preserve holes and packed Boolean neighbours.
 std::vector<int> axes;for(int d=0;d<out.dim();d++)if(out.size(d)>1)axes.push_back(d);
 std::sort(axes.begin(),axes.end(),[&](int a,int b){return out.stride(a)<out.stride(b);});
 uint64_t extent=1;for(int d:axes){TORCH_CHECK(uint64_t(out.stride(d))>=extent,"device copy requires non-overlapping destination strides");extent+=uint64_t(out.size(d)-1)*out.stride(d);}
 if(out.storage().is_alias_of(src.storage()))src=src.clone();
 std::reverse(axes.begin(),axes.end());
 const auto bytes=out.element_size();uint64_t start=out.storage_offset()*bytes/4,end=((out.storage_offset()+extent)*bytes+3)/4;
 std::vector<uint32_t> p={u32(end-start),u32(start),type_code(src.scalar_type()),type_code(out.scalar_type()),u32(src.storage_offset()),u32(out.storage_offset()),u32(axes.size())};p.resize(32,0);p[31]=uint32_t(src.is_neg())|(uint32_t(out.is_neg())<<1);
 for(size_t j=0;j<axes.size();j++){int d=axes[j];p[7+j]=u32(out.size(d));p[15+j]=u32(src.stride(d));p[23+j]=u32(out.stride(d));}
 run("device_copy",{base(src,src.scalar_type()),base(out,out.scalar_type())},{write_base(out,out.scalar_type())},p,(end-start+127)/128);
 return true;
}
}
