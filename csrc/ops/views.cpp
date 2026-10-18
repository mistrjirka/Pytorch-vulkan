#include "native_utils.h"
#include <ATen/NativeFunctions.h>
#include <ATen/native/Resize.h>
#include <torch/library.h>
namespace torch_vulkan {
void resize_vulkan_storage(c10::StorageImpl*,size_t);
namespace ops {
namespace {
at::Tensor& set_storage(at::Tensor& out,at::Storage storage,int64_t offset,at::IntArrayRef sizes,at::IntArrayRef strides){
 at::native::checkSetStorage(out,storage,offset,sizes,strides);
 auto* impl=out.unsafeGetTensorImpl();impl->set_storage_offset(offset);
 if(strides.empty())impl->set_sizes_contiguous(sizes);else impl->set_sizes_and_strides(sizes,strides);
 auto bytes=at::detail::computeStorageNbytes(out.sizes(),out.strides(),out.element_size(),offset);
 if(bytes>out.storage().nbytes())resize_vulkan_storage(out.storage().unsafeGetStorageImpl(),bytes);
 return out;
}
at::Tensor& set_tensor(at::Tensor& out,const at::Tensor& source){
 TORCH_CHECK(out.scalar_type()==source.scalar_type(),"Could not set tensor of type ",out.scalar_type()," to a tensor of type ",source.scalar_type());
 return set_storage(out,source.storage(),source.storage_offset(),source.sizes(),source.strides());
}
at::Tensor& set_plain_storage(at::Tensor& out,at::Storage source){auto n=source.nbytes()/out.element_size();return set_storage(out,std::move(source),0,{int64_t(n)},{1});}
}
TORCH_LIBRARY_IMPL(aten,PrivateUse1,m){m.impl("unfold",&at::native::unfold);m.impl("view.dtype",&at::native::view_dtype);m.impl("set_.source_Tensor",&set_tensor);m.impl("set_.source_Storage",&set_plain_storage);m.impl("set_.source_Storage_storage_offset",&set_storage);}
}
}
