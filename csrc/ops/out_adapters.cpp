// Adapt out= variants to existing native kernels with shared alias/cast checks.
// Register only kernels with a direct PrivateUse1 functional implementation:
// adapting composite functions that call their own out variant would recurse.
#include "native_utils.h"
#include <ATen/MemoryOverlap.h>
#include <ATen/native/Resize.h>
#include <torch/library.h>
namespace torch_vulkan::ops {
namespace {
struct NativeOut : c10::OperatorKernel {
 std::string name,overload;
 NativeOut(std::string n,std::string o):name(std::move(n)),overload(std::move(o)){}
 void operator()(const c10::OperatorHandle& op,c10::DispatchKeySet,torch::jit::Stack* stack){
  auto count=op.schema().arguments().size();auto begin=stack->size()-count;
  auto out=stack->back().toTensor();at::assert_no_internal_overlap(out);
  torch::jit::Stack args(stack->begin()+begin,stack->end()-1);
  for(const auto& v:args)if(v.isTensor()){
   auto t=v.toTensor();at::assert_no_partial_overlap(out,t);
   TORCH_CHECK(t.device()==out.device()||(t.is_cpu()&&t.dim()==0),"out tensor must be on the same device as the input");
  }
  for(const auto& value:args)if(value.isTensorList())for(const auto& t:value.toTensorVector()){
   at::assert_no_partial_overlap(out,t);TORCH_CHECK(t.device()==out.device(),"out tensor must be on the input device");
  }
  const auto target=c10::Dispatcher::singleton().findSchemaOrThrow(name.c_str(),overload.c_str());
  TORCH_INTERNAL_ASSERT(target.schema().arguments().size()==args.size());
  target.callBoxed(&args);
  TORCH_INTERNAL_ASSERT(args.size()==1&&args[0].isTensor());auto value=args[0].toTensor();
  TORCH_CHECK(c10::canCast(value.scalar_type(),out.scalar_type()),"result type ",value.scalar_type()," can't be cast to the desired output type ",out.scalar_type());
  at::native::resize_output(out,value.sizes());out.copy_(value);
  stack->resize(begin);stack->push_back(out);
 }
};
}
TORCH_LIBRARY_IMPL(aten,PrivateUse1,m){
 auto add=[&](const char* out,const char* name,const char* overload=""){
  m.impl(out,torch::CppFunction::makeFromBoxedFunctor(std::make_unique<NativeOut>(std::string("aten::")+name,overload)));
 };
 for(auto name:{"abs","sgn","exp","sqrt","rsqrt","sin","cos","tanh","sigmoid","neg","silu","gelu","mm","bmm","addmm","clamp"})add((std::string(name)+".out").c_str(),name);
 add("bitwise_and.Tensor_out","bitwise_and","Tensor");add("bitwise_or.Tensor_out","bitwise_or","Tensor");add("bitwise_xor.Tensor_out","bitwise_xor","Tensor");add("bitwise_not.out","bitwise_not");
 add("bitwise_and.Scalar_out","bitwise_and","Scalar");add("bitwise_or.Scalar_out","bitwise_or","Scalar");add("bitwise_xor.Scalar_out","bitwise_xor","Scalar");
 for(auto name:{"reciprocal","log","log1p","expm1","floor","ceil","trunc","round","sign","log10","log2","exp2","sinh","cosh","tan","asin","acos","atan","asinh","acosh","atanh","erf","erfc"})add((std::string(name)+".out").c_str(),name);
 add("round.decimals_out","round","decimals");
 add("cat.out","cat");
 for(auto name:{"clamp_min","clamp_max"}){add((std::string(name)+".out").c_str(),name);add((std::string(name)+".Tensor_out").c_str(),name,"Tensor");}
 add("clamp.Tensor_out","clamp","Tensor");add("hardtanh.out","hardtanh");
 add("masked_fill.Scalar_out","masked_fill","Scalar");add("masked_fill.Tensor_out","masked_fill","Tensor");
 add("any.out","any","dim");add("any.all_out","any");add("any.dims_out","any","dims");add("all.dims_out","all","dims");
 add("rsub.Tensor_out","rsub","Tensor");add("rsub.Scalar_out","rsub","Scalar");add("addmv.out","addmv");
 add("add.out","add","Tensor");add("sub.out","sub","Tensor");add("mul.out","mul","Tensor");add("div.out","div","Tensor");
 add("pow.Tensor_Scalar_out","pow","Tensor_Scalar");
 add("sum.IntList_out","sum","dim_IntList");add("sum.out","sum");
 add("mean.out","mean","dim");add("mean.dtype_out","mean");
 add("all.out","all","dim");add("all.all_out","all");
 add("cumsum.out","cumsum");add("flip.out","flip");
}
}
