#include "vulkan_generator.h"
#include <ATen/Functions.h>
#include <cstring>
#include <limits>
namespace torch_vulkan {
VulkanGenerator::VulkanGenerator(uint64_t seed):c10::GeneratorImpl(c10::Device(c10::DeviceType::PrivateUse1,0),c10::DispatchKeySet(c10::DispatchKey::PrivateUse1)),seed_(seed){}
uint64_t VulkanGenerator::seed(){auto value=c10::detail::getNonDeterministicRandom();set_current_seed(value);return value;}
VulkanGenerator* VulkanGenerator::clone_impl()const{auto* g=new VulkanGenerator(seed_);g->offset_=offset_;return g;}
c10::intrusive_ptr<c10::TensorImpl> VulkanGenerator::get_state()const{
 auto t=at::empty({24},at::TensorOptions().dtype(at::kByte).device(at::kCPU));uint64_t state[3]={1,seed_,offset_};std::memcpy(t.data_ptr(),state,24);return t.getIntrusivePtr();
}
void VulkanGenerator::set_state(const c10::TensorImpl& t){
 TORCH_CHECK(t.device_type()==c10::DeviceType::CPU&&t.dtype()==caffe2::TypeMeta::Make<uint8_t>()&&t.numel()==24&&t.is_contiguous(),"Vulkan RNG state must be a contiguous 24-byte CPU tensor");
 uint64_t state[3];std::memcpy(state,t.data(),24);TORCH_CHECK(state[0]==1,"Unsupported Vulkan RNG state version");seed_=state[1];offset_=state[2];
}
std::array<uint32_t,4> VulkanGenerator::reserve(uint64_t blocks){
 TORCH_CHECK(blocks<=std::numeric_limits<uint64_t>::max()-offset_,"Vulkan RNG counter exhausted");
 std::array<uint32_t,4> result={uint32_t(seed_),uint32_t(seed_>>32),uint32_t(offset_),uint32_t(offset_>>32)};offset_+=blocks;return result;
}
at::Generator new_generator(c10::DeviceIndex index){TORCH_CHECK(index==-1||index==0,"Vulkan generator supports only device 0");return at::make_generator<VulkanGenerator>();}
const at::Generator& default_generator(){static const at::Generator g=at::make_generator<VulkanGenerator>(c10::detail::getNonDeterministicRandom());return g;}
std::array<uint32_t,4> random_state(const std::optional<at::Generator>& generator,uint64_t blocks){
 auto g=generator.value_or(default_generator());TORCH_CHECK(g.device()==c10::Device(c10::DeviceType::PrivateUse1,0),"Expected a Vulkan generator");
 std::lock_guard<std::mutex> lock(g.mutex());return g.get<VulkanGenerator>()->reserve(blocks);
}
}
