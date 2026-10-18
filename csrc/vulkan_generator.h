#pragma once
#include <ATen/core/Generator.h>
#include <c10/core/GeneratorImpl.h>
#include <array>
namespace torch_vulkan {
class VulkanGenerator final:public c10::GeneratorImpl {
 public:
 explicit VulkanGenerator(uint64_t seed=c10::default_rng_seed_val);
 static c10::DeviceType device_type(){return c10::DeviceType::PrivateUse1;}
 void set_current_seed(uint64_t value) override {seed_=value;offset_=0;}
 uint64_t current_seed()const override{return seed_;}
 void set_offset(uint64_t value)override{offset_=value;}
 uint64_t get_offset()const override{return offset_;}
 uint64_t seed()override;
 void set_state(const c10::TensorImpl&)override;
 c10::intrusive_ptr<c10::TensorImpl> get_state()const override;
 std::array<uint32_t,4> reserve(uint64_t blocks);
 private:
 VulkanGenerator* clone_impl()const override;
 uint64_t seed_,offset_=0;
};
const at::Generator& default_generator();
at::Generator new_generator(c10::DeviceIndex index=-1);
std::array<uint32_t,4> random_state(const std::optional<at::Generator>& generator,uint64_t blocks);
}
