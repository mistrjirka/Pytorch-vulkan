#pragma once
#include <c10/core/Device.h>
#include <c10/util/Exception.h>
namespace torch_vulkan {
inline void validate_vulkan_device(c10::Device device) {
  TORCH_CHECK(device.type()==c10::DeviceType::PrivateUse1 &&
              (device.index()==-1 || device.index()==0),
              "Vulkan backend exposes only logical device 0; requested ", device);
}
} // namespace torch_vulkan
