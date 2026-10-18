// Exercise real queue ownership, timeout, fence reuse, and pending teardown.
#include <kompute/Sequence.hpp>
#include <kompute/FencePool.hpp>
#include <chrono>
#include <thread>
#include <iostream>

static void check(bool ok, const char* message) {
    if (!ok) throw std::runtime_error(message);
}
class WaitEvent : public kp::OpBase {
    vk::Event event_;
public:
    explicit WaitEvent(vk::Event event): event_(event) {}
    void record(const vk::CommandBuffer& command) override {
        command.waitEvents(1, &event_, vk::PipelineStageFlagBits::eHost,
                           vk::PipelineStageFlagBits::eAllCommands,
                           0, nullptr, 0, nullptr, 0, nullptr);
    }
    void preEval(const vk::CommandBuffer&) override {}
    void postEval(const vk::CommandBuffer&) override {}
};
int main(int argc, char** argv) {
    vk::ApplicationInfo app("fence-lifecycle", 1, "tests", 1, VK_API_VERSION_1_1);
    auto instance=vk::createInstance(vk::InstanceCreateInfo({}, &app));
    auto physical=std::make_shared<vk::PhysicalDevice>(instance.enumeratePhysicalDevices().at(argc>1 ? std::stoul(argv[1]) : 0));
    auto families=physical->getQueueFamilyProperties();
    uint32_t family=0;
    while (family<families.size() && !(families[family].queueFlags & vk::QueueFlagBits::eCompute)) ++family;
    check(family<families.size(), "No compute queue");
    float priority=1;
    vk::DeviceQueueCreateInfo queueInfo({}, family, 1, &priority);
    auto device=std::make_shared<vk::Device>(physical->createDevice(vk::DeviceCreateInfo({}, 1, &queueInfo)));
    auto queue=std::make_shared<vk::Queue>(device->getQueue(family,0));
    auto pool=std::make_shared<kp::FencePool>(device);
    auto make=[&](vk::Event event) {
        auto seq=std::make_shared<kp::Sequence>(physical,device,queue,family,0,pool);
        seq->record(std::make_shared<WaitEvent>(event));
        return seq;
    };
    auto event=device->createEvent({});
    auto first=make(event), second=make(event);
    first->evalAsync(); second->evalAsync();
    first->evalAwait(0); second->evalAwait(0);
    check(first->isRunning() && second->isRunning(), "Timeout dropped pending ownership");
    bool rejected=false;
    try { first->clear(); } catch(const std::runtime_error&) { rejected=true; }
    check(rejected,"Clear accepted in-flight resources");
    device->setEvent(event);
    second->evalAwait(); first->evalAwait();
    check(!first->isRunning() && !second->isRunning(), "Completed sequences still running");
    first.reset(); second.reset();
    // Many distinct sequences must safely reuse the completed fences.
    for(int i=0;i<64;++i) { auto seq=make(event); seq->evalAsync(); seq->evalAwait(); }
    device->resetEvent(event);
    auto pending=make(event); pending->evalAsync(); pending->evalAwait(0);
    check(pending->isRunning(),"Expected blocked queue");
    std::thread signal([&] { std::this_thread::sleep_for(std::chrono::milliseconds(20)); device->setEvent(event); });
    pending.reset(); // Must wait for the host event before freeing command resources.
    signal.join();
    device->waitIdle(); device->destroyEvent(event);
    pool->shutdown(); pool.reset(); queue.reset(); device->destroy(); instance.destroy();
    std::cout << "Fence lifecycle passed: concurrent leases, timeout, reuse, pending destruction\n";
}
