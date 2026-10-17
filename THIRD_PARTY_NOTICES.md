# Third-party notices

Initial Vulkan integration, operator registration, allocator, compute shaders,
Python package and build scaffolding are adapted from:

- **torch-vulkan** by Peter Clemente III, <https://github.com/Peterc3-dev/torch-vulkan>
- Starting revision: `a344d771ff6e6bdac55413470cf1ebc6d37608b7` (2026-05-29)
- License: MIT (see [`LICENSE`](LICENSE), including original copyright).

The third-party dependency [Kompute](https://github.com/KomputeProject/kompute)
(version v0.9.0) is fetched during CMake configuration and is licensed under
**Apache 2.0**. Its upstream copyright and NOTICE obligations apply separately. Other dependencies include PyTorch (BSD-style),
Vulkan headers/loader (Apache 2.0 / MIT-style licenses), and their own dependencies.
Do not assume this notice is an exhaustive inventory of dependencies.
