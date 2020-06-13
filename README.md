# vulkan-tutorial-test
This project was created to test an issue I experienced with creating a command buffer for bit blitting.
The code found in master is copied from https://github.com/matthew-russo/vulkan-tutorial-rs/blob/29_multisampling/src/bin/28_generating_mipmaps.rs and functions correctly on my system, and uses Vulkano 0.11.1. The branch`version_0-19-0`uses the latet versions of all dependencies, including Vulkano 0.19.0, and panics on a BlitImageError. The same behavior also occurs on version 0.18.0.

## My Specs
* OS: Manjaro Linux (Kernel version: 5.6.15-1-MANJARO)
* GPU: AMD Radeon RX580 
* GPU Driver: VGA compatible controller: Advanced Micro Devices, Inc. [AMD/ATI] Ellesmere [Radeon RX 470/480/570/570X/580/580X/590] (rev e7) (prog-if 00 [VGA controller]), kernel drivers and modules: amdgpu)

## Stacktrace
```thread 'main' panicked at 'called `Result::unwrap()` on an `Err` value: SyncCommandBufferBuilderError(Conflict { command1_name: "vkCmdBlitImage", command1_param: "source", command1_offset: 1, command2_name: "vkCmdBlitImage", command2_param: "destination", command2_offset: 1 })', src/main.rs:667:13
stack backtrace:
   0: backtrace::backtrace::libunwind::trace
             at /cargo/registry/src/github.com-1ecc6299db9ec823/backtrace-0.3.44/src/backtrace/libunwind.rs:86
   1: backtrace::backtrace::trace_unsynchronized
             at /cargo/registry/src/github.com-1ecc6299db9ec823/backtrace-0.3.44/src/backtrace/mod.rs:66
   2: std::sys_common::backtrace::_print_fmt
             at src/libstd/sys_common/backtrace.rs:78
   3: <std::sys_common::backtrace::_print::DisplayBacktrace as core::fmt::Display>::fmt
             at src/libstd/sys_common/backtrace.rs:59
   4: core::fmt::write
             at src/libcore/fmt/mod.rs:1063
   5: std::io::Write::write_fmt
             at src/libstd/io/mod.rs:1426
   6: std::sys_common::backtrace::_print
             at src/libstd/sys_common/backtrace.rs:62
   7: std::sys_common::backtrace::print
             at src/libstd/sys_common/backtrace.rs:49
   8: std::panicking::default_hook::{{closure}}
             at src/libstd/panicking.rs:204
   9: std::panicking::default_hook
             at src/libstd/panicking.rs:224
  10: std::panicking::rust_panic_with_hook
             at src/libstd/panicking.rs:470
  11: rust_begin_unwind
             at src/libstd/panicking.rs:378
  12: core::panicking::panic_fmt
             at src/libcore/panicking.rs:85
  13: core::option::expect_none_failed
             at src/libcore/option.rs:1211
  14: core::result::Result<T,E>::unwrap
             at /rustc/8d69840ab92ea7f4d323420088dd8c9775f180cd/src/libcore/result.rs:1003
  15: vulkan_tutorial::Renderer::create_texture_image
             at src/main.rs:667
  16: vulkan_tutorial::Renderer::initialize
             at src/main.rs:282
  17: vulkan_tutorial::HelloTriangleApplication::initialize
             at src/main.rs:1057
  18: vulkan_tutorial::main
             at src/main.rs:1069
  19: std::rt::lang_start::{{closure}}
             at /rustc/8d69840ab92ea7f4d323420088dd8c9775f180cd/src/libstd/rt.rs:67
  20: std::rt::lang_start_internal::{{closure}}
             at src/libstd/rt.rs:52
  21: std::panicking::try::do_call
             at src/libstd/panicking.rs:303
  22: __rust_maybe_catch_panic
             at src/libpanic_unwind/lib.rs:86
  23: std::panicking::try
             at src/libstd/panicking.rs:281
  24: std::panic::catch_unwind
             at src/libstd/panic.rs:394
  25: std::rt::lang_start_internal
             at src/libstd/rt.rs:51
  26: std::rt::lang_start
             at /rustc/8d69840ab92ea7f4d323420088dd8c9775f180cd/src/libstd/rt.rs:67
  27: main
  28: __libc_start_main
  29: _start
note: Some details are omitted, run with `RUST_BACKTRACE=full` for a verbose backtrace.
```
