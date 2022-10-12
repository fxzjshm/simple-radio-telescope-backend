# Simple radio telescope backend
Everything working in progress...

### Workarounds
#### 1. BOOST_INLINE and HIP conflicts
See https://github.com/boostorg/config/issues/392 , which means if compile for ROCm, Boost 1.80+ may be needed.

#### 2. configure error: "clangrt builtins lib not found"
If compile with intel/llvm, HIP may search 'clang_rt.builtins' in intel/llvm, but this module isn't built by default. A patch to `buildbot/configure.py` is
```diff
diff --git a/buildbot/configure.py b/buildbot/configure.py
index f3a43857b7..08cb75e5e3 100644
--- a/buildbot/configure.py
+++ b/buildbot/configure.py
@@ -13,7 +13,7 @@ def do_configure(args):
     if not os.path.isdir(abs_obj_dir):
       os.makedirs(abs_obj_dir)
 
-    llvm_external_projects = 'sycl;llvm-spirv;opencl;xpti;xptifw'
+    llvm_external_projects = 'sycl;llvm-spirv;opencl;xpti;xptifw;compiler-rt'
 
     # libdevice build requires a working SYCL toolchain, which is not the case
     # with macOS target right now.
```
Also add `openmp` if needed, e.g. use intel/llvm as a compiler for hipSYCL
