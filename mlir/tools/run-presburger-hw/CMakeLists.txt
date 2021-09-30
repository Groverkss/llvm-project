set (LIBS
)

add_llvm_tool(run-presburger-hw
  run-presburger-hw.cpp

  DEPENDS
  ${LIBS}
  )

target_link_libraries(run-presburger-hw PRIVATE ${LIBS} /usr/lib64/clang/11/lib/linux/libclang_rt.builtins-x86_64.a "/home/kunwar/.local/lib/libgmpxx.so")

llvm_update_compile_flags(run-presburger-hw)
mlir_check_all_link_libraries(run-presburger-hw)
