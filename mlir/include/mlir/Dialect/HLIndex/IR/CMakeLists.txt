set(LLVM_TARGET_DEFINITIONS IndexEnums.td)
add_mlir_dialect(HLIndexOps hl_index)
add_mlir_doc(HLIndexOps HLIndexOps Dialects/ -gen-dialect-doc)
