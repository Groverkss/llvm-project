set(LLVM_TARGET_DEFINITIONS HLIndexAttributes.td)
mlir_tablegen(HLIndexAttributes.h.inc -gen-attrdef-decls)
mlir_tablegen(HLIndexAttributes.cpp.inc -gen-attrdef-defs)
add_public_tablegen_target(MLIRHLIndexAttributesIncGen)

add_mlir_dialect(HLIndexOps hl_index)
add_mlir_doc(HLIndexOps HLIndexOps Dialects/ -gen-dialect-doc)
