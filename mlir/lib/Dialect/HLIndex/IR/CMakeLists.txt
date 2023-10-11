add_mlir_dialect_library(HLIndexDialect
  HLIndexOps.cpp

  ADDITIONAL_HEADER_DIRS
  ${MLIR_MAIN_INCLUDE_DIR}/mlir/Dialect/HLIndex

  DEPENDS
  MLIRHLIndexOpsIncGen

  LINK_LIBS PUBLIC
  MLIRArithDialect
  MLIRDialectUtils
  MLIRIR
  MLIRLoopLikeInterface
  MLIRMemRefDialect
  MLIRShapedOpInterfaces
  MLIRSideEffectInterfaces
  MLIRValueBoundsOpInterface
  )
