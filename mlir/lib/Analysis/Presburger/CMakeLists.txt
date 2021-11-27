add_mlir_library(MLIRPresburger
  Simplex.cpp
  Matrix.cpp
  PresburgerBasicSet.cpp

  DEPENDS
  MLIRBuiltinLocationAttributesIncGen

  LINK_LIBS PUBLIC
  MLIRIR
  )
