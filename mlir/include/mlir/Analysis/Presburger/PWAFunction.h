//===- PWAFunction.h - MLIR PWAFunction Class -------------------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// Support for PieceWiseAffine Functions.
//
//===----------------------------------------------------------------------===//

#ifndef MLIR_ANALYSIS_PRESBURGER_PWAFUNCTION_H
#define MLIR_ANALYSIS_PRESBURGER_PWAFUNCTION_H

#include "mlir/Analysis/Presburger/IntegerPolyhedron.h"

namespace mlir {

class PWAFunction {
public:
  PWAFunction(unsigned numInput, unsigned numOutput)
      : numInput(numInput), numOutput(numOutput) {}

  void addPiece(IntegerPolyhedron &domain, Matrix &field);

  IntegerPolyhedron &getDomain(unsigned i) { return pieceDomain[i]; };
  IntegerPolyhedron getDomain(unsigned i) const { return pieceDomain[i]; };

  Matrix &getField(unsigned i) { return pieceField[i]; };
  Matrix getField(unsigned i) const { return pieceField[i]; };

  unsigned getNumPieces() const { return pieceDomain.size(); }

  void print(raw_ostream &os) const;
  void dump() const;

private:
  SmallVector<IntegerPolyhedron, 4> pieceDomain;
  SmallVector<Matrix, 4> pieceField;

  unsigned numInput;
  unsigned numOutput;
};

} // namespace mlir

#endif // MLIR_ANALYSIS_PRESBURGER_PWAFUNCTION_H
