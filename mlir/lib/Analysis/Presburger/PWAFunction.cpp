//===- PWAFunction.cpp - MLIR PWAFunction Class ---------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "mlir/Analysis/Presburger/PWAFunction.h"

using namespace mlir;

void PWAFunction::addPiece(IntegerPolyhedron &domain, Matrix &field) {
  assert(domain.getNumIds() == numInput);
  assert(field.getNumColumns() == numInput + 1);
  assert(field.getNumRows() == numOutput);

  pieceDomain.push_back(IntegerPolyhedron(domain));
  pieceField.push_back(field);
}

void PWAFunction::print(raw_ostream &os) const {
  for (unsigned i = 0, e = getNumPieces(); i < e; ++i) {
    os << "Domain:";
    pieceDomain[i].print(os);
    os << "Field:\n";
    pieceField[i].print(os);
    os << "\n";
  }
}

void PWAFunction::dump() const { print(llvm::errs()); }
