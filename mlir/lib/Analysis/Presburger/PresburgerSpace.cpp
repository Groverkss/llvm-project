//===- PresburgerSpace.cpp - MLIR PresburgerSpace Class -------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "mlir/Analysis/Presburger/PresburgerSpace.h"

using namespace mlir;

unsigned PresburgerSpace::getNumIdKind(IdKind kind) const {
  if (kind == IdKind::Dimension)
    return getNumDimIds();
  if (kind == IdKind::Symbol)
    return getNumSymbolIds();
  llvm_unreachable(
      "PresburgerSpace only supports Dimensions and Symbol identifiers!");
}

unsigned PresburgerSpace::getIdKindOffset(IdKind kind) const {
  if (kind == IdKind::Dimension)
    return 0;
  if (kind == IdKind::Symbol)
    return getNumDimIds();
  llvm_unreachable(
      "PresburgerSpace only supports Dimensions and Symbol identifiers!");
}

void PresburgerSpace::insertIdSpace(IdKind kind, unsigned num) {
  if (kind == IdKind::Dimension)
    numDims += num;
  else if (kind == IdKind::Symbol)
    numSymbols += num;
  else
    llvm_unreachable(
        "PresburgerSpace only supports Dimensions and Symbol identifiers!");
}

unsigned PresburgerLocalSpace::getNumIdKind(IdKind kind) const {
  if (kind == IdKind::Local)
    return getNumLocalIds();
  return PresburgerSpace::getNumIdKind(kind);
}

unsigned PresburgerLocalSpace::getIdKindOffset(IdKind kind) const {
  if (kind == IdKind::Local)
    return getNumDimAndSymbolIds();
  return PresburgerSpace::getIdKindOffset(kind);
}

void PresburgerLocalSpace::insertIdSpace(IdKind kind, unsigned num) {
  if (kind != IdKind::Local)
    PresburgerSpace::insertIdSpace(kind, num);
  numIds += num;
}
