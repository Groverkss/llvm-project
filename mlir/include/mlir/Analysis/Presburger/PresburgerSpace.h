//===- PresburgerSpace.h - MLIR PresburgerSpace Class -----------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// Classes representing space information like number of identifiers and kind of
// identifiers.
//
//===----------------------------------------------------------------------===//

#ifndef MLIR_ANALYSIS_PRESBURGER_PRESBURGERSPACE_H
#define MLIR_ANALYSIS_PRESBURGER_PRESBURGERSPACE_H

#include "llvm/Support/ErrorHandling.h"

namespace mlir {

/// PresburgerSpace is a tuple of identifiers with information about what kind
/// they correspond to. The identifiers can be split into three types:
///
/// Dimension: Ordinary variables over which the space is represented.
///
/// Symbol: Symbol identifiers correspond to fixed but unknown values.
/// Mathematically, a space with symbolic identifiers is like a
/// family of spaces indexed by the symbolic identifiers.
///
/// Local: Local identifiers correspond to existentially quantified variables.
///
/// PresburgerSpace only supports identifiers of kind Dimension and Symbol.
class PresburgerSpace {
public:
  /// Kind of identifier (column).
  enum IdKind { Dimension, Symbol, Local };

  PresburgerSpace(unsigned numDims, unsigned numSymbols)
      : numDims(numDims), numSymbols(numSymbols) {}

  inline unsigned getNumIds() const { return numDims + numSymbols; }
  inline unsigned getNumDimIds() const { return numDims; }
  inline unsigned getNumSymbolIds() const { return numSymbols; }
  inline unsigned getNumDimAndSymbolIds() const { return numDims + numSymbols; }

  /// Get the number of ids of the specified kind.
  unsigned getNumIdKind(IdKind kind) const;

  /// Return the index at which the specified kind of id starts.
  unsigned getIdKindOffset(IdKind kind) const;

  /// Insert `num` identifiers of the specified kind.
  void insertIdSpace(IdKind kind, unsigned num = 1);

protected:
  /// Number of identifiers corresponding to real dimensions.
  unsigned numDims;

  /// Number of identifiers corresponding to symbols (unknown but constant for
  /// analysis).
  unsigned numSymbols;
};

/// Extension of PresburgerSpace supporting Local identifiers.
class PresburgerLocalSpace : public PresburgerSpace {
public:
  PresburgerLocalSpace(unsigned numDims, unsigned numSymbols,
                       unsigned numLocals)
      : PresburgerSpace(numDims, numSymbols),
        numIds(numDims + numSymbols + numLocals) {}

  explicit PresburgerLocalSpace(const PresburgerSpace &space)
      : PresburgerSpace(space), numIds(0) {}

  inline unsigned getNumIds() const { return numIds; }
  inline unsigned getNumLocalIds() const {
    return numIds - numDims - numSymbols;
  }

  /// Get the number of ids of the specified kind.
  unsigned getNumIdKind(IdKind kind) const;

  /// Return the index at which the specified kind of id starts.
  unsigned getIdKindOffset(IdKind kind) const;

  /// Insert `num` identifiers of the specified kind.
  void insertIdSpace(IdKind kind, unsigned num = 1);

protected:
  /// Total number of identifiers.
  unsigned numIds;
};

} // namespace mlir

#endif // MLIR_ANALYSIS_PRESBURGER_PRESBURGERSPACE_H
