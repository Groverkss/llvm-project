//===- PresburgerBasicSet.h - MLIR PresburgerBasicSet Class -----*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// A class to represent an integer polyhedra.
//
//===----------------------------------------------------------------------===//

#ifndef MLIR_ANALYSIS_PRESBURGER_PRESBURGERBASICSET_H
#define MLIR_ANALYSIS_PRESBURGER_PRESBURGERBASICSET_H

#include "mlir/Analysis/Presburger/Matrix.h"

namespace mlir {

class PresburgerBasicSet {
public:
  /// Kind of identifier (column).
  enum IdKind { Dimension, Symbol, Local };

  /// Constructs a constraint system reserving memory for the specified number
  /// of constraints and identifiers.
  PresburgerBasicSet(unsigned numReservedInequalities,
                     unsigned numReservedEqualities, unsigned numReservedCols,
                     unsigned numDims, unsigned numSymbols, unsigned numLocals)
      : numIds(numDims + numSymbols + numLocals), numDims(numDims),
        numSymbols(numSymbols),
        equalities(0, numIds + 1, numReservedEqualities, numReservedCols),
        inequalities(0, numIds + 1, numReservedInequalities, numReservedCols) {
    assert(numReservedCols >= numIds + 1);
  }

  /// Constructs a constraint system with the specified number of
  /// dimensions and symbols.
  PresburgerBasicSet(unsigned numDims = 0, unsigned numSymbols = 0,
                     unsigned numLocals = 0)
      : PresburgerBasicSet(/*numReservedInequalities=*/0,
                           /*numReservedEqualities=*/0,
                           /*numReservedCols=*/numDims + numSymbols +
                               numLocals + 1,
                           numDims, numSymbols, numLocals) {}

  virtual ~PresburgerBasicSet() = default;

  // Clones this object.
  std::unique_ptr<PresburgerBasicSet> clone() const;

  /// Clears any existing data and reserves memory for the specified
  /// constraints.
  virtual void reset(unsigned numReservedInequalities,
                     unsigned numReservedEqualities, unsigned numReservedCols,
                     unsigned numDims, unsigned numSymbols,
                     unsigned numLocals = 0);

  void reset(unsigned numDims = 0, unsigned numSymbols = 0,
             unsigned numLocals = 0);

  /// Appends constraints from `other` into `this`. This is equivalent to an
  /// intersection with no simplification of any sort attempted.
  void append(const PresburgerBasicSet &other);

  /// Returns the value at the specified equality row and column.
  inline int64_t atEq(unsigned i, unsigned j) const { return equalities(i, j); }
  inline int64_t &atEq(unsigned i, unsigned j) { return equalities(i, j); }

  /// Returns the value at the specified inequality row and column.
  inline int64_t atIneq(unsigned i, unsigned j) const {
    return inequalities(i, j);
  }
  inline int64_t &atIneq(unsigned i, unsigned j) { return inequalities(i, j); }

  unsigned getNumConstraints() const {
    return getNumInequalities() + getNumEqualities();
  }
  inline unsigned getNumIds() const { return numIds; }
  inline unsigned getNumDimIds() const { return numDims; }
  inline unsigned getNumSymbolIds() const { return numSymbols; }
  inline unsigned getNumDimAndSymbolIds() const { return numDims + numSymbols; }
  inline unsigned getNumLocalIds() const {
    return numIds - numDims - numSymbols;
  }

  /// Returns the number of columns in the constraint system.
  inline unsigned getNumCols() const { return numIds + 1; }

  inline unsigned getNumEqualities() const { return equalities.getNumRows(); }

  inline unsigned getNumInequalities() const {
    return inequalities.getNumRows();
  }

  inline unsigned getNumReservedEqualities() const {
    return equalities.getNumReservedRows();
  }

  inline unsigned getNumReservedInequalities() const {
    return inequalities.getNumReservedRows();
  }

  inline ArrayRef<int64_t> getEquality(unsigned idx) const {
    return equalities.getRow(idx);
  }

  inline ArrayRef<int64_t> getInequality(unsigned idx) const {
    return inequalities.getRow(idx);
  }

  /// Insert `num` identifiers of the specified kind at position `pos`.
  /// Positions are relative to the kind of identifier. The coefficient columns
  /// corresponding to the added identifiers are initialized to zero. Return the
  /// absolute column position (i.e., not relative to the kind of identifier)
  /// of the first added identifier.
  unsigned insertDimId(unsigned pos, unsigned num = 1);
  unsigned insertSymbolId(unsigned pos, unsigned num = 1);
  unsigned insertLocalId(unsigned pos, unsigned num = 1);
  virtual unsigned insertId(IdKind kind, unsigned pos, unsigned num = 1);

  /// Append `num` identifiers of the specified kind after the last identifier.
  /// of that kind. Return the position of the first appended column. The
  /// coefficient columns corresponding to the added identifiers are initialized
  /// to zero.
  unsigned appendDimId(unsigned num = 1);
  unsigned appendSymbolId(unsigned num = 1);
  unsigned appendLocalId(unsigned num = 1);

  /// Adds an inequality (>= 0) from the coefficients specified in `inEq`.
  void addInequality(ArrayRef<int64_t> inEq);
  /// Adds an equality from the coefficients specified in `eq`.
  void addEquality(ArrayRef<int64_t> eq);

  /// Removes identifiers of the specified kind with the specified pos (or
  /// within the specified range) from the system. The specified location is
  /// relative to the first identifier of the specified kind.
  void removeId(IdKind kind, unsigned pos);
  void removeIdRange(IdKind kind, unsigned idStart, unsigned idLimit);

  /// Removes the specified identifier from the system.
  void removeId(unsigned pos);

  void removeEquality(unsigned pos);
  void removeInequality(unsigned pos);

  /// Remove the (in)equalities at positions [start, end).
  void removeEqualityRange(unsigned start, unsigned end);
  void removeInequalityRange(unsigned start, unsigned end);

  /// Swap the posA^th identifier with the posB^th identifier.
  virtual void swapId(unsigned posA, unsigned posB);

  /// Removes all equalities and inequalities.
  void clearConstraints();

protected:
  /// Return the index at which the specified kind of id starts.
  unsigned getIdKindOffset(IdKind kind) const;

  /// Assert that `value` is at most the number of ids of the specified kind.
  void assertAtMostNumIdKind(unsigned value, IdKind kind) const;

  /// Removes identifiers in the column range [idStart, idLimit), and copies any
  /// remaining valid data into place, updates member variables, and resizes
  /// arrays as needed.
  virtual void removeIdRange(unsigned idStart, unsigned idLimit);

  /// Total number of identifiers.
  unsigned numIds;

  /// Number of identifiers corresponding to real dimensions.
  unsigned numDims;

  /// Number of identifiers corresponding to symbols (unknown but constant for
  /// analysis).
  unsigned numSymbols;

  /// Coefficients of affine equalities (in == 0 form).
  Matrix equalities;

  /// Coefficients of affine inequalities (in >= 0 form).
  Matrix inequalities;
};

} // namespace mlir

#endif // MLIR_ANALYSIS_PRESBURGER_PRESBURGERBASICSET_H
