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

#include "mlir/Support/TypeID.h"
#include "llvm/ADT/ArrayRef.h"
#include "llvm/Support/ErrorHandling.h"
#include "llvm/Support/raw_ostream.h"

namespace mlir {
namespace presburger {

/// Kind of identifier. Implementation wise SetDims are treated as Range
/// ids, and spaces with no distinction between dimension ids are treated
/// as relations with zero domain ids.
enum class IdKind { Symbol, Local, Domain, Range, SetDim = Range };

/// PresburgerSpace is the space of all possible values of a tuple of integer
/// valued variables/identifiers. Each identifier has one of the three types:
///
/// Dimension: Ordinary variables over which the space is represented.
///
/// Symbol: Symbol identifiers correspond to fixed but unknown values.
/// Mathematically, a space with symbolic identifiers is like a
/// family of spaces indexed by the symbolic identifiers.
///
/// Local: Local identifiers correspond to existentially quantified variables.
/// For example, consider the space: `(x, exists q)` where x is a dimension
/// identifier and q is a local identifier. Let us put the constraints:
///       `1 <= x <= 7, x = 2q`
/// on this space to get the set:
///       `(x) : (exists q : q <= x <= 7, x = 2q)`.
/// An assignment to symbolic and dimension variables is valid if there
/// exists some assignment to the local variable `q` satisfying these
/// constraints. For this example, the set is equivalent to {2, 4, 6}.
/// Mathematically, existential quantification can be thought of as the result
/// of projection. In this example, `q` is existentially quantified. This can be
/// thought of as the result of projecting out `q` from the previous example,
/// i.e. we obtained {2, 4, 6} by projecting out the second dimension from
/// {(2, 1), (4, 2), (6, 2)}.
///
/// Dimension identifiers are further divided into Domain and Range identifiers
/// to support building relations.
///
/// Identifiers are stored in the following order:
///       [Domain, Range, Symbols, Locals]
///
/// A space with no distinction between types of dimension identifiers can
/// be implemented as a space with zero domain. IdKind::SetDim should be used
/// to refer to dimensions in such spaces.
///
/// Compatibility of two spaces implies that number of identifiers of each kind
/// other than Locals are equal. Equality of two spaces implies that number of
/// identifiers of each kind are equal.
///
/// PresburgerSpace also allows attaching a `void` pointer to each identifier
/// to allow attaching information with a value. Local identifiers cannot have a
/// value attached to them.
///
/// To allow attaching these values, a space needs to be constructed with
/// `usingValues` as `true`. If values are being used, two values are considered
/// equal if they hold the same address. A `NULL` value is considered
/// uninitialized and is considered unique, i.e. 2 variables with a `NULL` value
/// are not equal.
class PresburgerSpace {
public:
  static PresburgerSpace getRelationSpace(unsigned numDomain = 0,
                                          unsigned numRange = 0,
                                          unsigned numSymbols = 0,
                                          unsigned numLocals = 0,
                                          bool usingValues = false) {
    return PresburgerSpace(numDomain, numRange, numSymbols, numLocals,
                           usingValues);
  }

  static PresburgerSpace getSetSpace(unsigned numDims = 0,
                                     unsigned numSymbols = 0,
                                     unsigned numLocals = 0,
                                     bool usingValues = false) {
    return PresburgerSpace(/*numDomain=*/0, /*numRange=*/numDims, numSymbols,
                           numLocals, usingValues);
  }

  unsigned getNumDomainIds() const { return numDomain; }
  unsigned getNumRangeIds() const { return numRange; }
  unsigned getNumSetDimIds() const { return numRange; }
  unsigned getNumSymbolIds() const { return numSymbols; }
  unsigned getNumLocalIds() const { return numLocals; }

  unsigned getNumDimIds() const { return numDomain + numRange; }
  unsigned getNumDimAndSymbolIds() const {
    return numDomain + numRange + numSymbols;
  }
  unsigned getNumIds() const {
    return numDomain + numRange + numSymbols + numLocals;
  }

  /// Get the number of ids of the specified kind.
  unsigned getNumIdKind(IdKind kind) const;

  /// Return the index at which the specified kind of id starts.
  unsigned getIdKindOffset(IdKind kind) const;

  /// Return the index at Which the specified kind of id ends.
  unsigned getIdKindEnd(IdKind kind) const;

  /// Get the number of elements of the specified kind in the range
  /// [idStart, idLimit).
  unsigned getIdKindOverlap(IdKind kind, unsigned idStart,
                            unsigned idLimit) const;

  /// Return the IdKind of the id at the specified position.
  IdKind getIdKindAt(unsigned pos) const;

  /// Insert `num` identifiers of the specified kind at position `pos`.
  /// Positions are relative to the kind of identifier. Return the absolute
  /// column position (i.e., not relative to the kind of identifier) of the
  /// first added identifier.
  unsigned insertId(IdKind kind, unsigned pos, unsigned num = 1);

  /// Removes identifiers of the specified kind in the column range [idStart,
  /// idLimit). The range is relative to the kind of identifier.
  void removeIdRange(IdKind kind, unsigned idStart, unsigned idLimit);

  /// TODO: Write doc.
  void swapId(IdKind kindA, IdKind kindB, unsigned posA, unsigned posB);

  /// Returns true if both the spaces are compatible i.e. if both spaces have
  /// the same number of identifiers of each kind (excluding locals).
  bool isCompatible(const PresburgerSpace &other) const;

  /// Returns true if both the spaces are equal including local identifiers i.e.
  /// if both spaces have the same number of identifiers of each kind (including
  /// locals).
  bool isEqual(const PresburgerSpace &other) const;

  /// Changes the partition between dimensions and symbols. Depending on the new
  /// symbol count, either a chunk of dimensional identifiers immediately before
  /// the split become symbols, or some of the symbols immediately after the
  /// split become dimensions.
  void setDimSymbolSeparation(unsigned newSymbolCount);

  void print(llvm::raw_ostream &os) const;
  void dump() const;

  //===--------------------------------------------------------------------===//
  //     Value Interactions
  //===--------------------------------------------------------------------===//

  /// Access value attached to `i^th` variable.
  void *&atValue(IdKind kind, unsigned i) {
    assert(usingValues && "Cannot access values when `usingValues` is false.");
    assert(kind != IdKind::Local &&
           "Values cannot be attached to local identifiers.");
    return values[getIdKindOffset(kind) + i];
  }
  void *atValue(IdKind kind, unsigned i) const {
    assert(usingValues && "Cannot access values when `usingValues` is false.");
    assert(kind != IdKind::Local &&
           "Values cannot be attached to local identifiers.");
    return values[getIdKindOffset(kind) + i];
  }

  TypeID &atType(IdKind kind, unsigned i) {
    assert(usingValues && "Cannot access values when `usingValues` is false.");
    assert(kind != IdKind::Local &&
           "Values cannot be attached to local identifiers.");
    return types[getIdKindOffset(kind) + i];
  }

  /// Set the value attached to the `i^th` variable to `value`.
  template <typename T>
  void setValue(IdKind kind, unsigned i, T value) {
    atValue(kind, i) = static_cast<void *>(value);

#ifndef NDEBUG
    types[getIdKindOffset(kind) + i] = TypeID::get<T>();
#endif
  }

  /// Get the value attached to the `i^th` variable casted to type `T`.
  template <typename T>
  T getValue(IdKind kind, unsigned i) const {
    assert(types[getIdKindOffset(kind) + i] == TypeID::get<T>() &&
           "Type mismatch");
    return static_cast<T>(atValue(kind, i));
  }

  bool isAligned(const PresburgerSpace &other) const {
    return isCompatible(other) && values == other.values;
  }

  unsigned findId(IdKind kind, void *val) {
    unsigned i = 0;
    for (unsigned e = getNumIdKind(kind); i < e; ++i)
      if (atValue(kind, i) != nullptr && atValue(kind, i) == val)
        break;
    return i;
  }

  bool isUsingValues() const { return usingValues; }

  /// Reset the stored values in the space. Enables `usingValues` if it was
  /// `false` before.
  void resetValues();

protected:
  PresburgerSpace(unsigned numDomain = 0, unsigned numRange = 0,
                  unsigned numSymbols = 0, unsigned numLocals = 0,
                  bool usingValues = false)
      : numDomain(numDomain), numRange(numRange), numSymbols(numSymbols),
        numLocals(numLocals), usingValues(usingValues) {
    if (usingValues) {
      values.resize(getNumDimAndSymbolIds(), nullptr);
#ifndef NDEBUG
      types.resize(getNumDimAndSymbolIds(), TypeID::get<void>());
#endif
    }
  }

private:
  // Number of identifiers corresponding to domain identifiers.
  unsigned numDomain;

  // Number of identifiers corresponding to range identifiers.
  unsigned numRange;

  /// Number of identifiers corresponding to symbols (unknown but constant for
  /// analysis).
  unsigned numSymbols;

  /// Number of identifers corresponding to locals (identifiers corresponding
  /// to existentially quantified variables).
  unsigned numLocals;

  /// Stores whether or not values are attached to this space.
  bool usingValues;

#ifndef NDEBUG
  // Stores debug type information for values.
  llvm::SmallVector<TypeID> types;
#endif

  /// Stores a value for each non-local identifier as a `void` pointer.
  llvm::SmallVector<void *> values;
};

} // namespace presburger
} // namespace mlir

#endif // MLIR_ANALYSIS_PRESBURGER_PRESBURGERSPACE_H
