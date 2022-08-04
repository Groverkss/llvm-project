//===- PWMAFunction.h - MLIR PWMAFunction Class------------------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// Support for piece-wise multi-affine functions. These are functions that are
// defined on a domain that is a union of IntegerPolyhedrons, and on each domain
// the value of the function is a tuple of integers, with each value in the
// tuple being an affine expression in the vars of the IntegerPolyhedron.
//
//===----------------------------------------------------------------------===//

#ifndef MLIR_ANALYSIS_PRESBURGER_PWMAFUNCTION_H
#define MLIR_ANALYSIS_PRESBURGER_PWMAFUNCTION_H

#include "mlir/Analysis/Presburger/IntegerRelation.h"
#include "mlir/Analysis/Presburger/PresburgerRelation.h"

namespace mlir {
namespace presburger {

enum class Compare { LT, GT, GE, LE, EQ };

/// This class represents a multi-affine function whose domain is given by an
/// IntegerPolyhedron. This can be thought of as an IntegerPolyhedron with a
/// tuple of integer values attached to every point in the polyhedron, with the
/// value of each element of the tuple given by an affine expression in the vars
/// of the polyhedron. For example we could have the domain
///
/// (x, y) : (x >= 5, y >= x)
///
/// and a tuple of three integers defined at every point in the polyhedron:
///
/// (x, y) -> (x + 2, 2*x - 3y + 5, 2*x + y).
///
/// In this way every point in the polyhedron has a tuple of integers associated
/// with it. If the integer polyhedron has local vars, then the output
/// expressions can use them as well. The output expressions are represented as
/// a matrix with one row for every element in the output vector one column for
/// each var, and an extra column at the end for the constant term.
///
/// Checking equality of two such functions is supported, as well as finding the
/// value of the function at a specified point.
class MultiAffineFunction {
public:
  MultiAffineFunction(const PresburgerSpace &space, const Matrix &output)
      : space(space), output(output),
        divs(space.getNumVars() - space.getNumRangeVars()) {
    assertIsConsistent();
  }

  MultiAffineFunction(const PresburgerSpace &space, const Matrix &output,
                      const DivisionRepr &divs)
      : space(space), output(output), divs(divs) {
    assertIsConsistent();
  }

  unsigned getNumDomainVars() const { return space.getNumDomainVars(); }
  unsigned getNumSymbolVars() const { return space.getNumSymbolVars(); }
  unsigned getNumOutputs() const { return space.getNumRangeVars(); }
  unsigned getNumDivs() const { return space.getNumLocalVars(); }

  PresburgerSpace getDomainSpace() const { return space.getDomainSpace(); }

  void assertIsConsistent() const;

  /// Get the space of the input domain of this function.
  const PresburgerSpace &getSpace() const { return space; }
  /// Get a matrix with each row representing row^th output expression.
  const Matrix &getOutputMatrix() const { return output; }
  /// Get the `i^th` output expression.
  ArrayRef<int64_t> getOutputExpr(unsigned i) const { return output.getRow(i); }

  const DivisionRepr &getDivs() const { return divs; }

  void removeOutputs(unsigned start, unsigned end);

  /// Given a MAF `other`, merges local variables such that both funcitons
  /// have union of local vars, without changing the set of points in domain or
  /// the output.
  void mergeLocalVars(MultiAffineFunction &other);

  SmallVector<int64_t, 8> valueAt(ArrayRef<int64_t> point) const;

  /// Return whether the `this` and `other` are equal. This is the case if
  /// they lie in the same space, i.e. have the same dimensions, their outputs
  /// are equal.
  /// TODO: Add more docs for restrict.
  bool isEqual(const MultiAffineFunction &other) const;
  bool isEqual(const MultiAffineFunction &other,
               const IntegerPolyhedron &restrict) const;
  bool isEqual(const MultiAffineFunction &other,
               const PresburgerSet &restrict) const;

  void subtract(const MultiAffineFunction &other);

  PresburgerSet getLexSet(Compare comp, const MultiAffineFunction &other) const;

  IntegerRelation getAsRelation() const;

  void print(raw_ostream &os) const;
  void dump() const;

private:
  PresburgerSpace space;

  /// The function's output is a tuple of integers, with the ith element of the
  /// tuple defined by the affine expression given by the ith row of this output
  /// matrix.
  Matrix output;

  DivisionRepr divs;
};

/// This class represents a piece-wise MultiAffineFunction. This can be thought
/// of as a list of MultiAffineFunction with disjoint domains, with each having
/// their own affine expressions for their output tuples. For example, we could
/// have a function with two input variables (x, y), defined as
///
/// f(x, y) = (2*x + y, y - 4)  if x >= 0, y >= 0
///         = (-2*x + y, y + 4) if x < 0,  y < 0
///         = (4, 1)            if x < 0,  y >= 0
///
/// Note that the domains all have to be *disjoint*. Otherwise, the behaviour of
/// this class is undefined. The domains need not cover all possible points;
/// this represents a partial function and so could be undefined at some points.
///
/// As in PresburgerSets, the input vars are partitioned into dimension vars and
/// symbolic vars.
///
/// Support is provided to compare equality of two such functions as well as
/// finding the value of the function at a point.
class PWMAFunction {
public:
  struct Piece {
    PresburgerSet domain;
    MultiAffineFunction output;
  };

  PWMAFunction(const PresburgerSpace &space) : space(space) {
    assert(space.getNumLocalVars() == 0 &&
           "PWMAFunction cannot have local vars.");
  }

  const PresburgerSpace &getSpace() const { return space; }

  void addPiece(const Piece &piece);

  unsigned getNumPieces() const { return pieces.size(); }
  unsigned getNumVarKind(VarKind kind) const {
    return space.getNumVarKind(kind);
  }
  unsigned getNumDomainVars() const { return space.getNumDomainVars(); }
  unsigned getNumOutputs() const { return space.getNumRangeVars(); }
  unsigned getNumSymbolVars() const { return space.getNumSymbolVars(); }

  void removeOutputs(unsigned start, unsigned end);

  PresburgerSpace getDomainSpace() const { return space.getDomainSpace(); }

  /// Return the domain of this piece-wise MultiAffineFunction. This is the
  /// union of the domains of all the pieces.
  PresburgerSet getDomain() const;

  ArrayRef<Piece> getAllPieces() const { return pieces; }

  Optional<SmallVector<int64_t, 8>> valueAt(ArrayRef<int64_t> point) const;

  /// Return whether `this` and `other` are equal as PWMAFunctions, i.e. whether
  /// they have the same dimensions, the same domain and they take the same
  /// value at every point in the domain.
  bool isEqual(const PWMAFunction &other) const;

  /// Return a function defined on the union of the domains of this and func,
  /// such that when only one of the functions is defined, it outputs the same
  /// as that function, and if both are defined, it outputs the lexmax/lexmin of
  /// the two outputs. On points where neither function is defined, the returned
  /// function is not defined either.
  ///
  /// Currently this does not support PWMAFunctions which have pieces containing
  /// local variables.
  /// TODO: Support local variables in peices.
  PWMAFunction unionLexMin(const PWMAFunction &func);
  PWMAFunction unionLexMax(const PWMAFunction &func);

  void print(raw_ostream &os) const;
  void dump() const;

private:
  /// Return a function defined on the union of the domains of `this` and
  /// `func`, such that when only one of the functions is defined, it outputs
  /// the same as that function, and if neither is defined, the returned
  /// function is not defined either.
  ///
  /// The provided `tiebreak` function determines which of the two functions'
  /// output should be used on inputs where both the functions are defined. More
  /// precisely, given two `MultiAffineFunction`s `mafA` and `mafB`, `tiebreak`
  /// returns the subset of the intersection of the two functions' domains where
  /// the output of `mafA` should be used.
  ///
  /// The PresburgerSet returned by `tiebreak` should be disjoint.
  /// TODO: Remove this constraint of returning disjoint set.
  PWMAFunction unionFunction(
      const PWMAFunction &func,
      llvm::function_ref<PresburgerSet(Piece mafA, Piece mafB)> tiebreak) const;

  PresburgerSpace space;

  SmallVector<Piece, 4> pieces;
};

} // namespace presburger
} // namespace mlir

#endif // MLIR_ANALYSIS_PRESBURGER_PWMAFUNCTION_H
