//===- PresburgerBasicSet.h - MLIR PresburgerBasicSet Class -----*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// Functionality to perform analysis on FlatAffineConstraints. In particular,
// support for performing emptiness checks.
//
//===----------------------------------------------------------------------===//

#ifndef MLIR_ANALYSIS_PRESBURGER_PRESBURGERBASICSET_H
#define MLIR_ANALYSIS_PRESBURGER_PRESBURGERBASICSET_H

#include "mlir/Analysis/Presburger/Constraint.h"
#include "mlir/Analysis/Presburger/Matrix.h"
#include "mlir/Analysis/Presburger/Simplex.h"

#include "llvm/ADT/ArrayRef.h"
#include "llvm/ADT/SmallVector.h"

namespace mlir {
namespace analysis {
namespace presburger {

class PresburgerSet;

class PresburgerBasicSet {
public:
  friend class PresburgerSet;

  PresburgerBasicSet(unsigned oNDim = 0, unsigned oNParam = 0,
                     unsigned oNExist = 0)
      : nDim(oNDim), nParam(oNParam), nExist(oNExist) {}

  unsigned getNumDims() const { return nDim; }
  unsigned getNumTotalDims() const {
    return nParam + nDim + nExist + divs.size();
  }
  unsigned getNumParams() const { return nParam; }
  unsigned getNumExists() const { return nExist; }
  unsigned getNumDivs() const { return divs.size(); }
  unsigned getNumInequalities() const { return ineqs.size(); }
  unsigned getNumEqualities() const { return eqs.size(); }

  void intersect(PresburgerBasicSet bs);

  void appendDivisionVariable(ArrayRef<SafeInteger> coeffs, SafeInteger denom);

  static void toCommonSpace(PresburgerBasicSet &a, PresburgerBasicSet &b);
  void appendDivisionVariables(ArrayRef<DivisionConstraint> newDivs);
  void prependDivisionVariables(ArrayRef<DivisionConstraint> newDivs);

  const InequalityConstraint &getInequality(unsigned i) const;
  const EqualityConstraint &getEquality(unsigned i) const;
  ArrayRef<InequalityConstraint> getInequalities() const;
  ArrayRef<EqualityConstraint> getEqualities() const;
  ArrayRef<DivisionConstraint> getDivisions() const;

  void addInequality(ArrayRef<SafeInteger> coeffs);
  void addEquality(ArrayRef<SafeInteger> coeffs);

  void removeLastInequality();
  void removeLastEquality();
  void removeLastDivision();

  void removeInequality(unsigned i);
  void removeEquality(unsigned i);

  /// Find a sample point satisfying the constraints. This uses a branch and
  /// bound algorithm with generalized basis reduction, which always works if
  /// the set is bounded. This should not be called for unbounded sets.
  ///
  /// Returns such a point if one exists, or an empty Optional otherwise.
  Optional<SmallVector<SafeInteger, 8>> findIntegerSample();

  bool isIntegerEmptyOnlyEqualities();
  bool isIntegerEmpty();

  /// Get a {denominator, sample} pair representing a rational sample point in
  /// this basic set.
  Optional<std::pair<SafeInteger, SmallVector<SafeInteger, 8>>>
  findRationalSample() const;

  PresburgerBasicSet makeRecessionCone() const;

  void dumpCoeffs() const;

  void dump() const;
  void print(raw_ostream &os) const;

  void printISL(raw_ostream &os) const;
  void dumpISL() const;

private:
  void substitute(ArrayRef<SafeInteger> values);

  /// Find a sample point in this basic set, when it is known that this basic
  /// set has no unbounded directions.
  ///
  /// \returns the sample point or an empty llvm::Optional if the set is empty.
  Optional<SmallVector<SafeInteger, 8>> findSampleBounded(bool onlyEmptiness);

  /// Find a sample for only the bounded dimensions of this basic set.
  ///
  /// \param cone should be the recession cone of this basic set.
  ///
  /// \returns the sample or an empty std::optional if no sample exists.
  Optional<SmallVector<SafeInteger, 8>>
  findBoundedDimensionsSample(const PresburgerBasicSet &cone,
                              bool onlyEmptiness) const;

  /// Find a sample for this basic set, which is known to be a full-dimensional
  /// cone.
  ///
  /// \returns the sample point or an empty std::optional if the set is empty.
  Optional<SmallVector<SafeInteger, 8>> findSampleFullCone();

  /// Project this basic set to its bounded dimensions. It is assumed that the
  /// unbounded dimensions occupy the last \p unboundedDims dimensions.
  void projectOutUnboundedDimensions(unsigned unboundedDims);

  /// Find a sample point in this basic set, which has unbounded directions.
  ///
  /// \param cone should be the recession cone of this basic set.
  ///
  /// \returns the sample point or an empty llvm::Optional if the set
  /// is empty.
  Optional<SmallVector<SafeInteger, 8>>
  findSampleUnbounded(PresburgerBasicSet &cone, bool onlyEmptiness);

  Matrix coefficientMatrixFromEqs() const;

  void insertDimensions(unsigned pos, unsigned count);
  void prependExistentialDimensions(unsigned count);
  void appendExistentialDimensions(unsigned count);

  PresburgerBasicSet makePlainBasicSet() const;
  bool isPlainBasicSet() const;

  void updateFromSimplex(const Simplex &simplex);

  SmallVector<InequalityConstraint, 8> ineqs;
  SmallVector<EqualityConstraint, 8> eqs;
  SmallVector<DivisionConstraint, 8> divs;
  unsigned nDim, nParam, nExist;
};
} // namespace presburger
} // namespace analysis
} // namespace mlir

#endif // MLIR_ANALYSIS_PRESBURGER_PRESBURGERBASICSET_H
