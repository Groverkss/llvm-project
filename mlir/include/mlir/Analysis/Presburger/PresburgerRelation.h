//===- PresburgerRelation.h - MLIR PresburgerRelation Class -----*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// A class to represent unions of IntegerRelation.
//
//===----------------------------------------------------------------------===//

#ifndef MLIR_ANALYSIS_PRESBURGER_PRESBURGERRELATION_H
#define MLIR_ANALYSIS_PRESBURGER_PRESBURGERRELATION_H

#include "mlir/Analysis/Presburger/IntegerRelation.h"

namespace mlir {
namespace presburger {

/// A PresburgerUnion represents a union of IntegerRelations that live in
/// the same PresburgerSpace with support for union, intersection, subtraction,
/// and complement operations, as well as sampling.
///
/// The IntegerRelations (polyhedrons) are stored in a vector, and the set
/// represents the union of these polyhedrons. An empty list corresponds to the
/// empty set.
///
/// Note that there are no invariants guaranteed on the list of polyhedrons
/// other than that they are all in the same PresburgerSpace. For example, the
/// polyhedrons may overlap each other.
template <typename PolyhedronImpl>
class PresburgerUnion : public PresburgerSpace {

  /// Assert that only Relation and Set are supported.
  static_assert(
      std::is_same<PolyhedronImpl, IntegerRelation>::value ||
          std::is_same<PolyhedronImpl, IntegerPolyhedron>::value,
      "PolyhedronImpl should be of type IntegerRelation or IntegerPolyhedron");

public:
  /// Return a universe set of the specified type that contains all points.
  static PresburgerUnion getUniverse(const PresburgerSpace &space);

  /// Return an empty set of the specified type that contains no points.
  static PresburgerUnion getEmpty(const PresburgerSpace &space);

  /// Return a universe set of the specified type that contains all points.
  static PresburgerUnion getUniverse(unsigned numDims = 0,
                                     unsigned numSymbols = 0) {
    return getUniverse(PresburgerSpace(/*numDomain=*/0, numDims, numSymbols));
  }

  /// Return an empty set of the specified type that contains no points.
  static PresburgerUnion getEmpty(unsigned numDims = 0,
                                  unsigned numSymbols = 0) {
    return getEmpty(PresburgerSpace(/*numDomain=*/0, numDims, numSymbols));
  }

  explicit PresburgerUnion(const PolyhedronImpl &poly);

  /// Return the number of Polys in the union.
  unsigned getNumPolys() const;

  /// Return a reference to the list of PolyhedronImpls.
  ArrayRef<PolyhedronImpl> getAllPolys() const;

  /// Return the PolyhedronImpl at the specified index.
  const PolyhedronImpl &getPoly(unsigned index) const;

  /// Mutate this set, turning it into the union of this set and the given
  /// PolyhedronImpl.
  void unionInPlace(const PolyhedronImpl &poly);

  /// Mutate this set, turning it into the union of this set and the given set.
  void unionInPlace(const PresburgerUnion &set);

  /// Return the union of this set and the given set.
  PresburgerUnion unionSet(const PresburgerUnion &set) const;

  /// Return the intersection of this set and the given set.
  PresburgerUnion intersect(const PresburgerUnion &set) const;

  /// Return true if the set contains the given point, and false otherwise.
  bool containsPoint(ArrayRef<int64_t> point) const;

  /// Return the complement of this set. All local variables in the set must
  /// correspond to floor divisions.
  PresburgerUnion complement() const;

  /// Return the set difference of this set and the given set, i.e.,
  /// return `this \ set`. All local variables in `set` must correspond
  /// to floor divisions, but local variables in `this` need not correspond to
  /// divisions.
  PresburgerUnion subtract(const PresburgerUnion &set) const;

  /// Return true if this set is a subset of the given set, and false otherwise.
  bool isSubsetOf(const PresburgerUnion &set) const;

  /// Return true if this set is equal to the given set, and false otherwise.
  /// All local variables in both sets must correspond to floor divisions.
  bool isEqual(const PresburgerUnion &set) const;

  /// Return true if all the sets in the union are known to be integer empty
  /// false otherwise.
  bool isIntegerEmpty() const;

  /// Find an integer sample from the given set. This should not be called if
  /// any of the polyhedrons in the union are unbounded.
  bool findIntegerSample(SmallVectorImpl<int64_t> &sample);

  /// Compute an overapproximation of the number of integer points in the
  /// polyhedron. Symbol ids are currently not supported. If the computed
  /// overapproximation is infinite, an empty optional is returned.
  ///
  /// This currently just sums up the overapproximations of the volumes of the
  /// disjuncts, so the approximation might be far from the true volume in the
  /// case when there is a lot of overlap between disjuncts.
  Optional<uint64_t> computeVolume() const;

  /// Simplifies the representation of a PresburgerUnion.
  ///
  /// In particular, removes all polyhedrons which are subsets of other
  /// polyhedrons in the union.
  PresburgerUnion coalesce() const;

  /// Print the set's internal state.
  void print(raw_ostream &os) const;
  void dump() const;

protected:
  /// Construct an empty PresburgerUnion with the specified number of
  /// dimension and symbols.
  PresburgerUnion(unsigned numDomain = 0, unsigned numRange = 0,
                  unsigned numSymbols = 0)
      : PresburgerSpace(numDomain, numRange, numSymbols) {}

  /// The list of polyhedrons that this set is the union of.
  SmallVector<PolyhedronImpl, 2> polyhedronImpls;
};

using PresburgerRelation = PresburgerUnion<IntegerRelation>;
using PresburgerSet = PresburgerUnion<IntegerPolyhedron>;

} // namespace presburger
} // namespace mlir

#endif // MLIR_ANALYSIS_PRESBURGER_PRESBURGERRELATION_H
