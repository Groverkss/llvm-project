//===- SetTest.cpp - Tests for PresburgerRelation
//------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This file contains tests for PresburgerRelation. The tests for union,
// intersection, subtract, and complement work by computing the operation on
// two sets and checking, for a set of points, that the resulting set contains
// the point iff the result is supposed to contain it. The test for isEqual just
// checks if the result for two sets matches the expected result.
//
//===----------------------------------------------------------------------===//

#include "mlir/Analysis/Presburger/PresburgerRelation.h"
#include "./Utils.h"
#include "mlir/IR/MLIRContext.h"

#include <gmock/gmock.h>
#include <gtest/gtest.h>

using namespace mlir;
using namespace presburger;

/// Construct a PresburgerRelation having `numDims` dimensions and no symbols
/// from the given list of IntegerPolyhedron. Each Poly in `polys` should also
/// have `numDims` dimensions and no symbols, although it can have any number of
/// local ids.
static PresburgerRelation makeRelFromPoly(const PresburgerSpace &space,
                                          ArrayRef<IntegerPolyhedron> polys) {
  PresburgerRelation rel = PresburgerRelation::getEmpty(space);
  for (IntegerPolyhedron poly : polys) {
    poly.convertIdKind(IdKind::Range, 0, space.getNumDomainIds(),
                       IdKind::Domain);
    rel.unionInPlace(poly);
  }
  return rel;
}

TEST(PresburgerRelationTest, valueTests) {
  int values[6] = {1, 2, 3, 4, 5, 6};

  PresburgerSpace space1 = PresburgerSpace::getRelationSpace(2, 1, 2);
  PresburgerSpace space2 = PresburgerSpace::getRelationSpace(2, 1, 1);

  space1.resetValues<int *>();
  space2.resetValues<int *>();

  space1.setValue(IdKind::Domain, 0, &values[0]);
  space1.setValue(IdKind::Domain, 1, &values[1]);
  space1.setValue(IdKind::Range, 0, &values[2]);
  space1.setValue(IdKind::Symbol, 0, &values[3]);
  space1.setValue(IdKind::Symbol, 1, &values[4]);

  space2.setValue(IdKind::Domain, 0, &values[1]);
  space2.setValue(IdKind::Domain, 1, &values[0]);
  space2.setValue(IdKind::Range, 0, &values[2]);
  space1.setValue(IdKind::Symbol, 0, &values[5]);

  PresburgerRelation rel1 = makeRelFromPoly(
      space1, {parsePoly("(x, y, z)[a, b] : (x + y + z - a - b >= 0)"),
               parsePoly("(x, y, z)[a, b] : (x + y + z - a + b >= 0)")});

  PresburgerRelation rel2 =
      makeRelFromPoly(space2, {parsePoly("(y, x, z)[c] : (x + y + z - c == 0)"),
                               parsePoly("(y, x, z)[c] : (x + y + c == 0)")});

  rel1.mergeAndAlign(rel2);
  EXPECT_TRUE(rel1.getSpace().isAligned(rel2.getSpace()));

  PresburgerRelation oldRel1 = rel1;
  rel1.unionInPlace(rel2);
  EXPECT_TRUE(rel1.subtract(rel2).isSubsetOf(oldRel1));
}
