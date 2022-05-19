//===- PresburgerSpaceTest.cpp - Tests for PresburgerSpace ----------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "mlir/Analysis/Presburger/PresburgerSpace.h"
#include <gmock/gmock.h>
#include <gtest/gtest.h>

using namespace mlir;
using namespace presburger;

TEST(PresburgerSpaceTest, insertId) {
  PresburgerSpace space = PresburgerSpace::getRelationSpace(2, 2, 1);

  // Try inserting 2 domain ids.
  space.insertId(IdKind::Domain, 0, 2);
  EXPECT_EQ(space.getNumDomainIds(), 4u);

  // Try inserting 1 range ids.
  space.insertId(IdKind::Range, 0, 1);
  EXPECT_EQ(space.getNumRangeIds(), 3u);
}

TEST(PresburgerSpaceTest, insertIdSet) {
  PresburgerSpace space = PresburgerSpace::getSetSpace(2, 1);

  // Try inserting 2 dimension ids. The space should have 4 range ids since
  // spaces which do not distinguish between domain, range are implemented like
  // this.
  space.insertId(IdKind::SetDim, 0, 2);
  EXPECT_EQ(space.getNumRangeIds(), 4u);
}

TEST(PresburgerSpaceTest, removeIdRange) {
  PresburgerSpace space = PresburgerSpace::getRelationSpace(2, 1, 3);

  // Remove 1 domain identifier.
  space.removeIdRange(IdKind::Domain, 0, 1);
  EXPECT_EQ(space.getNumDomainIds(), 1u);

  // Remove 1 symbol and 1 range identifier.
  space.removeIdRange(IdKind::Symbol, 0, 1);
  space.removeIdRange(IdKind::Range, 0, 1);
  EXPECT_EQ(space.getNumDomainIds(), 1u);
  EXPECT_EQ(space.getNumRangeIds(), 0u);
  EXPECT_EQ(space.getNumSymbolIds(), 2u);
}

TEST(PresburgerSpaceTest, insertIdValue) {
  PresburgerSpace space = PresburgerSpace::getRelationSpace(2, 2, 1, 0, true);

  // Attach value to domain ids.
  int values[2] = {0, 1};
  space.setValue<int *>(IdKind::Domain, 0, &values[0]);
  space.setValue<int *>(IdKind::Domain, 1, &values[1]);

  // Try inserting 2 domain ids.
  space.insertId(IdKind::Domain, 0, 2);
  EXPECT_EQ(space.getNumDomainIds(), 4u);

  // Try inserting 1 range ids.
  space.insertId(IdKind::Range, 0, 1);
  EXPECT_EQ(space.getNumRangeIds(), 3u);

  // Check if the values for the old ids are still attached properly.
  EXPECT_EQ(*space.getValue<int *>(IdKind::Domain, 2), values[0]);
  EXPECT_EQ(*space.getValue<int *>(IdKind::Domain, 3), values[1]);
}

TEST(PresburgerSpaceTest, removeIdRangeValue) {
  PresburgerSpace space = PresburgerSpace::getRelationSpace(2, 1, 3, 0, true);

  int values[6] = {0, 1, 2, 3, 4, 5};

  // Attach values to domain identifiers.
  space.setValue<int *>(IdKind::Domain, 0, &values[0]);
  space.setValue<int *>(IdKind::Domain, 1, &values[1]);

  // Attach values to range identifiers.
  space.setValue<int *>(IdKind::Range, 0, &values[2]);

  // Attach values to symbol identifiers.
  space.setValue<int *>(IdKind::Symbol, 0, &values[3]);
  space.setValue<int *>(IdKind::Symbol, 1, &values[4]);
  space.setValue<int *>(IdKind::Symbol, 2, &values[5]);

  // Remove 1 domain identifier.
  space.removeIdRange(IdKind::Domain, 0, 1);
  EXPECT_EQ(space.getNumDomainIds(), 1u);

  // Remove 1 symbol and 1 range identifier.
  space.removeIdRange(IdKind::Symbol, 0, 1);
  space.removeIdRange(IdKind::Range, 0, 1);
  EXPECT_EQ(space.getNumDomainIds(), 1u);
  EXPECT_EQ(space.getNumRangeIds(), 0u);
  EXPECT_EQ(space.getNumSymbolIds(), 2u);

  // Check if domain values are attached properly.
  EXPECT_EQ(*space.getValue<int *>(IdKind::Domain, 0), values[1]);

  // Check if symbol values are attached properly.
  EXPECT_EQ(*space.getValue<int *>(IdKind::Range, 0), values[4]);
  EXPECT_EQ(*space.getValue<int *>(IdKind::Range, 1), values[5]);
}
