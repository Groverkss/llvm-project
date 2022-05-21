//===- IntegerPolyhedron.cpp - Tests for IntegerPolyhedron class ----------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "mlir/Analysis/Presburger/IntegerRelation.h"
#include "./Utils.h"

#include <gmock/gmock.h>
#include <gtest/gtest.h>

#include <numeric>

using namespace mlir;
using namespace presburger;

static IntegerRelation parseRel(StringRef mapStr, StringRef conditionsStr) {
  IntegerRelation map = parseMap(mapStr);
  IntegerRelation set = parsePoly(conditionsStr);

  set.convertIdKind(IdKind::SetDim, 0, map.getNumIdKind(IdKind::Domain),
                    IdKind::Domain);

  return map.intersect(set);
}

TEST(IntegerRelationTest, getDomainAndRangeSet) {
  IntegerRelation rel =
      parseRel("(x)[N] -> (x + 10)", "(x, xr)[N] : (xr >= 0, N - xr >= 0)");

  IntegerPolyhedron domainSet = rel.getDomainSet();

  IntegerPolyhedron expectedDomainSet =
      parsePoly("(x)[N] : (x + 10 >= 0, N - x - 10 >= 0)");

  EXPECT_TRUE(domainSet.isEqual(expectedDomainSet));

  IntegerPolyhedron rangeSet = rel.getRangeSet();

  IntegerPolyhedron expectedRangeSet =
      parsePoly("(x)[N] : (x >= 0, N - x >= 0)");

  EXPECT_TRUE(rangeSet.isEqual(expectedRangeSet));
}

