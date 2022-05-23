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

static IntegerRelation parseRelationFromSet(StringRef set, unsigned numDomain) {
  IntegerRelation rel = parsePoly(set);

  rel.convertIdKind(IdKind::SetDim, 0, numDomain, IdKind::Domain);

  return rel;
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

TEST(IntegerRelationTest, inverse) {
  IntegerRelation rel =
      parseRel("(x, y)[N, M] -> (x + y)",
               "(x, y, z)[N, M] : (x >= 0, N - x >= 0, y >= 0, M - y >= 0)");

  IntegerRelation inverseRel =
      parseRelationFromSet("(z, x, y)[N, M]  : (x >= 0, N - x >= 0, y >= 0, M "
                           "- y >= 0, x + y - z == 0)",
                           1);

  rel.inverse();

  EXPECT_TRUE(rel.isEqual(inverseRel));
}

TEST(IntegerRelationTest, intersectDomainAndRange) {
  IntegerRelation rel = parseRelationFromSet(
      "(x, y, z)[N, M]: (y floordiv 2 - N >= 0, z floordiv 5 - M"
      ">= 0, x + y + z floordiv 7 == 0)",
      1);

  {
    IntegerPolyhedron poly = parsePoly("(x)[N, M] : (x >= 0, M - x - 1 >= 0)");

    IntegerRelation expectedRel = parseRelationFromSet(
        "(x, y, z)[N, M]: (y floordiv 2 - N >= 0, z floordiv 5 - M"
        ">= 0, x + y + z floordiv 7 == 0, x >= 0, M - x - 1 >= 0)",
        1);

    IntegerRelation copyRel = rel;
    copyRel.intersectDomain(poly);
    EXPECT_TRUE(copyRel.isEqual(expectedRel));
  }

  {
    IntegerPolyhedron poly =
        parsePoly("(y, z)[N, M] : (y >= 0, M - y - 1 >= 0, y + z == 0)");

    IntegerRelation expectedRel = parseRelationFromSet(
        "(x, y, z)[N, M]: (y floordiv 2 - N >= 0, z floordiv 5 - M"
        ">= 0, x + y + z floordiv 7 == 0, y >= 0, M - y - 1 >= 0, y + z == 0)",
        1);

    IntegerRelation copyRel = rel;
    copyRel.intersectRange(poly);
    EXPECT_TRUE(copyRel.isEqual(expectedRel));
  }
}

TEST(IntegerRelationTest, applyDomainAndRange) {

  {
    IntegerRelation map1 = parseMap("(x, y)[N] -> (x + N, y - N)");
    IntegerRelation map2 = parseMap("(x, y)[N] -> (x + y)");

    map1.applyRange(map2);

    IntegerRelation map3 = parseMap("(x, y)[N] -> (x + y)");

    EXPECT_TRUE(map1.isEqual(map3));
  }

  {
    IntegerRelation map1 = parseMap("(x, y)[N] -> (x - N, y + N)");
    IntegerRelation map2 = parseMap("(x, y)[N] -> (N, N)");

    IntegerRelation map3 =
        parseRelationFromSet("(x, y, r0, r1)[N] : (x - N == 0, y - N == 0)", 2);

    map1.applyDomain(map2);

    EXPECT_TRUE(map1.isEqual(map3));
  }
}

TEST(IntegerRelationTest, mergeAndAlign) {
  int values[3] = {0, 1, 2};

  {
    IntegerRelation map1 =
        parseRelationFromSet("(x, y)[N, M] : (x - N >= 0, y - N - M >= 0)", 1);
    IntegerRelation map2 =
        parseRelationFromSet("(x, y)[K, N] : (x - N >= 0, y - N - K >= 0)", 1);
    map1.resetValues();
    map2.resetValues();

    map1.setValue(IdKind::Symbol, 0, &values[0]);
    map1.setValue(IdKind::Symbol, 1, &values[1]);

    map2.setValue(IdKind::Symbol, 0, &values[2]);
    map2.setValue(IdKind::Symbol, 1, &values[0]);

    map1.mergeAndAlign(IdKind::Symbol, map2);

    EXPECT_EQ(map1.getNumIdKind(IdKind::Symbol), 3u);
    EXPECT_EQ(map2.getNumIdKind(IdKind::Symbol), 3u);
    EXPECT_TRUE(map1.getSpace().isAligned(map2.getSpace()));
    for (unsigned i = 0; i < 3; ++i)
      EXPECT_EQ(map1.getValue<int *>(IdKind::Symbol, i), &values[i]);
  }

  {
    IntegerRelation map1 = parseRelationFromSet(
        "(x, y, z)[N, M] : (x - N >= 0, y - N - M >= 0)", 1);
    IntegerRelation map2 = parseRelationFromSet(
        "(z, y, x)[K, N, M] : (x - N >= 0, y - N - K >= 0)", 2);
    map1.resetValues();
    map2.resetValues();

    map1.setValue(IdKind::Range, 0, &values[0]);
    map1.setValue(IdKind::Range, 1, &values[1]);

    map2.setValue(IdKind::Domain, 0, &values[1]);
    map2.setValue(IdKind::Domain, 1, &values[0]);

    map1.mergeAndAlign(IdKind::Range, IdKind::Domain, map2);

    EXPECT_EQ(map1.getNumIdKind(IdKind::Range), 2u);
    EXPECT_EQ(map2.getNumIdKind(IdKind::Domain), 2u);
    for (unsigned i = 0; i < 2; ++i)
      EXPECT_EQ(map1.getValue<int *>(IdKind::Range, i), &values[i]);
    for (unsigned i = 0; i < 2; ++i)
      EXPECT_EQ(map2.getValue<int *>(IdKind::Domain, i), &values[i]);
  }
}
