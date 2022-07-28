//===- PWMAFunction.cpp - MLIR PWMAFunction Class -------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "mlir/Analysis/Presburger/PWMAFunction.h"
#include "mlir/Analysis/Presburger/Simplex.h"

using namespace mlir;
using namespace presburger;

void MultiAffineFunction::assertIsConsistent() const {
  assert(space.getNumVars() - space.getNumRangeVars() + 1 ==
             output.getNumColumns() &&
         "Inconsistent number of output columns");
  assert(space.getNumDomainVars() + space.getNumSymbolVars() ==
             divs.getNumNonDivs() &&
         "Inconsistent number of non-division variables in divs");
  assert(space.getNumRangeVars() == output.getNumRows() &&
         "Inconsistent number of output rows");
  assert(space.getNumLocalVars() == divs.getNumDivs() &&
         "Inconsistent number of divisions.");
  assert(divs.hasAllReprs() && "All divisions should have a representation");
}

// Return the result of subtracting the two given vectors pointwise.
// The vectors must be of the same size.
// e.g., [3, 4, 6] - [2, 5, 1] = [1, -1, 5].
static SmallVector<int64_t, 8> subtractExprs(ArrayRef<int64_t> vecA,
                                             ArrayRef<int64_t> vecB) {
  assert(vecA.size() == vecB.size() &&
         "Cannot subtract vectors of differing lengths!");
  SmallVector<int64_t, 8> result;
  result.reserve(vecA.size());
  for (unsigned i = 0, e = vecA.size(); i < e; ++i)
    result.push_back(vecA[i] - vecB[i]);
  return result;
}

PresburgerSet PWMAFunction::getDomain() const {
  PresburgerSet domain = PresburgerSet::getEmpty(getDomainSpace());
  for (const Piece &piece : pieces)
    domain.unionInPlace(piece.domain);
  return domain;
}

void MultiAffineFunction::print(raw_ostream &os) const {
  space.print(os);
  os << "Division Representation:\n";
  divs.print(os);
  os << "Output:\n";
  output.print(os);
}

void MultiAffineFunction::dump() const { print(llvm::errs()); }

SmallVector<int64_t, 8>
MultiAffineFunction::valueAt(ArrayRef<int64_t> point) const {
  assert(point.size() == getNumDomainVars() + getNumSymbolVars());

  SmallVector<int64_t, 8> result(getNumOutputs());
  SmallVector<Optional<int64_t>, 4> divValues = divs.divValuesAt(point);
  for (unsigned i = 0, e = getNumOutputs(); i < e; ++i) {
    // Evaluate at non div-points.
    for (unsigned j = 0, f = point.size(); j < f; ++j)
      result[i] += output(i, j) * point[j];
    // Evalue at div points.
    for (unsigned j = 0, f = getNumDivs(); j < f; ++j) {
      assert(divValues[j].hasValue() && "All divisions should have a value.");
      result[i] += output(i, j + divs.getDivOffset()) * divValues[j].getValue();
    }
    // Add constant.
    result[i] += getOutputExpr(i).back();
  }

  return result;
}

bool MultiAffineFunction::isEqual(const MultiAffineFunction &other) const {
  assert(space.isCompatible(other.space) &&
         "Spaces should be compatible for equality check.");
  return getAsRelation().isEqual(other.getAsRelation());
}

bool MultiAffineFunction::isEqual(const MultiAffineFunction &other,
                                  const IntegerPolyhedron &restrict) const {
  assert(space.isCompatible(other.space) &&
         "Spaces should be compatible for equality check.");
  IntegerRelation restrictedThis = getAsRelation();
  restrictedThis.intersectDomain(restrict);

  IntegerRelation restrictedOther = other.getAsRelation();
  restrictedOther.intersectDomain(restrict);

  return restrictedThis.isEqual(restrictedOther);
}

bool MultiAffineFunction::isEqual(const MultiAffineFunction &other,
                                  const PresburgerSet &restrict) const {
  assert(space.isCompatible(other.space) &&
         "Spaces should be compatible for equality check.");
  return std::all_of(restrict.getAllDisjuncts().begin(),
                     restrict.getAllDisjuncts().end(),
                     [&](const IntegerRelation &disjunct) {
                       return isEqual(other, IntegerPolyhedron(disjunct));
                     });
}

void MultiAffineFunction::removeOutputs(unsigned start, unsigned end) {
  assert(end <= getNumOutputs() && "Invalid range");

  if (start >= end)
    return;

  space.removeVarRange(VarKind::Range, start, end);
  output.removeRows(start, end - start);
}

void MultiAffineFunction::mergeLocalVars(MultiAffineFunction &other) {
  assert(space.isCompatible(other.space) && "Funcitons should be compatible");

  unsigned nDivs = getNumDivs();
  unsigned divOffset = divs.getDivOffset();

  other.divs.insertDiv(0, nDivs);

  SmallVector<int64_t, 8> div(other.divs.getNumVars() + 1);
  for (unsigned i = 0; i < nDivs; ++i) {
    // Zero fill.
    std::fill(div.begin(), div.end(), 0);
    // Fill div with dividend from `divs`. Do not fill the constant.
    std::copy(divs.getDividend(i).begin(), divs.getDividend(i).end() - 1,
              div.begin());
    // Fill constant.
    div.back() = divs.getDividend(i).back();
    other.divs.setDiv(i, div, divs.getDenom(i));
  }

  other.space.insertVar(VarKind::Local, 0, nDivs);
  other.output.insertColumns(divOffset, nDivs);

  auto merge = [&](unsigned i, unsigned j) {
    // We only merge from local at pos j to local at pos i, where j > i.
    if (i >= j)
      return false;

    // If i < nDivs, we are trying to merge duplicate divs in `this`. Since we
    // do not want to merge duplicates in `this`, we ignore this call.
    if (j < nDivs)
      return false;

    // Merge things in space and output.
    other.space.removeVarRange(VarKind::Local, j, j + 1);
    other.output.addToColumn(divOffset + i, divOffset + j, 1);
    other.output.removeColumn(divOffset + j);
    return true;
  };

  other.divs.removeDuplicateDivs(merge);

  unsigned newDivs = other.divs.getNumDivs() - nDivs;

  space.insertVar(VarKind::Local, nDivs, newDivs);
  output.insertColumns(divOffset + nDivs, newDivs);
  divs = other.divs;

  // Check consistency.
  assertIsConsistent();
  other.assertIsConsistent();
}

/// Two PWMAFunctions are equal if they have the same dimensionalities,
/// the same domain, and take the same value at every point in the domain.
bool PWMAFunction::isEqual(const PWMAFunction &other) const {
  if (!space.isCompatible(other.space))
    return false;

  if (!this->getDomain().isEqual(other.getDomain()))
    return false;

  // Check if, whenever the domains of a piece of `this` and a piece of `other`
  // overlap, they take the same output value. If `this` and `other` have the
  // same domain (checked above), then this check passes iff the two functions
  // have the same output at every point in the domain.
  for (const Piece &pieceA : pieces) {
    for (const Piece &pieceB : other.pieces) {
      // Check equality inside intersection of domain.
      PresburgerSet commonDomain = pieceA.domain.intersect(pieceB.domain);
      if (!pieceA.output.isEqual(pieceB.output, commonDomain)) {
        return false;
      }
    }
  }

  return true;
}

void PWMAFunction::addPiece(const Piece &piece) {
  assert(piece.domain.getSpace().isCompatible(piece.output.getDomainSpace()) &&
         "Domain spaces in piece should match");
  pieces.push_back(piece);
}

void PWMAFunction::print(raw_ostream &os) const {
  space.print(os);
  os << getNumPieces() << " pieces:\n";
  for (const Piece &piece : pieces) {
    os << "Domain of piece:\n";
    piece.domain.print(os);
    os << "Output of piece\n";
    piece.output.print(os);
  }
}

void PWMAFunction::dump() const { print(llvm::errs()); }

PWMAFunction PWMAFunction::unionFunction(
    const PWMAFunction &func,
    llvm::function_ref<PresburgerSet(Piece maf1, Piece maf2)> tiebreak) const {
  assert(getNumOutputs() == func.getNumOutputs() &&
         "Range of functions should be same.");
  assert(getSpace().isCompatible(func.getSpace()) &&
         "Space is not compatible.");

  // The algorithm used here is as follows:
  // - Add the output of pieceB for the part of the domain where both pieceA and
  //   pieceB are defined, and `tiebreak` chooses the output of pieceB.
  // - Add the output of pieceA, where pieceB is not defined or `tiebreak`
  // chooses
  //   pieceA over pieceB.
  // - Add the output of pieceB, where pieceA is not defined.

  // Add parts of the common domain where pieceB's output is used. Also
  // add all the parts where pieceA's output is used, both common and
  // non-common.
  PWMAFunction result(getSpace());
  for (const Piece &pieceA : pieces) {
    PresburgerSet dom(pieceA.domain);
    for (const Piece &pieceB : func.pieces) {
      PresburgerSet better = tiebreak(pieceB, pieceA);
      // Add the output of pieceB, where it is better than output of pieceA.
      // The disjuncts in "better" will be disjoint as tiebreak should gurantee
      // that.
      result.addPiece({better, pieceB.output});
      dom = dom.subtract(better);
    }
    // Add output of pieceA, where it is better than pieceB, or pieceB is not
    // defined.
    //
    // `dom` here is guranteed to be disjoint from already added pieces
    // because because the pieces added before are either:
    // - Subsets of the domain of other MAFs in `this`, which are guranteed
    //   to be disjoint from `dom`, or
    // - They are one of the pieces added for `pieceB`, and we have been
    //   subtracting all such pieces from `dom`, so `dom` is disjoint from those
    //   pieces as well.
    result.addPiece({dom, pieceA.output});
  }

  // Add parts of pieceB which are not shared with pieceA.
  PresburgerSet dom = getDomain();
  for (const Piece &pieceB : func.pieces)
    result.addPiece({pieceB.domain.subtract(dom), pieceB.output});

  return result;
}

/// A tiebreak function which breaks ties by comparing the outputs
/// lexicographically. If `lexMin` is true, then the ties are broken by
/// taking the lexicographically smaller output and otherwise, by taking the
/// lexicographically larger output.
template <bool lexMin>
static PresburgerSet tiebreakLex(const PWMAFunction::Piece &pieceA,
                                 const PWMAFunction::Piece &pieceB) {
  // TODO: Support local variables here.
  assert(pieceA.domain.getSpace().isCompatible(pieceB.domain.getSpace()) &&
         "Number of outputs of both functions should be same.");
  assert(pieceA.domain.getSpace().getNumLocalVars() == 0 &&
         "Local variables are not supported yet.");

  PresburgerSpace compatibleSpace = pieceA.domain.getSpace();
  const PresburgerSpace &space = pieceA.domain.getSpace();

  // We first create the set `result`, corresponding to the set where output
  // of pieceA is lexicographically larger/smaller than pieceB. This is done by
  // creating a PresburgerSet with the following constraints:
  //
  //    (outA[0] > outB[0]) U
  //    (outA[0] = outB[0], outA[1] > outA[1]) U
  //    (outA[0] = outB[0], outA[1] = outA[1], outA[2] > outA[2]) U
  //    ...
  //    (outA[0] = outB[0], ..., outA[n-2] = outB[n-2], outA[n-1] > outB[n-1])
  //
  // where `n` is the number of outputs.
  // If `lexMin` is set, the complement inequality is used:
  //
  //    (outA[0] < outB[0]) U
  //    (outA[0] = outB[0], outA[1] < outA[1]) U
  //    (outA[0] = outB[0], outA[1] = outA[1], outA[2] < outA[2]) U
  //    ...
  //    (outA[0] = outB[0], ..., outA[n-2] = outB[n-2], outA[n-1] < outB[n-1])
  PresburgerSet result = PresburgerSet::getEmpty(compatibleSpace);
  IntegerPolyhedron levelSet(
      /*numReservedInequalities=*/1,
      /*numReservedEqualities=*/pieceA.output.getNumOutputs(),
      /*numReservedCols=*/space.getNumVars() + 1, space);
  for (unsigned level = 0; level < pieceA.output.getNumOutputs(); ++level) {

    // Create the expression `outA - outB` for this level.
    SmallVector<int64_t, 8> subExpr = subtractExprs(
        pieceA.output.getOutputExpr(level), pieceB.output.getOutputExpr(level));

    if (lexMin) {
      // For lexMin, we add an upper bound of -1:
      //        outA - outB <= -1
      //        outA <= outB - 1
      //        outA < outB
      levelSet.addBound(IntegerPolyhedron::BoundType::UB, subExpr, -1);
    } else {
      // For lexMax, we add a lower bound of 1:
      //        outA - outB >= 1
      //        outA > outB + 1
      //        outA > outB
      levelSet.addBound(IntegerPolyhedron::BoundType::LB, subExpr, 1);
    }

    // Union the set with the result.
    result.unionInPlace(levelSet);
    // There is only 1 inequality in `levelSet`, so the index is always 0.
    levelSet.removeInequality(0);
    // Add equality `outA - outB == 0` for this level for next iteration.
    levelSet.addEquality(subExpr);
  }

  // We then intersect `result` with the domain of pieceA and pieceB, to only
  // tiebreak on the domain where both are defined.
  result = result.intersect(pieceA.domain).intersect(pieceB.domain);

  return result;
}

PWMAFunction PWMAFunction::unionLexMin(const PWMAFunction &func) {
  return unionFunction(func, tiebreakLex</*lexMin=*/true>);
}

PWMAFunction PWMAFunction::unionLexMax(const PWMAFunction &func) {
  return unionFunction(func, tiebreakLex</*lexMin=*/false>);
}

void MultiAffineFunction::subtract(const MultiAffineFunction &other) {
  assert(space.isCompatible(other.space) &&
         "Spaces should be compatible for subtraction.");

  MultiAffineFunction copyOther = other;
  mergeLocalVars(copyOther);
  for (unsigned i = 0, e = getNumOutputs(); i < e; ++i)
    output.addToRow(i, copyOther.getOutputExpr(i), -1);

  // Check consistency.
  assertIsConsistent();
}

IntegerRelation MultiAffineFunction::getAsRelation() const {
  IntegerRelation result(PresburgerSpace::getRelationSpace(
      space.getNumDomainVars(), 0, space.getNumSymbolVars(),
      space.getNumLocalVars()));

  // Add division inequalities.
  for (unsigned i = 0, e = getNumDivs(); i < e; ++i) {
    result.addInequality(getDivUpperBound(divs.getDividend(i), divs.getDenom(i),
                                          divs.getDivOffset() + i));
    result.addInequality(getDivLowerBound(divs.getDividend(i), divs.getDenom(i),
                                          divs.getDivOffset() + i));
  }

  // Add output equalities.
  result.insertVar(VarKind::Range, 0, getNumOutputs());
  SmallVector<int64_t, 8> eq(result.getNumCols());
  for (unsigned i = 0, e = getNumOutputs(); i < e; ++i) {
    // Zero fill.
    std::fill(eq.begin(), eq.end(), 0);
    // Fill equality.
    for (unsigned j = 0, f = getNumDomainVars(); j < f; ++j)
      eq[j] = output(i, j);
    for (unsigned j = getNumDomainVars(), f = output.getNumColumns(); j < f;
         ++j)
      eq[j + getNumOutputs()] = output(i, j);
    // Set this dimension to -1 to equate lhs and rhs and add equality.
    eq[getNumDomainVars() + i] = -1;
    result.addEquality(eq);
  }

  return result;
}

void PWMAFunction::removeOutputs(unsigned start, unsigned end) {
  space.removeVarRange(VarKind::Range, start, end);
  for (Piece &piece : pieces)
    piece.output.removeOutputs(start, end);
}

Optional<SmallVector<int64_t, 8>>
PWMAFunction::valueAt(ArrayRef<int64_t> point) const {
  assert(point.size() == getNumDomainVars() + getNumSymbolVars());

  for (const Piece &piece : pieces)
    if (piece.domain.containsPoint(point))
      return piece.output.valueAt(point);
  return None;
}
