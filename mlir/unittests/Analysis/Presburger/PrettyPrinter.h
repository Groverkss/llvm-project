//===- PrettyPrinter.h - Pretty Printer for Presburger library --*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This file defines functions to parse strings into Presburger library
// constructs.
//
//===----------------------------------------------------------------------===//

#ifndef MLIR_UNITTESTS_ANALYSIS_PRESBURGER_PRETTYPRINTER_H
#define MLIR_UNITTESTS_ANALYSIS_PRESBURGER_PRETTYPRINTER_H

#include "mlir/Analysis/Presburger/IntegerRelation.h"
#include "mlir/Analysis/Presburger/PWMAFunction.h"
#include "mlir/Analysis/Presburger/PresburgerRelation.h"
#include "mlir/IR/AffineExpr.h"
#include "mlir/IR/AffineMap.h"
#include "mlir/IR/IntegerSet.h"
#include "mlir/IR/MLIRContext.h"

namespace mlir {
namespace presburger {

template <typename T>
inline void prettyPrint(const T &obj, raw_ostream &os, bool simplify = true);

template <typename T>
inline void prettyDump(const T &obj, bool simplify = true) {
  prettyPrint(obj, llvm::errs(), simplify);
  llvm::errs() << "\n";
}

/// Pretty Print implementations

inline std::pair<unsigned, SmallVector<AffineExpr>>
getDivReprs(const DivisionRepr &divs, unsigned numDims, unsigned numSymbols,
            MLIRContext *context) {

  SmallVector<AffineExpr> divReprs(divs.getNumDivs());
  SmallVector<bool> foundRepr(divs.getNumDivs());
  unsigned numExists = 0;
  bool changed = true;
  while (changed) {
    changed = false;
    for (unsigned i = 0, e = divs.getNumDivs(); i < e; ++i) {
      if (foundRepr[i])
        continue;

      if (!divs.hasRepr(i)) {
        // No explicit representation, create existential.
        divReprs[i] = getAffineDimExpr(numDims + numExists, context);
        foundRepr[i] = true;
        numExists++;
        changed = true;
      } else {
        // Check if any other explicit representation are required before this.
        unsigned j, f;
        for (j = 0, f = divs.getNumDivs(); j < f; ++j)
          if (divs.getDividend(i)[divs.getDivOffset() + j] != 0 &&
              !foundRepr[j])
            break;
        if (j < f)
          continue;

        // Explicit representation can be found.
        AffineExpr dividend = getAffineExprFromFlatForm(
            divs.getDividend(i), numDims, numSymbols, divReprs, context);
        divReprs[i] = dividend.floorDiv(divs.getDenom(i));
        foundRepr[i] = true;
        changed = true;
      }
    }
  }

  return {numExists, divReprs};
}

inline std::pair<SmallVector<AffineExpr>, SmallVector<AffineExpr>>
getConstraintsFromRel(const IntegerRelation &rel, ArrayRef<AffineExpr> divReprs,
                      MLIRContext *context) {
  // Create inequality/equality constraints.
  SmallVector<AffineExpr> ineqs(rel.getNumInequalities());
  SmallVector<AffineExpr> eqs(rel.getNumEqualities());
  for (unsigned i = 0, e = rel.getNumInequalities(); i < e; ++i)
    ineqs[i] =
        getAffineExprFromFlatForm(rel.getInequality(i), rel.getNumDimVars(),
                                  rel.getNumSymbolVars(), divReprs, context);
  for (unsigned i = 0, e = rel.getNumEqualities(); i < e; ++i)
    eqs[i] =
        getAffineExprFromFlatForm(rel.getEquality(i), rel.getNumDimVars(),
                                  rel.getNumSymbolVars(), divReprs, context);
  return {ineqs, eqs};
}

template <bool setSpace>
inline void printSpace(const PresburgerSpace &space, raw_ostream &os) {
  assert((!setSpace || space.getNumDomainVars() == 0) && "Invalid set space.");

  os << "(";
  if (!setSpace) {
    os << "(";
    for (unsigned i = 0, e = space.getNumDomainVars(); i < e; ++i) {
      if (i == e - 1)
        os << "d" << i;
      else
        os << "d" << i << ", ";
    }
    os << ")";
    os << " -> ";
  }

  if (!setSpace)
    os << "(";
  for (unsigned i = 0, e = space.getNumRangeVars(); i < e; ++i) {
    if (i == e - 1)
      os << "d" << i + space.getNumDomainVars();
    else
      os << "d" << i + space.getNumDomainVars() << ", ";
  }
  if (!setSpace)
    os << ")";

  os << ")";

  os << "[";
  for (unsigned i = 0, e = space.getNumSymbolVars(); i < e; ++i) {
    if (i == e - 1)
      os << "s" << i;
    else
      os << "s" << i << ", ";
  }
  os << "]";
}

inline void printConstraints(ArrayRef<AffineExpr> ineqs,
                             ArrayRef<AffineExpr> eqs,
                             const PresburgerSpace &space, unsigned numExists,
                             raw_ostream &os, bool simplify = true) {
  if (numExists != 0) {
    os << "(exists ";
    unsigned offset = space.getNumDimVars();
    for (unsigned i = 0, e = numExists; i < e; ++i) {
      if (i == e - 1)
        os << "d" << i + offset << " : ";
      else
        os << "d" << i + offset << ", ";
    }
  } else {
    os << "(";
  }

  for (unsigned i = 0, e = ineqs.size(); i < e; ++i) {
    if (i == e - 1 && eqs.empty()) {
      ineqs[i].print(os);
      os << " >= 0";
    } else {
      ineqs[i].print(os);
      os << " >= 0, ";
    }
  }

  for (unsigned i = 0, e = eqs.size(); i < e; ++i) {
    if (i == e - 1) {
      eqs[i].print(os);
      os << " == 0";
    } else {
      eqs[i].print(os);
      os << " == 0, ";
    }
  }

  os << ")";
}

inline void printIntegerRelationWithoutSpace(const IntegerRelation &obj,
                                             raw_ostream &os,
                                             bool simplify = true) {
  MLIRContext context(MLIRContext::Threading::DISABLED);
  IntegerRelation poly = obj;

  std::vector<MaybeLocalRepr> reprs(poly.getNumLocalVars());
  DivisionRepr divs = poly.getLocalReprs(&reprs);
  unsigned numExists;
  SmallVector<AffineExpr> divReprs;
  std::tie(numExists, divReprs) = getDivReprs(
      divs, poly.getNumDimVars(), poly.getNumSymbolVars(), &context);

  // TODO: Some equalities can be converted to inequalities.
  // TODO: Only strong upper bounds can be removed. This is not true always
  // here.
  llvm::SmallBitVector redundantIneqs(poly.getNumInequalities());
  for (const MaybeLocalRepr &repr : reprs) {
    if (repr.kind == ReprKind::Inequality) {
      redundantIneqs[repr.repr.inequalityPair.lowerBoundIdx] = true;
      redundantIneqs[repr.repr.inequalityPair.upperBoundIdx] = true;
    }
  }

  for (int i = poly.getNumInequalities() - 1; i >= 0; --i)
    if (redundantIneqs[i])
      poly.removeInequality(i);

  SmallVector<AffineExpr> ineqs, eqs;
  std::tie(ineqs, eqs) = getConstraintsFromRel(poly, divReprs, &context);
  printConstraints(ineqs, eqs, poly.getSpace(), numExists, os);
}

inline void printPresburgerRealtionWithoutSpace(const PresburgerRelation &obj,
                                                raw_ostream &os,
                                                bool simplify = true) {
  if (obj.getNumDisjuncts() == 0)
    os << "false";

  PresburgerRelation prel = obj.coalesce();
  for (unsigned i = 0, e = prel.getNumDisjuncts(); i < e; ++i) {
    printIntegerRelationWithoutSpace(prel.getDisjunct(i), os, simplify);
    if (i != e - 1)
      os << " or ";
  }
}

inline void printMAFWithoutSpace(const MultiAffineFunction &maf,
                                 raw_ostream &os) {
  MLIRContext context(MLIRContext::Threading::DISABLED);
  unsigned numExists;
  SmallVector<AffineExpr> divReprs;
  std::tie(numExists, divReprs) = getDivReprs(
      maf.getDivs(), maf.getNumDomainVars(), maf.getNumSymbolVars(), &context);

  SmallVector<AffineExpr> outputExprs(maf.getNumOutputs());
  for (unsigned i = 0, e = maf.getNumOutputs(); i < e; ++i)
    outputExprs[i] =
        getAffineExprFromFlatForm(maf.getOutputExpr(i), maf.getNumDomainVars(),
                                  maf.getNumSymbolVars(), divReprs, &context);

  os << "(";
  for (AffineExpr expr : outputExprs) {
    expr.print(os);
    if (expr != outputExprs.back())
      os << ", ";
  }
  os << ")";
}

inline void prettyPrint(const IntegerPolyhedron &obj, raw_ostream &os,
                        bool simplify) {
  printSpace</*setSpace=*/true>(obj.getSpace(), os);
  os << " : ";
  printIntegerRelationWithoutSpace(obj, os, simplify);
}

inline void prettyPrint(const IntegerRelation &obj, raw_ostream &os,
                        bool simplify) {
  printSpace</*setSpace=*/false>(obj.getSpace(), os);
  os << " : ";
  printIntegerRelationWithoutSpace(obj, os, simplify);
}

inline void prettyPrint(const PresburgerSet &obj, raw_ostream &os,
                        bool simplify) {
  printSpace</*setSpace=*/true>(obj.getSpace(), os);
  os << " : ";
  printPresburgerRealtionWithoutSpace(obj, os, simplify);
}

inline void prettyPrint(const PresburgerRelation &obj, raw_ostream &os,
                        bool simplify) {
  printSpace</*setSpace=*/false>(obj.getSpace(), os);
  os << " : ";
  printPresburgerRealtionWithoutSpace(obj, os, simplify);
}

inline void prettyPrint(const MultiAffineFunction &obj, raw_ostream &os,
                        bool simplify) {
  printSpace</*setSpace=*/true>(obj.getDomainSpace(), os);
  os << " -> ";
  printMAFWithoutSpace(obj, os);
}

inline void prettyPrint(const PWMAFunction &obj, raw_ostream &os,
                        bool simplify) {
  printSpace</*setSpace=*/true>(obj.getDomainSpace(), os);
  os << " -> {\n";
  for (const PWMAFunction::Piece &piece : obj.getAllPieces()) {
    printPresburgerRealtionWithoutSpace(piece.domain, os, simplify);
    os << " -> ";
    printMAFWithoutSpace(piece.output, os);
    os << "\n";
  }
  os << "}";
}

} // namespace presburger
} // namespace mlir

#endif // MLIR_UNITTESTS_ANALYSIS_PRESBURGER_PRETTYPRINTER_H
