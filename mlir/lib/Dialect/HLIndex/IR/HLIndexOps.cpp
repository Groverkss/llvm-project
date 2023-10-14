//===- HLIndexOps.cpp - HLIndex Operations --------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "mlir/Dialect/HLIndex/IR/HLIndexOps.h"
#include "mlir/IR/Matchers.h"
#include "mlir/IR/PatternMatch.h"
#include "llvm/Support/Debug.h"

using namespace mlir;
using namespace mlir::hl_index;

#define DEBUG_TYPE "hlindex-ops"

//===----------------------------------------------------------------------===//
// HLIndexApplyOp
//===----------------------------------------------------------------------===//

ParseResult HLIndexApplyOp::parse(OpAsmParser &parser, OperationState &result) {
  auto &builder = parser.getBuilder();
  auto indexTy = builder.getIndexType();

  IndexMapAttr mapAttr;
  SmallVector<OpAsmParser::UnresolvedOperand, 8> opInfos;
  if (parser.parseAttribute(mapAttr, "map", result.attributes) ||
      parser.parseOperandList(opInfos, OpAsmParser::Delimiter::Paren) ||
      parser.resolveOperands(opInfos, indexTy, result.operands) ||
      parser.parseOptionalAttrDict(result.attributes))
    return failure();
  auto map = mapAttr.getValue();

  if (map.getNumSymbols() != opInfos.size()) {
    return parser.emitError(
        parser.getNameLoc(),
        "number of operands should be equal to number of symbols in the map");
  }

  result.types.append(map.getNumResults(), indexTy);
  return success();
}

void HLIndexApplyOp::print(OpAsmPrinter &p) {
  p << " " << getMapAttr();
  p << "(" << getOperands() << ")";
  p.printOptionalAttrDict((*this)->getAttrs(), /*elidedAttrs=*/{"map"});
}

LogicalResult HLIndexApplyOp::verify() {
  AffineMap indexMap = getIndexMap();

  // Verify that the map only produces one result.
  if (indexMap.getNumResults() != 1)
    return emitOpError("mapping must produce one value");

  // Verify that operand count matches affine map symbol count.
  if (getNumOperands() != indexMap.getNumSymbols())
    return emitOpError("operand count and variable count must match");

  return success();
}

/// Materialize a single constant operation from a given attribute value with
/// the desired resultant type.
Operation *HLIndexDialect::materializeConstant(OpBuilder &builder,
                                               Attribute value, Type type,
                                               Location loc) {
  return arith::ConstantOp::materialize(builder, value, type, loc);
}

OpFoldResult HLIndexApplyOp::fold(FoldAdaptor adaptor) {
  auto map = getIndexMap();

  // Fold variables to constant values.
  auto expr = map.getResult(0);
  if (auto sym = expr.dyn_cast<AffineSymbolExpr>())
    return getOperand(sym.getPosition());

  // Otherwise, default to folding the map.
  SmallVector<Attribute, 1> result;
  if (failed(map.constantFold(adaptor.getMapOperands(), result)))
    return {};
  return result[0];
}

static void canonicalizeIndexMapWithValues(AffineMap *indexMap,
                                           SmallVectorImpl<Value> *values) {
  assert(indexMap->getNumInputs() == values->size() &&
         "map inputs must match number of values");

  if (!indexMap || values->empty())
    return;

  // Check to see what vars are used.
  llvm::SmallBitVector usedVars(indexMap->getNumInputs());
  indexMap->walkExprs([&](AffineExpr expr) {
    if (auto symExpr = expr.dyn_cast<AffineSymbolExpr>())
      usedVars[symExpr.getPosition()] = true;
  });

  auto *context = indexMap->getContext();

  SmallVector<Value, 8> resultValues;
  resultValues.reserve(values->size());

  llvm::SmallDenseMap<Value, AffineExpr, 8> seenVars;
  SmallVector<AffineExpr, 8> varRemapping(indexMap->getNumInputs());
  unsigned nextSym = 0;
  for (unsigned i = 0, e = indexMap->getNumInputs(); i != e; ++i) {
    if (!usedVars[i])
      continue;
    // Handle constant values.
    IntegerAttr valueCst;
    if (matchPattern((*values)[i], m_Constant(&valueCst))) {
      varRemapping[i] =
          getAffineConstantExpr(valueCst.getValue().getSExtValue(), context);
      continue;
    }

    // Remap var positions for duplicate values.
    auto it = seenVars.find((*values)[i + indexMap->getNumDims()]);
    if (it == seenVars.end()) {
      varRemapping[i] = getAffineSymbolExpr(nextSym++, context);
      resultValues.push_back((*values)[i]);
      seenVars.insert(std::make_pair((*values)[i], varRemapping[i]));
    } else {
      varRemapping[i] = it->second;
    }
  }
  *indexMap = indexMap->replaceDimsAndSymbols({}, varRemapping, 0, nextSym);
  *values = resultValues;
}

namespace {

struct SimplifyHLIndexApplyOp : public OpRewritePattern<HLIndexApplyOp> {
  using OpRewritePattern<HLIndexApplyOp>::OpRewritePattern;

  LogicalResult matchAndRewrite(HLIndexApplyOp applyOp,
                                PatternRewriter &rewriter) const override {
    AffineMap map = applyOp.getIndexMap();
    AffineMap oldMap = map;

    auto oldOperands = applyOp.getMapOperands();
    SmallVector<Value, 8> resultOperands(oldOperands);
    // composeAffineMapAndOperands(&map, &resultOperands);
    canonicalizeIndexMapWithValues(&map, &resultOperands);
    if (map == oldMap && std::equal(oldOperands.begin(), oldOperands.end(),
                                    resultOperands.begin()))
      return failure();

    rewriter.replaceOpWithNewOp<HLIndexApplyOp>(
        applyOp, applyOp->getResultTypes(), IndexMapAttr::get(map),
        resultOperands);

    return success();
  }
};

} // namespace

void HLIndexApplyOp::getCanonicalizationPatterns(RewritePatternSet &results,
                                                 MLIRContext *context) {
  results.add<SimplifyHLIndexApplyOp>(context);
}

