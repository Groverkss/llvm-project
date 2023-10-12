//===- HLIndexOps.cpp - HLIndex Operations --------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "mlir/Dialect/HLIndex/IR/HLIndexOps.h"
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
  // Check input and output dimensions match.
  AffineMap affineMap = getIndexMap();

  // Verify that operand count matches affine map dimension and symbol count.
  if (getNumOperands() != affineMap.getNumSymbols())
    return emitOpError("operand count and variable count must match");

  // Verify that the map only produces one result.
  if (affineMap.getNumResults() != 1)
    return emitOpError("mapping must produce one value");

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
