//===- PresburgerSpace.cpp - MLIR PresburgerSpace Class -------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "mlir/Analysis/Presburger/PresburgerSpace.h"
#include "llvm/ADT/ArrayRef.h"
#include <algorithm>
#include <cassert>

using namespace mlir;
using namespace presburger;

AbstractIdValue *&PresburgerSpace::atIdValue(IdKind kind, unsigned pos) {
  assert(pos <= getNumIdKind(kind));
  return idValues[getIdKindOffset(kind) + pos];
}

unsigned PresburgerSpace::getNumIdKind(IdKind kind) const {
  if (kind == IdKind::Domain)
    return getNumDomainIds();
  if (kind == IdKind::Range)
    return getNumRangeIds();
  if (kind == IdKind::Symbol)
    return getNumSymbolIds();
  if (kind == IdKind::Local)
    return numLocals;
  llvm_unreachable("IdKind does not exist!");
}

unsigned PresburgerSpace::getIdKindOffset(IdKind kind) const {
  if (kind == IdKind::Domain)
    return 0;
  if (kind == IdKind::Range)
    return getNumDomainIds();
  if (kind == IdKind::Symbol)
    return getNumDimIds();
  if (kind == IdKind::Local)
    return getNumDimAndSymbolIds();
  llvm_unreachable("IdKind does not exist!");
}

unsigned PresburgerSpace::getIdKindEnd(IdKind kind) const {
  return getIdKindOffset(kind) + getNumIdKind(kind);
}

unsigned PresburgerSpace::getIdKindOverlap(IdKind kind, unsigned idStart,
                                           unsigned idLimit) const {
  unsigned idRangeStart = getIdKindOffset(kind);
  unsigned idRangeEnd = getIdKindEnd(kind);

  // Compute number of elements in intersection of the ranges [idStart, idLimit)
  // and [idRangeStart, idRangeEnd).
  unsigned overlapStart = std::max(idStart, idRangeStart);
  unsigned overlapEnd = std::min(idLimit, idRangeEnd);

  if (overlapStart > overlapEnd)
    return 0;
  return overlapEnd - overlapStart;
}

IdKind PresburgerSpace::getIdKindAt(unsigned pos) const {
  assert(pos < getNumIds() && "`pos` should represent a valid id position");
  if (pos < getIdKindEnd(IdKind::Domain))
    return IdKind::Domain;
  if (pos < getIdKindEnd(IdKind::Range))
    return IdKind::Range;
  if (pos < getIdKindEnd(IdKind::Symbol))
    return IdKind::Symbol;
  if (pos < getIdKindEnd(IdKind::Local))
    return IdKind::Local;
  llvm_unreachable("`pos` should represent a valid id position");
}

unsigned PresburgerSpace::insertId(IdKind kind, unsigned pos, unsigned num) {
  assert(pos <= getNumIdKind(kind));

  unsigned absolutePos = getIdKindOffset(kind) + pos;
  unsigned offsetPos = getIdKindOffset(kind) + pos;

  switch (kind) {
  case IdKind::Domain:
    numDomain += num;
    idValues.insert(idValues.begin() + offsetPos, num, nullptr);
    break;
  case IdKind::Range:
    numRange += num;
    idValues.insert(idValues.begin() + offsetPos, num, nullptr);
    break;
  case IdKind::Symbol:
    numSymbols += num;
    idValues.insert(idValues.begin() + offsetPos, num, nullptr);
    break;
  case IdKind::Local:
    numLocals += num;
    break;
  }

  return absolutePos;
}

void PresburgerSpace::removeIdRange(IdKind kind, unsigned idStart,
                                    unsigned idLimit) {
  assert(idLimit <= getNumIdKind(kind) && "invalid id limit");

  if (idStart >= idLimit)
    return;

  unsigned numIdsEliminated = idLimit - idStart;
  unsigned offsetStart = getIdKindOffset(kind) + idStart;
  unsigned offsetLimit = getIdKindOffset(kind) + idLimit;

  switch (kind) {
  case IdKind::Domain:
    numDomain -= numIdsEliminated;
    idValues.erase(idValues.begin() + offsetStart,
                   idValues.begin() + offsetLimit);
    break;
  case IdKind::Range:
    numRange -= numIdsEliminated;
    idValues.erase(idValues.begin() + offsetStart,
                   idValues.begin() + offsetLimit);
    break;
  case IdKind::Symbol:
    numSymbols -= numIdsEliminated;
    idValues.erase(idValues.begin() + offsetStart,
                   idValues.begin() + offsetLimit);
    break;
  case IdKind::Local:
    numLocals -= numIdsEliminated;
    break;
  }
}

void PresburgerSpace::truncateIdKind(IdKind kind, unsigned num) {
  unsigned curNum = getNumIdKind(kind);
  assert(num <= curNum && "Can't truncate to more ids!");
  removeIdRange(kind, num, curNum);
}

static bool checkIdValuesEqual(ArrayRef<AbstractIdValue *> values1,
                               ArrayRef<AbstractIdValue *> values2) {
  if (values1.size() != values2.size())
    return false;

  for (unsigned i = 0, e = values1.size(); i < e; ++i) {
    // Empty values are considered equal.
    if (values1[i] == nullptr && values2[i] == nullptr)
      continue;

    // If one of the values is nullptr while other isn't, return false.
    if (values1[i] == nullptr || values2[i] == nullptr)
      return false;

    if (!values1[i]->isEqual(values2[i]))
      return false;
  }
  return true;
}

bool PresburgerSpace::isSpaceCompatible(const PresburgerSpace &other) const {
  return getNumDomainIds() == other.getNumDomainIds() &&
         getNumRangeIds() == other.getNumRangeIds() &&
         getNumSymbolIds() == other.getNumSymbolIds() &&
         checkIdValuesEqual(idValues, other.idValues);
}

bool PresburgerSpace::isSpaceEqual(const PresburgerSpace &other) const {
  return isSpaceCompatible(other) && getNumLocalIds() == other.getNumLocalIds();
}

void PresburgerSpace::setDimSymbolSeparation(unsigned newSymbolCount) {
  assert(newSymbolCount <= getNumDimAndSymbolIds() &&
         "invalid separation position");
  // Due to the way symbols are stored in idValues,
  // we do no need to modify idValues.
  numRange = numRange + numSymbols - newSymbolCount;
  numSymbols = newSymbolCount;
}

void PresburgerSpace::print(llvm::raw_ostream &os) const {
  os << "Domain: " << getNumDomainIds() << ", "
     << "Range: " << getNumRangeIds() << ", "
     << "Symbols: " << getNumSymbolIds() << ", "
     << "Locals: " << getNumLocalIds() << "\n";
}

void PresburgerSpace::dump() const { print(llvm::errs()); }
