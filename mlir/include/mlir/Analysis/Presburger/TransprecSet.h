#ifndef MLIR_ANALYSIS_TRANSPREC_SET_H
#define MLIR_ANALYSIS_TRANSPREC_SET_H

#include "mlir/Analysis/Presburger/Set.h"
#include "mlir/Analysis/Presburger/Coalesce.h"

namespace mlir {
namespace analysis {
namespace presburger {

using BigInt = mpz_class;

class TransprecSet {
public:
  TransprecSet() {};
  TransprecSet(PresburgerSet<int16_t> set) : setvar(std::move(set)) {}
  TransprecSet(PresburgerSet<int64_t> set) : setvar(std::move(set)) {}
  TransprecSet(PresburgerSet<BigInt> set) : setvar(std::move(set)) {}

  static void harmonizePrecisions(TransprecSet &a, TransprecSet &b) {
    while (a.setvar.index() < b.setvar.index())
      a.increasePrecision();
    while (a.setvar.index() > b.setvar.index())
      b.increasePrecision();
  }

  void increasePrecision() {
    if (std::holds_alternative<PresburgerSet<int16_t>>(setvar)) {
      setvar = PresburgerSet<int64_t>(std::get<PresburgerSet<int16_t>>(setvar));
    } else if (std::holds_alternative<PresburgerSet<int64_t>>(setvar)) {
      setvar = PresburgerSet<BigInt>(std::get<PresburgerSet<int64_t>>(setvar));
    } else {
      llvm_unreachable("GMP overflowed??");
    }
  }

  void unionSet(TransprecSet &set) {
    harmonizePrecisions(*this, set);
    std::visit([&set](auto &&thisPS) {
      std::visit([&thisPS](auto &&oPS) {
        using typeA = std::decay_t<decltype(thisPS)>;
        using typeB = std::decay_t<decltype(oPS)>;

        if constexpr (std::is_same<typeA, typeB>::value)
          thisPS.unionSet(oPS);
        else
          llvm_unreachable("Types not harmonized!");
      }, set.setvar);
    }, setvar);
  }

  void intersectSet(TransprecSet &set) {
    harmonizePrecisions(*this, set);
    std::visit([&](auto &&thisPS) {
      std::visit([&](auto &&oPS) {
        using typeA = std::decay_t<decltype(thisPS)>;
        using typeB = std::decay_t<decltype(oPS)>;

        if constexpr (std::is_same<typeA, typeB>::value)
          thisPS.intersectSet(oPS);
        else
          llvm_unreachable("Types not harmonized!");
      }, set.setvar);
    }, setvar);
  }

  void subtract(TransprecSet &set) {
    harmonizePrecisions(*this, set);
    std::visit([&](auto &&thisPS) {
      std::visit([&](auto &&oPS) {
        try {
          using typeA = std::decay_t<decltype(thisPS)>;
          using typeB = std::decay_t<decltype(oPS)>;

          if constexpr (std::is_same<typeA, typeB>::value)
            thisPS.subtract(oPS);
          else
            llvm_unreachable("Types not harmonized!");
        } catch (const std::overflow_error &e) {
          increasePrecision();
          set.increasePrecision();
          this->subtract(set);
        }
      }, set.setvar);
    }, setvar);
  }

  bool equal(TransprecSet &set) {
    harmonizePrecisions(*this, set);
    return std::visit([&](auto &&thisPS) {
      return std::visit([&](auto &&oPS) {
        try {
          using typeA = std::decay_t<decltype(thisPS)>;
          using typeB = std::decay_t<decltype(oPS)>;

          if constexpr (std::is_same<typeA, typeB>::value)
            return typeA::equal(thisPS, oPS);
          else
            llvm_unreachable("Types not harmonized!");
        } catch (const std::overflow_error &e) {
          increasePrecision();
          set.increasePrecision();
          return this->equal(set);
        }
      }, set.setvar);
    }, setvar);
  }

  TransprecSet coalesce() {
    return std::visit([this](auto &&set) {
      try {
        return TransprecSet(mlir::coalesce(set));
      } catch (const std::overflow_error &e) {
        increasePrecision();
        return this->coalesce();
      }
    }, setvar);
  }

  TransprecSet complement() {
    return std::visit([this](auto &&set) {
      try {
        using Set = std::decay_t<decltype(set)>;
        return TransprecSet(Set::complement(set));
      } catch (const std::overflow_error &e) {
        increasePrecision();
        return this->complement();
      }
    }, setvar);
  }

  TransprecSet eliminateExistentials() {
    return std::visit([this](auto &&set) {
      try {
        using Set = std::decay_t<decltype(set)>;
        return TransprecSet(Set::eliminateExistentials(set));
      } catch (const std::overflow_error &e) {
        increasePrecision();
        return this->eliminateExistentials();
      }
    }, setvar);
  }

  bool isIntegerEmpty() {
    return std::visit([this](auto &&set) {
      try {
        return set.isIntegerEmpty();
      } catch (const std::overflow_error &e) {
        increasePrecision();
        return this->isIntegerEmpty();
      }
    }, setvar);
  }

  void dumpISL() {
    std::visit([](auto &&set) { set.dumpISL(); }, setvar);
  }

  std::variant<PresburgerSet<int16_t>, PresburgerSet<int64_t>, PresburgerSet<BigInt>> setvar;
};

} // namespace presburger
} // namespace analysis
} // namespace mlir

#endif // MLIR_ANALYSIS_PRESBURGER_SET_H
