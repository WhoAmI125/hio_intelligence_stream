# Cash Scene Detection Guidelines

This ruleset operates under the `CASH_TRANSACTION` scenario. Follow these policies before raising a tier-2 alert.

## Detection Rules
- **Rule 1**: Cashier must be physically engaging with the register or client holding money.
- **Rule 2**: Mere presence of a drawer is not enough; exchange motion must occur.

## Exclusion Rules
- **Rule 3**: Passing receipts or cards should NOT trigger CASH.
- **Rule 4**: Background personnel walking past the register should NOT trigger CASH.
