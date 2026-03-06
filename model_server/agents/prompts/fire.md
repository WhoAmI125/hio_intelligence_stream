# Fire Detection Guidelines

This ruleset operates under the `FIRE` scenario. Follow these policies before raising a tier-2 alert.

## Detection Rules
- **Rule 1**: Visible flame, glowing embers, or active combustion must be present in the scene.
- **Rule 2**: Dense smoke originating from a specific source (not fog or steam) should be flagged.
- **Rule 3**: Rapidly spreading orange/red glow with rising smoke strongly indicates fire.
- **Rule 4**: Sparks or electrical arcing from equipment should trigger a fire warning.

## Confirmation Signals
- **Signal A**: Smoke concentration increasing over consecutive frames.
- **Signal B**: People evacuating or pointing toward the fire source.
- **Signal C**: Visible damage to objects (melting, charring, discoloration).

## Exclusion Rules
- **Rule 5**: Cooking flames in a kitchen setting (controlled) should NOT trigger FIRE unless abnormally large.
- **Rule 6**: Candles, lighters, or cigarettes should NOT trigger FIRE.
- **Rule 7**: Red/orange decorative lighting or LED displays should NOT trigger FIRE.
- **Rule 8**: Steam from hot beverages or air conditioning vents should NOT be confused with smoke.
- **Rule 9**: Reflections of sunlight (lens flare) on glass surfaces should NOT trigger FIRE.

## Severity Escalation
- Small contained flame → LOW priority (monitor)
- Spreading flame with smoke → HIGH priority (immediate Tier-2 validation)
- Structural fire with evacuation → CRITICAL (alert + clip save)
