# Harm Reward And Chmmr Baseline Specification

## Status And Authority

This document is the implementation source of truth and progress tracker for:

1. Satellite HP bars on Chmmr satellites.
2. Baseline behaviours for fighting a Chmmr that still has satellites.
3. A planet-avoidance lookahead scaled to each ship's agility.
   3A. Pre-emptible held weapons (Chenjesu, Kohr-Ah, Melnorme).
4. A discount half-life setting in seconds, replacing raw gamma.
   4A. Reward magnitude steps from ±0.001 to ±1.024.
   4B. Epsilon and baseline-use decay as half-lives in batches.
5. A "harm" reward scheme that rewards only lasting harm to the opponent.
6. A gradual refactor of baseline strategies into shared, prioritised
   behaviours.
   6B. A perception delay for computer control: enemy objects are seen only
   once they are a few frames old.

Phases are implemented in the order of the tracker at the end. A later phase
starts only after the earlier phase's acceptance checks pass. Phase 6 is
implemented last, one ship at a time.

Design decisions are recorded under "Decisions". The phase sections are
written to match them.

Created 2026-11-20. Decisions recorded 2026-11-20.

## Verified Game Facts

These were checked in the source on 2026-11-20 and drive the designs below.

### Chmmr satellites

- 3 satellites, 10 HP each (30 total), orbiting 256 from the Chmmr. The orbit
  phase advances one step per frame with a period of 64 frames (2.7 s), so a
  satellite passes any given bearing about every 21 frames (0.9 s).
- Each satellite fires a 1-damage laser every 3 frames (`weapon_wait` 2) at a
  target within 256.
- Target choice (`ChmmrSatellite._select_laser_target`): the opponent ship
  first, if it is in range, visible and not cloaked; otherwise the enemy object
  with the lowest HP, then the nearest.
- Satellites collide with enemy projectiles. They do not collide with ships.
- Satellites block lasers (default `blocks_lasers`).
- Satellites do not regenerate.

### Chmmr weapons

- A1 laser: range 600, 2 damage.
- A2 tractor: an impulse of 12 ÷ target mass toward the Chmmr on each use. It
  needs a visible, inertial target, so cloaked ships and non-inertial ships
  (Arilou, Slylandro) are not pulled. Its cost of 1 and battery regeneration of
  1 per 2 frames allow a use about every other frame.

### Distance rings around the Chmmr

R is the distance from the Chmmr. Ship and satellite sizes shift these numbers
by a few tens of units.

- Out of every satellite's range: R > 512.
- Also out of the Chmmr laser: R > 600.
- A weapon with reach r can hit the near satellite when R ≤ 256 + r.

### Weapons and costs for this work

| Ship | Weapon facts |
|---|---|
| Thraddash | <ul><li>A1: 2 HP shot, reach 1,620, one shot per 13 frames.</li><li>A2 flames: 1 HP, 2 damage each.</li><li>Mass 7.</li></ul> |
| Supox | <ul><li>A1: 1 HP shot, reach 1,080, one shot per 3 frames, the same rate as one satellite's zaps.</li></ul> |
| Pkunk | <ul><li>A1: 1 HP shots, reach 432, front and both side guns per press, can fire every frame.</li><li>12 battery, refilled only by A2.</li><li>Mass 1.</li></ul> |
| Arilou | <ul><li>A1: laser, reach 400. It aims itself at the opponent ship and can't be pointed elsewhere.</li><li>2 energy per shot, 20 battery.</li><li>Not inertial.</li></ul> |
| Vux | <ul><li>A1: laser, reach 644, fixed forward.</li><li>A2 limpets: 1 HP.</li><li>Mass 6.</li></ul> |

Ship costs run from 5 (Shofixti) to 30 (Chmmr, Kohr-Ah, Kzer-Za). Vmax = 30.

### Pkunk rebirth

The chance is `0.7 × 0.3^rebirths`. The expected number of deaths per Pkunk is
about 1.86, so 0.5 kill credit per death averages about 0.93 kills per Pkunk.

### Training returns

- Returns are discounted sums of reward, cut off where the discount weight falls
  below 0.01 (`discount_cutoff_frames`). There is no bootstrapping.
- The default gamma of 0.99 per frame equals a half-life of 2.9 s and a cutoff
  of 459 frames (19 s).
- The match limit is 2,400 frames (100 s). The game runs at 24 FPS.
- The `causal` reward mode already places trainee-caused positive effects on
  the frame of the action that caused them (`EVENT_REWARD_UPDATE_SPEC.md`).

## Baseline Measurement: Opponent vs Chmmr Before Changes

Measured with the baseline arena, which now reports satellite HP removed and
mean fight length:

```text
python -m src.training.baseline.arena --ships all --opponents Chmmr --seeds 20 --workers 8
```

20 fights per ship, baseline strategy on both sides (the Chmmr included),
100 s limit, before any Phase 2 change. W-L-T counts timeouts as T. Crew
dealt is the mean share of the Chmmr's crew removed (100% for a win).

| Ship | W-L-T | Chmmr crew dealt | Satellite HP dealt (of 30) | Mean fight (s) |
|---|---|---|---|---|
| Androsynth | 0-20-0 | 11% | 0.2 | 15 |
| Arilou | 0-20-0 | 2% | 0.0 | 5 |
| Chenjesu | 0-19-1 | 26% | 6.6 | 17 |
| Chmmr | 2-3-15 | 40% | 26.8 | 80 |
| Druuge | 19-1-0 | 99% | 7.3 | 20 |
| Earthling | 0-20-0 | 0% | 1.9 | 12 |
| Ilwrath | 0-20-0 | 0% | 0.0 | 12 |
| KohrAh | 2-14-4 | 47% | 16.6 | 33 |
| KzerZa | 0-6-14 | 25% | 2.1 | 80 |
| Mmrnmrhm | 0-3-17 | 8% | 0.1 | 90 |
| Melnorme | 0-20-0 | 17% | 3.7 | 19 |
| Mycon | 0-20-0 | 17% | 10.1 | 20 |
| Orz | 0-20-0 | 4% | 0.0 | 7 |
| Pkunk | 0-20-0 | 0% | 0.1 | 10 |
| Shofixti | 0-20-0 | 26% | 23.0 | 2 |
| Slylandro | 0-20-0 | 6% | 1.9 | 7 |
| Spathi | 0-19-1 | 1% | 0.0 | 20 |
| Supox | 0-17-3 | 2% | 0.2 | 46 |
| Syreen | 0-20-0 | 12% | 0.0 | 4 |
| Thraddash | 0-20-0 | 1% | 0.1 | 4 |
| Umgah | 0-20-0 | 0% | 0.0 | 4 |
| Utwig | 0-20-0 | 15% | 1.4 | 11 |
| Vux | 0-20-0 | 47% | 2.1 | 5 |
| Yehat | 0-20-0 | 4% | 0.7 | 16 |
| ZoqFotPik | 0-20-0 | 8% | 0.1 | 3 |

---

## Phase 1: Satellite HP Bars

### Requirements

- Draw a horizontal bar above each live Chmmr satellite.
- Length is proportional to current HP ÷ 10. The full length is a fixed width,
  scaled with the view like other sprites.
- Colour, by HP value:

  | HP | Colour |
  |---|---|
  | 10 (full) | white `#FFFFFF` |
  | 9 | green `#00FF00` |
  | 5 | yellow `#FFFF00` |
  | 1 | red `#FF0000` |

  - Between 1 and 9, interpolate linearly: red to yellow from 1 to 5, then
    yellow to green from 5 to 9.
  - A satellite that starts below full HP uses the same table. Colour and
    length follow its HP value, not a fraction of its own starting HP.
- Rendering only. No change to gameplay, collisions, training or headless runs.
- Shown wherever satellites are drawn: normal play and the training display.

### Tests

- A pure function maps HP to (length fraction, colour). Unit-test it at 10, 9,
  7 (yellow-green midpoint), 5, 3 (red-yellow midpoint) and 1 HP.
- A satellite draw call with a dummy surface does not raise.

### Supox shot end animation

- Play the Supox A1 end animation only when the shot hits something: a ship, a
  satellite, or another object it damages.
- No animation when the shot expires or is shot down, as in the original game.
- Other projectiles are unchanged.
- Test: the end animation plays on a hit, and not on expiry or on destruction by
  a satellite laser.

---

## Phase 2: Chmmr Matchup Behaviours

All changes apply only when the opponent is a Chmmr. Other matchups are
unchanged by construction.

### Shared helpers (engine)

- `chmmr_satellites(ctx)`: the live enemy satellites from the observation, with
  position, velocity and HP. Observed objects already carry `hit_points`.
  Verify how satellites are identified in the observation (name or kind field).
- `satellite_ring(ctx, reach)`: the distance band (R_min, R_max) around the
  Chmmr.
  - R_min = 512 + margin. Use 600 + margin when staying out of the Chmmr laser
    is possible.
  - R_max = 256 + reach − margin.
  - Returns None when R_max < R_min.
- `satellite_target(ctx)`: the satellite to shoot, by default the lowest-HP live
  satellite within reach. Returns the Chmmr when no satellites are left.
- `fire_when_intercept(ctx, target)`: fire when the predicted shot meets the
  target. Lasers fire when the target lies on the laser line within reach.

### Per-ship plans

Each ship is implemented and measured on its own, in this order. Keep a change
only if it raises satellite HP or Chmmr crew dealt without shortening the
average time alive. A win isn't required.

| # | Ship | Plan | Notes and verification |
|---|---|---|---|
| 1 | Thraddash | <ul><li>Lay the A2 flame trail between itself and the Chmmr while satellites are near.</li><li>Fire A1 at the nearest satellite, or at the Chmmr once the trail is absorbing zaps.</li></ul> | <ul><li>In the original game the bait trail beats the Chmmr. Here it is 0-20, dealing 0.6 crew and dying in 4.2 s.</li><li>Investigate first by watching fights: is the Chmmr in `THRADDASH_SLOW_TURNERS`, does our baseline Chmmr follow into the trail, and does the tractor pull the Thraddash in?</li><li>Flames (1 HP) are shot before A1 shots (2 HP), so the trail works as a decoy.</li><li>A1 survives one zap.</li></ul> |
| 2 | Arilou | <ul><li>Hold R ≈ 600–656.</li><li>Fire A1 when a satellite crosses the line from the Arilou to the Chmmr.</li></ul> | <ul><li>The laser aims itself at the Chmmr, and satellites block lasers, so the crossing satellite takes the hit.</li><li>The tractor doesn't affect the Arilou.</li><li>One battery is 10 shots, one satellite.</li></ul> |
| 3 | Supox | <ul><li>Hold the satellite ring for its reach.</li><li>Stream A1 at the target satellite.</li></ul> | <ul><li>Its fire rate equals one satellite's zap rate, so success is uncertain. Its current results leave nothing to lose.</li><li>Phase 1 HP bars make hits visible.</li></ul> |
| 4 | Pkunk | <ul><li>Fly tangentially at R ≈ 600–688.</li><li>Fire continuously, side gun toward the target, as a satellite passes.</li></ul> | <ul><li>Fires up to 3× one satellite's zap rate.</li><li>The tractor pulls it hardest (mass 1).</li><li>Builds on the existing side-gun strafing.</li></ul> |
| 5 | Vux | <ul><li>No movement goal of its own against the tractor.</li><li>While within 256 of any satellite (satellites shoot the ship first), launch limpets at the Chmmr.</li><li>Laser the target satellite or the Chmmr, whichever is lined up.</li></ul> | Limpets (1 HP) are shot down only when the satellites have no ship to shoot. |
| 6 | Orz | <ul><li>Launch marines only when ≤ 1 satellite is left.</li><li>Otherwise shoot satellites with the turret.</li></ul> | Today marines launch below 2 regardless of satellites. |
| 7 | Kzer-Za | Fighters only when 0 satellites are left. | **Already implemented** (`strategies.py`, `fighter_counter`). Confirm by measurement only. |
| 8 | Androsynth | <ul><li>Blazer form at the Chmmr.</li><li>Save battery until close, then go all out.</li></ul> | Satellites don't collide with ships. Verify whether the blazer can hit satellites at all; if not, target only the Chmmr. |
| 9 | Ilwrath | <ul><li>Stay cloaked until in A1 range of the Chmmr.</li><li>Then unload A1 until one ship dies.</li></ul> | Satellites ignore cloaked ships, and the tractor needs a visible target. |
| 10 | Utwig | <ul><li>Hold the shield while in satellite range to stay charged from their lasers.</li><li>Fire A1 at the Chmmr whenever the battery allows.</li></ul> | It can't shield and fire at once. Measure crew dealt, not just wins. |
| 11 | Earthling | <ul><li>Build lateral speed so it orbits the Chmmr instead of being pulled in.</li><li>Use the A2 lasers once close.</li></ul> | <ul><li>Verify whether the A2 point-defence picks satellites. It picks enemy objects by distance.</li><li>A1 can't get through the satellites.</li></ul> |
| 12 | Umgah | Point at the Chmmr and keep the cone on. | No real hope. Verify whether the cone damages satellites. |
| 13 | Slylandro, ZoqFotPik, Spathi, Syreen, Yehat | Try the shared helpers: ring plus satellite targeting. | Keep per ship only if it helps. |

### Follow-ups added during implementation

#### 2.8a Blazer intercept steering (all opponents but two)

- The Androsynth blazer steers toward the intercept point, as it does against
  the Chmmr: the bearing on which the blazer, at its own speed, meets the
  opponent moving as it is. It falls back to the opponent's bearing when no
  intercept exists.
- Exceptions: against the Supox and the Mmrnmrhm it keeps steering straight at
  the opponent.
- Evidence (20 seeds per opponent, intercept everywhere): 202-174 against
  177-186. Gains include the Thraddash 13-5 → 16-0, Melnorme 3-17 → 10-10 and
  Vux 6-0 → 10-0. Losses: Supox 12-8 → 8-12, Mmrnmrhm 7-12 → 4-13.
- Measure again after the change, against all opponents.
- Tests: the blazer turns toward the intercept point against a crossing
  opponent; against the Supox and the Mmrnmrhm it turns toward the bearing.

#### 2.8b Blazer damages Chmmr satellites

- Today satellites collide only with enemy projectiles, so a blazer passes
  through them unharmed. Change the game rule so a blazer touching an enemy
  satellite deals it the blazer's ramming damage (3), as a blazer already
  rams asteroids and projectiles (`_apply_ramming_damage_to_*` in
  `collision_responses.py`).
- The satellite stays on its orbit. The blazer bounces off it, as in the
  original game, the way it bounces off a planet (its velocity reflected about
  the contact normal), and takes no damage from it.
- One hit per contact: a satellite already touching the blazer isn't hit again
  until the contact ends. A blazer at speed 60 overlaps a satellite for about a
  frame, so this guards slow or glancing passes.
- Only the blazer form rams satellites; the base Androsynth hull, like other
  ships, still passes through them.
- Satellite HP bars and the arena satellite-HP metric show the hits.
- Then evaluate the Androsynth strategy against the Chmmr again. Candidates:
  - the current single ram on the Chmmr (38% crew dealt, 80 seeds);
  - ram satellites on the way in (each takes 4 hits), then the Chmmr;
  - hit-and-run passes through the ring aimed at satellites;
  - satellites approached prograde (moving with the satellite's orbit, a slow
    closing speed) and retrograde (against it, head-on).

  Keep whichever deals the most harm, counting satellite HP as the harm scheme
  does.
- Tests:
  - a blazer touching an enemy satellite takes 3 HP from it, once per contact;
  - the base form and other ships still pass through satellites;
  - the satellite's orbit is unchanged by the contact, and the blazer bounces
    away from it;
  - a satellite brought to 0 HP is destroyed.

#### 2.10a Chmmr tractor timing

- Today the Chmmr baseline holds the tractor (A2) only while the opponent is
  beyond 1000 (`LONG_RANGE`), its battery is at least 30%, the opponent isn't
  tractor-immune (Arilou, Slylandro) and isn't cloaked. Each frame held costs
  1 battery and pulls the opponent 12 ÷ its mass toward the Chmmr, at any
  range.
- Change it to use the tractor where it helps the laser (range 600):
  - **Bring a mid-range opponent closer:** the opponent is beyond laser reach
    but near enough that the pull brings it into reach soon (a mid-range band
    starting at 600; its outer edge to be measured, for example 900, 1200 or
    1500).
  - **Keep a close opponent from escaping:** the opponent is within laser reach
    (or just beyond it) and moving away (closing speed below 0).
- Keep the battery, immunity and cloak gates. Measure whether the battery
  threshold should differ between the two uses, and whether the long-range use
  beyond the mid-range band is still worth its battery.
- The tractor pulls inertial ships only, so skip it against inertialess ones,
  where it costs battery for nothing.
- Measure the Chmmr against every opponent, before and after, 20–40 seeds each.
  Then re-measure the Phase 2 ships whose plans depend on the tractor's reach:
  the Orz holds 900–1250 and the Utwig 800–1050, chosen partly to stay under
  today's 1000; re-tune their bands if the new tractor changes their results.
- Tests:
  - pulls an opponent in the mid-range band, and not one beyond it;
  - pulls a close opponent moving away, and not one closing in;
  - no pull against a tractor-immune, cloaked or inertialess opponent, or below
    the battery threshold.

### Measurement

- Measure each ship before and after, 20–40 fights against the Chmmr. Metrics:
  - satellite HP dealt;
  - Chmmr crew dealt;
  - seconds alive;
  - W-L.
- Make the measurement script reusable: extend the baseline arena tooling to
  report satellite HP dealt. It is not part of the unit test suite.

### Tests

- Ring computation: values, and None when R_max < R_min.
- Satellite target choice: lowest HP, then nearest, then the Chmmr when none are
  left.
- Arilou line test: a satellite on the line fires; one off the line doesn't.
- Per-ship gates:
  - Orz marines only at ≤ 1 satellite.
  - Vux limpets only while a satellite is in range of the Vux.
  - Ilwrath stays cloaked until in range.
- Behaviour against non-Chmmr opponents is unchanged.

---

## Phase 3: Planet Lookahead By Agility

Replace the fixed `PLANET_AVOID_HORIZON_SECONDS = 4.0` with a per-ship horizon:

```text
horizon = turn_180_frames + accel_frames + margin_frames
```

- `turn_180_frames`: from `ShipFlight` turning rules.
- `accel_frames`: frames from 0 to `max_thrust`, from `thrust_increment` and
  `thrust_wait`. It is 0 for non-inertial ships, and for the Slylandro's
  constant speed.
- Clamp to [1 s, 4 s] until measured.

Acceptance:

- Planet deaths do not rise.
- Frames claimed by the planet rule fall for agile ships (Arilou, Pkunk, Supox).

Tests: horizon values for an agile ship, a slow ship and a non-inertial ship.

---

## Phase 3A: Pre-emptible Held Weapons

### Problem

Three ships fire weapon 1 by holding it, and can't fire again until they let
go:

| Ship | While A1 is held | On release | Today's baseline |
|---|---|---|---|
| Chenjesu | The crystal flies on. | It bursts into shards. A new crystal needs a release, then a fresh press. | <ul><li>Holds while the crystal still closes on the enemy (`_hold_while_closing_on_enemy`).</li><li>The rule 1 shoot-down includes the Chenjesu, but it presses A1. While a crystal is held, pressing only keeps holding it, so the planned new shot never fires.</li></ul> |
| Kohr-Ah | The saw flies on. | It parks where it is. A new saw needs a release, then a fresh press. | <ul><li>Holds while the saw still closes on the enemy.</li><li>Excluded from the shoot-down (`SHOOT_DOWN_EXCLUDED_SHIPS`).</li></ul> |
| Melnorme | The shot charges. | It fires at the current charge, in the nose direction. | <ul><li>Held from the start, released only when in range and pointing at the enemy.</li><li>Excluded from the shoot-down.</li></ul> |

A shot in flight or charging is kept even when something matters more:
- shooting down an incoming projectile or enemy object, such as a fighter,
  marine, mine or Dogi;
- a better shot at the enemy.

This applies especially when the held shot has little chance of hitting.

### Requirements

1. **Value of the held shot.** Estimate its expected damage to the enemy:
   - Chenjesu and Kohr-Ah: from the predicted closest approach of the shot in
     flight to the enemy, whether it is still closing, and the time to reach it.
   - Melnorme: from the current charge level and whether the enemy is in range
     along the nose.

   A shot that is receding or predicted to miss is worth about 0.
2. **Pre-emption.**
   - A competing goal may take weapon 1 when its value exceeds the held shot's
     value. Competing goals include a shoot-down from the dodge rule and a new
     shot at the enemy or another enemy object.
   - When the held shot is worth about 0, any useful goal wins.
3. **Release mechanics are modelled.**
   - A new shot needs one frame with A1 released, then a fresh press once A1 is
     ready.
   - Plans that fire a new shot (shoot-down options, the dodge rule's
     simulation) start from that release frame. They must not assume holding A1
     fires.
   - The release has its own effect, and options include it:
     - the Chenjesu crystal bursts, and its shards may themselves stop a threat;
     - the Kohr-Ah saw parks as an obstacle;
     - the Melnorme shot fires at its current charge.
4. **Melnorme release as the shot.** An option releases the charging shot
   itself at the threat or enemy once the nose points at it, without waiting
   for full charge or for the enemy to be in range.
5. **Shoot-down exclusions.**
   - Remove the Kohr-Ah and Melnorme from `SHOOT_DOWN_EXCLUDED_SHIPS` once 3 and
     4 model them correctly.
   - Keep a ship excluded if measurement shows the release costs more than it
     saves.
6. **No change for other ships.**

### Measurement

For each of the three ships, before and after:
- crew lost per minute to shots that could be shot down;
- crew dealt per minute;
- W-L.

Use opponents with projectiles or launched objects: Druuge, Kzer-Za, Orz,
Spathi, Chenjesu, Mycon and Earthling. Keep the change per ship only if crew
lost falls without crew dealt falling more.

### Tests

- Chenjesu with a crystal held and an imminent threat on a shoot-down line:
  - it releases, then presses again once ready, firing a new crystal at the
    threat;
  - with no competing goal and the crystal closing, it keeps holding.
- A receding crystal or saw is released for a new shot at the enemy.
- Kohr-Ah: the same release-then-fire sequence; the parked saw remains.
- Melnorme: it releases a partial charge at an imminent threat it points at,
  and keeps charging when nothing competes.
- The dodge rule's simulation includes the release frame for held weapons.
- Strategies for other ships are unchanged.

---

## Phase 4: Discount Half-Life Setting

### Requirements

- The Regimen setting "Gamma" becomes "Discount half-life (s)", h, with a range
  of 1–30 s and a default of 5 s.
- Derived values (FPS = 24):

  ```text
  gamma = 0.5 ** (1 / (h * FPS))
  cutoff_frames = ceil(ln(0.01) / ln(gamma)) ≈ 6.644 * h * FPS
  ```

  | Half-life | Gamma | Cutoff |
  |---|---|---|
  | 1 s | 0.97153 | 160 frames (6.6 s) |
  | 2.9 s (today's 0.99) | 0.99 | 459 frames (19 s) |
  | 5 s | 0.99424 | 798 frames (33 s) |
  | 10 s | 0.99712 | 1,595 frames (66 s) |
  | 15 s | 0.99808 | 2,392 frames (100 s) |
  | 30 s | 0.99904 | 4,784 frames (199 s) |

- At the default 100 s match limit, the cutoff exceeds the match time above
  about 15 s. Most returns then end at the match end instead of the cutoff.
  Samples also wait longer in staging before reaching replay.
- Cutoff warning:
  - Show a warning next to the setting whenever `cutoff_frames` exceeds the
    match frame length currently set on the Regimen tab.
  - Recheck it when either setting changes.
  - Example text: "Credit reaches back further than a match lasts (cutoff 33 s,
    match 30 s); returns will end at the match end."
  - The warning does not block saving or starting.
- Saved configs:
  - Store the half-life.
  - When loading an old config with only gamma, derive h = ln 0.5 ÷ (ln gamma ×
    FPS) and round it to 0.1 s for display.
- A coordinated run requires equal half-lives, as it now requires equal gamma.

### Tests

- Half-life ↔ gamma round trip.
- Cutoff frames at 1, 5 and 30 s.
- The warning appears exactly when the cutoff exceeds the match frame length,
  including after either value changes.
- Old configs load with the derived half-life.
- The coordinated mismatch check uses the half-life.

---

## Phase 4A: Reward Magnitude Steps

### Requirements

- Every reward and penalty weight slider, in both the Classic and Harm schemes,
  takes one of 23 values:

  ```text
  -1.024, -0.512, -0.256, -0.128, -0.064, -0.032, -0.016, -0.008, -0.004, -0.002, -0.001,
  0.000,
  0.001, 0.002, 0.004, 0.008, 0.016, 0.032, 0.064, 0.128, 0.256, 0.512, 1.024
  ```

  That is ±0.001 × 2^k for k = 0–10, and 0. Today's values are ±0.01 × 2^k
  for k = 0–12 (up to ±40.96).
- Labels show three decimals ("0.001", "1.024").
- Classic defaults, divided by 40 so they land on the new steps with their
  ratios kept:

  | Component | Today | New |
  |---|---|---|
  | Kill enemy | 40.96 | 1.024 |
  | Reduce enemy crew | 5.12 | 0.128 |
  | Get debuffed | -1.28 | -0.032 |
  | Lose crew | -2.56 | -0.064 |
  | Die | -20.48 | -0.512 |

- Harm defaults: D_enemy and K_enemy 1.024 (1.28 is no longer a step);
  D_self and K_self 0.
- Old configs: divide each stored classic weight by 40 and snap it to the
  nearest step on the log scale, keeping its sign. 0.01 and 0.02 become 0.001
  (the smallest step); 0 stays 0. Every other old step maps exactly, so the
  ratios between components are kept.
- The reward scale falls 40-fold, and so do the value targets. A model trained
  on the old scale predicts values 40 times too large, so resuming one needs
  time to relearn magnitudes. Record a reward scale version in the model
  metadata and warn when resuming a model saved under the old scale.

### Tests

- The value list, its order and labels.
- Classic and Harm defaults.
- Old config migration: 40.96 → 1.024, 5.12 → 0.128, -1.28 → -0.032,
  0.01 → 0.001, 0 → 0.
- The warning on resuming a model saved under the old scale.

---

## Phase 4B: Decay Half-Lives In Batches

### Requirements

- The "Epsilon decay" and "Baseline use decay" Regimen sliders show a half-life
  in batches instead of a per-batch multiplier: the number of batches over
  which the value halves.
- Values: 10, 30, 100, 300, 1,000, 3,000, 10,000, 30,000 and 100,000 batches,
  plus "Off" (no decay), which keeps today's 1.000 setting available.
- Per-batch factor: d = 0.5 ^ (1 / n).

  | Half-life (batches) | Factor per batch |
  |---|---|
  | 10 | 0.93303 |
  | 30 | 0.97716 |
  | 100 | 0.99309 |
  | 300 | 0.99769 |
  | 1,000 | 0.999307 |
  | 3,000 | 0.999769 |
  | 10,000 | 0.9999307 |
  | 30,000 | 0.9999769 |
  | 100,000 | 0.9999931 |
  | Off | 1 |

- Defaults: epsilon 300 batches (today 0.998, a half-life of 346); baseline
  use 100 batches (today 0.995, 138).
- Today's range, 0.950–1.000, covers half-lives from 13.5 batches to Off, with
  most steps below 700 batches. The new list reaches much slower decay.
- Saved configs:
  - Store the half-life ("off" for no decay).
  - When loading an old config with only a factor d, derive n = ln 0.5 ÷ ln d,
    snap it to the nearest value on the log scale, and treat 1.000 as Off.
- Training uses the derived factor, so the decay code is unchanged. The
  current-batch status row shows the half-life.
- The "go to epsilon" no-decay warning uses Off.
- A coordinated run requires equal half-lives, as it now requires equal decay
  factors.
- Tooltips: "Epsilon decay half-life: batches over which epsilon halves.", and
  the same for baseline use.

### Tests

- Half-life ↔ factor round trip for every value.
- Old configs: 0.998 → 300, 0.995 → 100, 0.95 → 10, 1.000 → Off.
- The no-decay warning fires for Off.
- The coordinated mismatch check uses the half-lives.

---

## Phase 5: Harm Reward Scheme

### Concept

A new reward scheme, `harm`, rewards only lasting harm to the opponent, and
optionally penalises lasting harm to the trainee. Every other component is off
in this scheme: point, range, speed, use, battery, debuff and object kills. The
baseline strategy supplies the behavioural guidance those components used to.

The existing scheme stays available as `classic`. The scheme is separate from
the existing placement mode (`legacy`, `shadow`, `causal`). Harm rewards use
causal placement as today:

- Trainee-caused harm to the opponent goes on the causing action's frame.
- Penalties stay on the frame the harm happens.
- Self-inflicted own losses, such as the Druuge crew burn, go on the trainee
  action frame that caused them.
- Opponent crew gains produce no reward or penalty.

### Settings (Rewards tab, harm scheme)

Penalty weights are set separately from reward weights, as they are today.

| Weight | Meaning | Default |
|---|---|---|
| D_enemy | Damage to the opponent | 1.024 (Phase 4A) |
| K_enemy | Kill of the opponent | 1.024 (Phase 4A) |
| D_self | Damage taken, as a penalty | 0 |
| K_self | Own death, as a penalty | 0 |

### Formulas

Symbols:

- V: the damaged ship's cost.
- s = sqrt(V / Vmax), with Vmax = 30, the highest cost in the ship catalog.
- HP: the ship's normal starting crew.
- satHP: normal total satellite HP, 30.
- has_sats: 1 for the Chmmr, otherwise 0.
- p: 0.5 for the Pkunk, otherwise 1.

The same formulas give the reward and the penalty:

- The reward uses the opponent's V, HP, satellites and p, with D_enemy and
  K_enemy.
- The penalty uses the trainee's own values, with D_self and K_self.

```text
Damage = D * s * [0.5 * p * hp_damage/HP + L + 0.5 * sat_damage/satHP] / (1 + 0.5 * has_sats)
Kill   = K * s * (ship_kills + sat_kills/3) / (1 + has_sats)
```

The reward on a frame is the change in each value since the previous frame.

Damage definitions:

- Crew means effective crew: crew on board plus launched crew that is away
  (Kzer-Za fighters, Orz marines).
  - Launching or recovering a unit doesn't change effective crew.
  - A unit lost for good is a 1-crew loss at that moment.
- Pkunk: each life is tracked separately, from that life's starting crew. A
  rebirth's crew refill is not a gain. Damage counts afresh for each life, halved
  by p.
- `hp_damage` is measured differently for the opponent and the trainee.

#### Opponent `hp_damage`: new lows caused by the trainee

The same rule applies to every opponent. For ships that can't gain crew it is
the same as plain damage.

```text
low = opponent's starting crew            (per life for the Pkunk)

on each opponent crew loss from c_before to c_after:
    new_low_amount = max(0, low - c_after)
    if the loss was caused by the trainee:
        hp_damage += new_low_amount        (placed on the causing action's frame)
    low = min(low, c_after)

on each opponent crew gain:
    nothing; low is unchanged
```

- Losses the trainee didn't cause still lower `low`, so the trainee is never
  paid for them later. Examples: planet hits, Druuge crew burn, own fighters
  lost.
- Opponent gains (Syreen pods, Mycon regeneration) neither penalise the trainee
  nor reset `low`. Damage that only undoes a gain earns nothing.
- Total credit can't exceed the opponent's starting crew, so this can't be
  farmed.
- A kill always pays the full starting crew less any new lows the trainee didn't
  cause.
- An opponent Syreen above its starting crew must be brought back below it
  before damage earns anything. This is intended: crew is normalised to the
  normal starting crew.

#### Trainee `hp_damage`: follows the crew level

```text
hp_damage = trainee's starting crew - current effective crew   (per life for the Pkunk)
```

- Every loss raises the penalty and every gain lowers it, from any cause and at
  any level, including above the starting crew.
- The total always equals the net change in crew, so this can't be farmed: crew
  can only be regained after a loss that was penalised by the same amount.
- Crew collected above the starting crew stays at risk, so a Syreen has reason
  to protect it.
- Mycon regeneration lowers the penalty directly, giving it a reason to
  regenerate.
- Gains pay only through D_self, so they earn nothing while D_self = 0.
- Placement:
  - Losses caused by the opponent or the environment go on the frame they
    happen.
  - Self-inflicted losses go on the trainee action frame that caused them.
  - Gains go on the frame they happen.
- For the trainee, the damage term depends only on the crew level reached, not
  the path.

- `L`, limpets:
  - The n-th limpet on a ship is worth 0.25 × 0.5^(n−1), so L stays below 0.5.
  - Limpets already on the ship at fight start set n, so a later Vux continues
    the series.
- `sat_damage`: satellite HP removed.

Kill definitions:

- `ship_kills`: 1 per qualifying death, 0.5 for each Pkunk death.
- `sat_kills`: satellites destroyed.

### Checks of the formulas

- **Kill:** a full kill gives exactly K × s for every ship.
  - A Chmmr kill is (1 + 3/3) ÷ 2 = 1. Each satellite is worth 1/6 and the ship
    itself 1/2.
  - The Pkunk's 0.5 per death averages about 0.93 kills per Pkunk.
- **Damage:**
  - Killing all crew gives 0.5 × D × s, or 0.67 × D × s for a Chmmr with all
    satellites destroyed.
  - With limpets at their 0.5 limit, the total is exactly D for every ship. For
    a Chmmr it is (0.5 + 0.5 + 0.5) ÷ 1.5 = 1. Any finite number of limpets
    leaves it just short.
  - This is intended. Limpets, the first especially, cripple a ship, while a
    ship down to 1 crew can still fight at full strength.
- **Vux vs Chmmr example** (D = K = 1.28, s = 1):
  - 3 satellites destroyed;
  - 3 limpets (L = 0.4375);
  - all crew and the kill.

  Damage is (0.5 + 0.4375 + 0.5) ÷ 1.5 = 0.958 D = 1.23, plus the kill 1.28,
  total 2.51. With limpets at their limit it would be 1.0 D = 1.28, total 2.56.
- **Opponent Mycon example** (starting crew 20):

  | Step | Crew | Low-water mark | Trainee credit |
  |---|---|---|---|
  | Trainee hits for 8 | 12 | 12 | +8 |
  | Mycon regenerates 6 | 18 | 12 | 0 |
  | Trainee hits for 10 | 8 | 8 | +4 (only the part below 12) |
  | Planet hit for 3 | 5 | 5 | 0 (not trainee-caused) |
  | Trainee kills it | 0 | 0 | +5 |
  | **Total** | | | **17** |

  - Following the opponent's crew level would also total 17, but would put a
    −6 on whatever the trainee was doing when the Mycon regenerated. The
    new-low rule avoids that.
  - If the Mycon regenerated to full and survived, the trainee would keep its
    +8. The Mycon had to spend battery to regenerate, and the credit is
    bounded.
- **Trainee Syreen example** (start 12, max 42):

  | Step | Crew | Penalty change |
  |---|---|---|
  | Collects 8 | 20 | −8 (credit) |
  | Loses 5 | 15 | +5 |
  | Regains 5 | 20 | −5 (credit) |
  | Collects 2 | 22 | −2 (credit) |
  | Loses 9 | 13 | +9 |
  | **Net** | | **−1, a credit of 1 crew, equal to the net change** |

  - The most a trainee Syreen can be credited is 30 crew above its start:
    0.5 × 30/12 × D_self × s = 1.25 × D_self × s.
  - Crew the song pulls off the opponent is separately opponent damage, paid
    through D_enemy under the new-low rule.

### Attribution

Rewards (opponent harm):

- Only harm the trainee caused counts: the ledger's causal source resolves to
  the trainee. That includes:
  - satellite lasers fired by a trainee Chmmr;
  - marines and fighters launched by the trainee.
- Opponent crew losses not caused by the trainee earn nothing. Examples:
  - planet and asteroid hits;
  - Druuge crew burn;
  - Shofixti self-destruct;
  - fighters or marines lost to the environment.
- Opponent losses the trainee didn't cause still lower the opponent's
  low-water mark. Opponent gains change nothing (see "Opponent `hp_damage`").
- No kill credit for opponent deaths the trainee didn't cause. This matches
  today's factors, which are 0 for planet, Druuge A2 and Shofixti A2 deaths
  (`_enemy_death_reward_credit`).

Penalties (own harm):

- Every own loss counts, whatever the cause, and every own gain counts back
  (see "Trainee `hp_damage`"). Self-inflicted losses count too:
  - Druuge crew burn;
  - Shofixti self-destruct, as both crew damage and a death;
  - launched crew lost for good to enemy fire, asteroids or the planet.
- Notes on launched crew:
  - Kzer-Za fighters route around the planet but can die on asteroids.
  - Orz marines avoid the planet but can hit it under adversarial conditions.
- The harm scheme penalises launched crew only when it is lost for good, not at
  launch. Unlike the classic causal escrow, nothing is charged at launch and
  refunded on recovery.

### UI and storage

- A two-position slider at the top of the Rewards pane switches between Classic
  and Harm. The pane has room for it.
- The pane shows the four harm weights in Harm and the existing sliders in
  Classic.
- Harm weights are stored under new keys. The magnitudes differ from the classic
  per-crew components, so reusing those keys would misread old weights.
- Old configs load as `classic`.
- The console's reward table lists the active scheme's components, each as the
  mean weighted return per sample, as for the classic components. Raw harm
  totals per fight (damage, kill and the two penalties) are deferred.

### Tests

- Damage and kill values for:
  - a 6-crew ship and a 30-cost ship;
  - the Chmmr with satellites;
  - Pkunk damage and deaths, halved, including rebirth;
  - the opponent new-low rule, using the Mycon example: gains ignored, damage
    that only undoes a gain unpaid, losses the trainee didn't cause lowering
    the mark without credit, and a new low partly crossed by one hit;
  - the opponent new-low rule against a ship that can't gain crew, equal to
    plain damage;
  - trainee crew-level following, using the Syreen example: the penalty total
    equals the net crew change, and gains above the start are credited;
  - Pkunk per-life tracking for both the opponent and the trainee;
  - a limpet series continuing from existing limpets.
- Per-frame change rewards sum to the formula totals over a fight.
- Opponent losses the trainee didn't cause earn nothing. Own self-inflicted
  losses are penalised.
- Launched crew:
  - recovered crew costs and earns nothing;
  - trainee-destroyed enemy fighters count;
  - own fighters lost to asteroids are penalised.
- No kill credit for opponent planet, Druuge A2 or Shofixti A2 deaths.
- Causal placement: damage goes on the origin frame, penalties on the effect
  frame.
- Old configs load as classic, and the harm keys round-trip.
- The scheme slider switches the visible weights and persists.

### Expected training impact

- Rewards are sparser and noisier.
- Pair the scheme with a longer half-life (Phase 4) and slower epsilon and
  baseline-use decay.
- Future option, not in scope: a small, decaying imitation term toward baseline
  actions if learning stalls without shaping.

---

## Phase 6: Shared Prioritised Behaviours (last, one ship at a time)

### Target structure

Control priority is decided per control (turn, thrust, A1, A2), as today. Shared
behaviours, each with per-ship parameters and gates:

1. Avoid damage:
   - dodge with a brief turn, then resume the movement goal;
   - shoot down threats;
   - run from threats, as a new option.
2. Avoid the planet, with the Phase 3 horizon.
3. Leave opponent weapon range while recharging, when battery is below the cost
   of the next useful shot.
4. Hold a distance band. This one behaviour covers both "optimal weapon range"
   and "as far as possible", with a preferred side for wrap-around.
5. Keep moving.
6. Fire: from predicted outcomes up close, by rule at long range.
7. Mode switch (Androsynth, Mmrnmrhm), plus a per-ship slot for unique specials:
   Umgah cone, Syreen song, Orz marines, Kzer-Za fighters, shields.

### Schema

The table is `SHIP_BEHAVIOURS` in `src/training/baseline/behaviours.py`. A
moved ship's strategy is `strategies.shared`, which runs the ship's entry. The
controller runs rules 0 and 1 first and keep moving last, as for every ship.
In between, `shared` runs in this order, each part claiming only the controls it
sets:

1. specials;
2. mode switch;
3. recharge, or else the band;
4. weapon 1, then weapon 2.

The schema was checked against every current strategy, so each behaviour has a
place in it:

| Part | Settings | Covers |
|---|---|---|
| Opponents | `opponents`: the fields that differ against a named opponent. | Per-opponent branches such as the `_<ship>_vs_chmmr` rules, and per-opponent sets such as `ORZ_KITE_OPPONENTS`. |
| Avoid damage (rule 1) | `evade`, `evade_headings`, `shoot_down`, `run_from_threats`. State-dependent gates stay functions (`strategies.EVADE_GATES`). | Dodging, shoot-downs, running. |
| Planet (rule 0) | `planet_horizon_frames`; None uses the Phase 3 horizon. | Rule 0. |
| Recharge | `recharge_battery`, `recharge_resume` (hysteresis), `recharge_run_offset`. | <ul><li>The flee-at-half-battery rules (Spathi, Pkunk).</li><li>Recharging from empty to full (Shofixti, Umgah).</li><li>Fleeing at 30% (Supox, ZoqFotPik) and at 10% (Androsynth).</li></ul> |
| Band | `Band`: `near`, `far` (None means as far as possible), `outside_reach_margin`, `min_width`, `approach_speed`, `inside` (free, face or orbit), `run_offset`. | <ul><li>`approach_range` and `keep_standoff` users.</li><li>`avoid_opponent`: as far as possible.</li><li>`approach_opponent`: a near edge of 0.</li><li>The Supox's band outside the opponent's reach.</li><li>Orbiting the Chmmr (Earthling, Slylandro).</li></ul> |
| Weapons | `WeaponUse` per weapon, one of: <ul><li>`aimed`: range fraction, battery reserve, whether it may turn the nose, satellite targets, aim tolerance;</li><li>`launch`: a self-aiming shot fired whenever it would outrun the ship;</li><li>`defend`: imminent hit within a time, or crew lost;</li><li>`special`.</li></ul> | <ul><li>Aimed shots and their battery reserves (ZoqFotPik tongue, Syreen song, Thraddash burn, Vux laser, Arilou teleport).</li><li>Homing launches (Spathi missile).</li><li>Satellite shots.</li><li>Shields (Utwig, Yehat), the Arilou teleport, the Thraddash afterburner dodge.</li></ul> |
| Mode switch | `form_switch` names a rule in `strategies.FORM_SWITCHES`. | Androsynth blazer, Mmrnmrhm forms. |
| Specials | `specials` names rules in `strategies.SPECIAL_RULES`, run in order. | <ul><li>Escorts: Kzer-Za fighters, Chenjesu Dogi, Orz marines.</li><li>Held weapons: Chenjesu, Kohr-Ah, Melnorme.</li><li>Resources: Mycon regeneration, Druuge crew for battery, Pkunk insults, Slylandro asteroids, Syreen crew pods.</li><li>One-off abilities: Ilwrath cloak, Chmmr tractor, Shofixti glory device, Kohr-Ah gas, Earthling point defence, Druuge recoil escape.</li><li>Unique movement: Umgah cone and hop runs, Supox slides, Pkunk passes, Vux blind-side approach and keep-away, Thraddash tail burn, gravity whips.</li></ul> |

Not yet in the schema:

- a preferred wrap-around side for "as far as possible";
- firing from predicted outcomes up close (behaviour 6).

Each will be added when the first ship that needs it moves.

### Evaluation

Baselines are training opponents and the trainee's teacher in baseline phases,
so they should engage. Judging by wins and losses rewards stalling: a timeout
counts as a non-loss. The first Spathi move showed this. Timeouts rose from 116
to 190 of 500 fights while crew dealt per minute stayed flat, and against the
Utwig the Spathi sat beyond its bolts for 0-0-20. Ships are therefore compared
by harm, with the same formulas as the Phase 5 reward scheme.

**Fights.**

- Each fight runs a fixed 1,200 frames (50 s), the training default.
- Either ship respawns when destroyed, as in training rounds, and every new
  ship is tracked from scratch.
- A fight ends only at the time limit, so a quick kill is followed by more
  fighting rather than ending the fight.

**Starting conditions.** Each seed is flown in a 2 × 2 matrix: each side starts
strong or weak.

- **Strong:** its normal starting crew and, for the Chmmr, all 3 satellites.
- **Weak:** a quarter of its normal starting crew, rounded up, and, for the
  Chmmr, 1 satellite.
- A side respawns in the same condition throughout the fight.
- Battery and everything else start as normal.
- Positions and headings come from the seed. The conditions come from the cell,
  not from the battle's random stream, so every variant meets the same
  conditions in the same order.

**Harm.**

- The arena runs the harm tracker (`src/training/harm.py`) on the battle's
  events, as training does.
- **Harm dealt:** the evaluated ship's damage and kill credit against the
  opponent, by the new-low rule.
- **Harm taken:** its own damage taken and deaths.
- Both are in harm units, so a ship's own cost already weighs its losses.

**Ranking.** Causing harm to the opponent comes first. A weighted score can't
guarantee that, because less harm dealt can win if it also costs less. With
λ = 0.25:

| Setting | Harm dealt | Harm taken | Dealt − λ × taken |
|---|---|---|---|
| A | 1.0 | 0 | 1.0 |
| B | 1.5 | 2.5 | 0.875 |

A scores higher despite dealing less harm. So settings are ranked by harm dealt
first, and a λ score only breaks near-ties:

- **Harm dealt:** the mean per fight.
- **λ score per fight:** 0 when harm dealt is 0; otherwise harm dealt − λ × harm
  taken. Its mean per fight is the setting's λ score.
- **Clearly ahead:** setting X is clearly ahead of Y on a measure when the
  paired 95% interval of the mean difference excludes zero and the gap is more
  than 5% of the larger mean.
- **Comparing two settings:**
  - whichever is clearly ahead on harm dealt ranks higher;
  - when neither is, they're tied on harm dealt, and whichever is clearly ahead
    on the λ score ranks higher;
  - otherwise they're equal.
- A setting that deals no harm ranks below any that does. A fight that deals no
  harm adds nothing to either measure.

**λ.** Each ship has its own explicit λ (`harm_taken_weight` in its table entry,
with per-opponent overrides). It is set when the ship moves, and its rationale
is recorded. λ decides only how much survival matters among settings that deal
about the same harm; it never buys survival at the cost of harm dealt.

- **A ship that can escape** can stall by running, so it gets a low λ and
  barely weighs harm taken. Example: the Spathi.
- **A ship that can't escape** can't stall, so a high λ only makes it pick the
  safer of two equally harmful plans. Example: the Chenjesu, which is costly
  and outruns nothing.
- Where escape depends on the opponent, the per-opponent overrides set it.

**Harm taken still matters.** With respawns, dying costs little in harm dealt,
since a fresh ship keeps fighting, so ranking by harm dealt could favour
suicide dives. For now, a dive that really does deal more harm is accepted, and
harm taken is reported so one would show up. If dives appear, a per-ship cap
can be added, for example accepting more harm dealt only when the extra harm
taken is at most 3 times the extra harm dealt.

**Reports.**

- Per setting, per opponent and per matrix cell:
  - total harm dealt;
  - total harm taken;
  - total and mean λ score;
  - mean harm dealt per fight.
- Diagnostics: kills and deaths, the share of fights with no harm dealt, and
  still frames.
- Wins, losses and timeouts no longer apply, as fights always run to the time
  limit.

**Comparisons.**

- Variants fly the same seeds and cells, so each comparison is paired, fight
  by fight. The arena reports the mean differences in harm dealt and in the λ
  score, with 95% intervals, and the ranking result.
- When screening settings, settings are dropped as they fall behind:
  - each starts on 10 seeds;
  - settings whose fights all came out identical so far fly the same way,
    so they are merged into the one listed first, and the merge is logged;
  - the settings are ordered. The lead group comes first: those within 5% of
    the highest harm dealt, ordered by λ score. The rest follow by harm dealt;
  - settings clearly behind the leader (the comparison above) are dropped;
  - at most the better half is kept;
  - once one setting is left the screen stops, without flying it further.
    Otherwise the seeds for the rest are doubled, and this repeats up to 100
    seeds;
  - the winner is confirmed on 100 seeds not used for tuning.
- "About as good or better" means the moved ship ranks equal to or higher than
  its earlier rules.
- "Marginally worse" means tied on harm dealt and behind on the λ score by less
  than 10%. A ship clearly behind on harm dealt is not accepted.

**Speed.**

- **Where the time goes** (profiled on first-kill fights):
  - about 35% encoding the observations;
  - about 34% baseline decisions, mostly rule 1's flight simulations;
  - about 29% the battle simulation.
- **Savings made:**
  - the arena builds each frame's observation context once for both ships;
  - it skips the float32 validation of observations, which never reach a
    network here;
  - visual-only effects are off.
  - One 1,200-frame fight took 1.48 s instead of 1.73 s (14% less). The
    observations are identical (`test_shared_observation_context_changes_nothing`).
- **Rule 1:** straight threats already reuse one predicted path across the
  moves rule 1 compares (`ThreatTrack.fixed_path`). Homing threats can't,
  because their path depends on the move.
- **Profiled 2026-11-21** (the Spathi against every opponent, one
  1,200-frame fight each, timed by part):
  - Nothing builds up over a fight: objects in the world stay at 10–33, and
    the ledger and harm tracker cost almost nothing.
  - Later frames cost no more than early ones. Equal-work batches of 600-
    and 1,200-frame fights took the same time (99 and 109 s, then 99 and
    92 s). An earlier gap of 2× between two runs came from the machine: the
    i9-12900K's 24 threads mix 8 performance and 8 efficiency cores, and
    throughput varies by up to 2× between runs.
  - Time per fight: battle step 39%, baseline decisions 34%, encoding 19%,
    observation context 8%. Decisions are 52–62% of the fight against the
    Kzer-Za, Orz, Chenjesu and Androsynth, nearly all in rule 1 predicting
    homing fighters, Dogi and marines (`threat_path`, `first_contact`).
  - `threat_path` and `first_contact` now bind names locally and inline
    `bearing_of` and `_segment_closest` with the same arithmetic. Without
    gravity, a homing threat's velocity is recomputed only after it turns.
    Every ship against every opponent, one seed (625 fights), gave identical
    results before and after. Time fell 5.5% over all opponents (46.3 to
    43.7 s), and by 18% against the Kzer-Za, 12% against the Chenjesu and 8%
    against the Orz.
- **Workers:** one per logical CPU by default. Fixed-length fights are equal in
  length, so more workers keep paying off. 384 fights of 1,200 frames ran at:

  | Workers | Frames per second |
  |---|---|
  | 24 | 6,430 |
  | 20 | 6,069 |
  | 16 | 5,718 |
  | 12 | 5,126 |

  An earlier measurement favoured 16 over 22 workers, but on first-kill fights
  of uneven length, which left workers idle at the end.
- The battle simulation is the game's own code and is left alone.

### Process

1. Extract shared behaviours as engine helpers.
2. Add a per-ship parameter table: distance band, recharge threshold, gates and
   planet horizon.
3. Move one ship at a time. Compare it before and after by the harm ranking
   (see "Evaluation"). Harm dealt and taken, kills and deaths, the share of fights
   with no harm dealt, and still frames are diagnostics.
4. A moved ship uses the shared behaviours and its table entry throughout,
   against every opponent. It doesn't keep its own movement rules or fall back
   to its earlier rules for some opponents. Find settings within the schema
   that work: per-opponent settings where needed, and new parameters or
   behaviours when the schema can't express what a ship needs.
5. The moved ship should rank about as well as before, or better. Where no
   setting gets there, marginally worse is acceptable.
6. Commit each ship separately. Revert any ship that regresses by more than
   that.
7. Survival-focused tunings from Phase 2 are rechecked by the harm ranking when their
   ship moves. Examples: the Orz at 1,550–1,800 from the Chmmr, the Spathi at
   1,000–1,300, and the Arilou hold.

### Passes

Every retuned ship changes the opponents every other ship was tuned against,
so ships are tuned in passes of growing effort, like tightening wheel nuts in
turn, rather than each ship finished before the next starts. A ship's
defaults weigh all 25 opponents, so one opponent changing moves them little.
A per-opponent setting rests wholly on that opponent's behaviour, so it is
tuned only once that opponent has moved.

1. **Pass 1, every ship, light effort.** Move the ship to `shared` and screen
   only its defaults (band, recharge, weapon use) at the default perception
   delay; λ is set, not screened. Add a per-opponent override only where the defaults plainly fail
   against that opponent: no harm dealt, or harm taken far above harm dealt.
   Earlier overrides are kept only where the data still clearly need them.
   Steps 3–7 of "Process" apply to each ship as it moves.
2. **Pass 2, per-opponent settings.** With every ship moved, screen the
   per-opponent overrides, now against opponents that have settled.
3. **Pass 3, changed pairs.** When ship B's settings change, recheck the
   overrides other ships hold against B; other pairs aren't rerun.
4. **Settling.** After each pass, fly every ship against every opponent on
   the same seeds (5 seeds × 4 cells). Passes stop when no ship's harm dealt or
   λ score moves by more than its 95% interval. Settings are screened on fixed
   seeds and confirmed on unused ones; fights are deterministic per seed, so a
   setting that flips back and forth between passes shows as a cycle. Matchups
   can beat each other in a loop, so passes may cycle rather than settle; a
   cycle is resolved by keeping the setting that ranks better over both
   passes. Baselines are sparring partners, so stable and sensible is enough.

### Irregular fire against shields

A shield that is held drains battery whether or not shots arrive: the Utwig's
costs `a2_cost` every 6 frames, and absorbed hits refill it. A ship that fires
whenever it can lets a shielding opponent time its shield to the fire, and,
against the Utwig, feeds the battery that keeps it up. Firing at irregular
moments might make the shield cost more battery than it earns, and catch it
down. Lasers, which arrive the frame they are fired, and ships that fire fast
from close are the candidates. Whether this works is not known, so it is
checked rather than assumed.

- **Deterministic gate.** Fights must stay repeatable and the battle's random
  stream untouched, so the gate draws no random numbers. While in range, a
  shot that would otherwise fire goes ahead only when a hash of the quantised
  distance and the exact bearing to the target falls under the fire share.
  Both change every frame the ships move, so the pattern is irregular without
  being random. A new `aimed` setting, `fire_share` (1.0, the default, is
  always-on firing), holds the share.
- **Check.** For each candidate shooter (at least the Arilou, Chmmr, Vux and
  Slylandro, plus any close-range fast firer that the shot outcomes show is
  mostly absorbed) against the Utwig and Yehat, screen `fire_share` 1.0, 0.75,
  0.5 and 0.33 by the harm ranking. Report the shot outcomes (absorbed and hit
  shares) and the shielding ship's mean battery.
- **Keep it** as a per-opponent override only where a share below 1.0 is
  clearly ahead. Today's Utwig and Yehat rules shield on an imminent shot or a
  crew loss (and against the Chmmr, wherever its lasers reach), so the result
  depends on those rules and is rechecked when the shield ships move.

**Result** (after pass 1, delay 5). `WeaponUse.fire_share` gates aimed and
launched shots through `engine.fire_gate_open` (a CRC of the distance to the
nearest unit and the exact bearing). The Arilou, Chmmr, Slylandro and Spathi
weapon 1 were screened at shares 1.0, 0.75, 0.5 and 0.33 against the moved
Utwig and Yehat, then shot outcomes and the shielding ship's mean battery
were counted on seeds 0–19 × 4 cells. No share below 1.0 was clearly ahead
anywhere, so no override is kept:

- **Utwig:** the Arilou laser hits 98% and the Slylandro lightning 67% at
  every share, none absorbed, so there is no shield to beat; lower shares
  only cost harm (clearly behind for the Slylandro). The Chmmr laser is 87%
  absorbed (its Utwig rule shields wherever the laser reaches), and lower
  shares make that worse (98% at 0.33) while dealing clearly less. The
  Utwig's mean battery doesn't fall at any share (7.5 against the Arilou,
  about 14 against the Chmmr). The Spathi is level at 0.75.
- **Yehat:** lower shares do catch the shield down more often: the Arilou
  hit share rises from 25% to 34% at 0.33 and the Slylandro's from 53% to
  61%, with absorbed falling. But fewer shots fire, so harm dealt is level
  (Chmmr, Slylandro) or clearly lower (Arilou at every lower share). The
  Spathi deals next to nothing at any share.
- **Vux:** not checked: its laser fires inside its whole-rule special,
  which the share doesn't gate. It is checked when the Vux's rules are
  split into the schema in pass 2.

Script and results: `%USERPROFILE%\StarAI-measurements\phase6\fire_share.*`.

### Ship checks

Questions to settle when the ship comes up in a pass, beyond its screen:

- **Androsynth against the Chmmr.** Today it saves its battery and blazes in
  once, from close in, on a full battery; the bubbles (A1) aren't fired.
  Check whether the bubbles get through the satellites, which shoot shots
  down, with 3, 2, 1 and 0 satellites up (the strong and weak cells cover 3
  and 1; respawns and kills give the others). Report the bubbles' shot
  outcomes (`shot_outcomes.ShotTracker`: hit and shot-down shares) and harm
  dealt and taken per satellite count. If bubbles land, compare the current
  ram with firing bubbles from outside the satellites' reach and keeping the
  blazer (A2) to escape the tractor and a closing Chmmr: A2 for escape, A1
  for damage. Keep whichever ranks higher, per satellite count if that
  differs (the special can read `ctx.chmmr_satellites()`).

  **Result** (pass 1, delay 5; 20 seeds per count, the Chmmr at full crew
  with 0, 1, 2 or 3 satellites). The bubbles don't get through: they hit
  8–13% of the time with no satellites up, 4–6% with one, 2–4% with two and
  0–1% with three; the satellites shoot down 31–37%, 41–55% and 53–65%,
  and with none up about 90% expire or are outrun. Bubbles from 900–1,500
  deal 0.04–0.09 harm per fight; the ram deals 1.21 (no satellites), 2.33,
  1.84 and 1.72 while taking 2.3–7.2. Bubbles with the blazer kept to escape
  cut harm taken to about 0.1 but deal 0.01–0.04, a stall the ranking
  rejects. The ram stays at every satellite count. Script and results:
  `~/StarAI-measurements/phase6/androsynth_chmmr.*`, outside the repository.

---

## Phase 6B: Perception Delay For Computer Control

### Problem

Computer controllers react to an enemy shot on the frame it appears. Measured
2026-11-21 (5 seeds × 4 cells per pairing, every shot followed to its end):

- **Arilou:** of more than 1,000 Spathi shots, 2 hit. About 60% of forward
  shots were within 400 of the Arilou when it teleported; it outflew the
  rest. The baseline teleports whenever any threat is due within 0.1 s, and
  its battery (3 per teleport, about 3.4 regained per second) never runs
  short.
- **Yehat:** its shield absorbed 62–69% of Spathi forward shots and 58% of
  missiles; 1–6 of about 450 forward shots hit. With 2 battery back every 7
  frames against a 3-battery, 10-frame shield, it can shield almost
  continuously.
- **Utwig:** its shield absorbed 70–79% of forward shots, and its bolts shot
  down most of the rest. Absorbed hits recharge it.
- **Lasers, which give no warning, do land** (3 seeds × 4 cells):
  - **Arilou:** 302 Slylandro hits in 12 fights, during which it teleported
    once.
  - **Yehat:** 52–77% of Arilou, Slylandro and Vux laser attempts landed.
  - **Utwig:** all 424 Arilou laser attempts landed.
  - **Exceptions:** shields block most laser attempts where a rule keeps them
    up in advance (the Phase 2 rules against the Chmmr), or where a burst's
    first hit triggers the crew-loss shield (Mmrnmrhm).

The advantage is the warning, not the abilities. A human needs about 0.2 s to
react, and can't respond to a shot fired from close by. UQM's Arilou AI
(`arilou_intelligence`) is imperfect by design: it teleports on a random 1
frame in 4 while a shot is due within 3 frames, and below its top difficulty
it ignores non-homing shots. A rule like that would bind only the baseline,
and a trained network could still learn perfect reflexes, so the limit goes
in the observation both share.

### Requirements

1. **Setting.**
   - "Perception delay (frames)" on the Game Play settings page, an integer
     from 0 to 24. The default is 5 (0.21 s at 24 FPS).
   - Stored with the other game-play settings (`DEFAULT_GAMEPLAY`,
     `GameSettings`). A settings file without it loads the default.
   - Applied at runtime with the other game settings (`apply_game_settings`).
2. **What is delayed.**
   - In a computer controller's observation, an object owned by the enemy
     ship (its root owner, followed up the parents) is left out until it has
     been in the world for the delay. That covers projectiles, fighters,
     marines, limpets in flight, mines, Dogi, saws, shards, and any object a
     release or a parent object spawns.
   - The age counts frames since the object entered the world. A respawned
     ship's objects, Chmmr satellites included, are new objects and wait like
     any other.
   - A delay of 0 gives observations identical to today's.
3. **What is not delayed.**
   - The ships themselves, including a respawned enemy ship.
   - My own objects, which I know I fired.
   - Neutral objects: the planet, asteroids, and Syreen crew pods.
   - My own state, including limpets already attached to my ship and crew
     lost.
4. **Left out means left out everywhere.** An object under the delay is
   absent from every observation feature built from objects: the slots, and
   any counts or nearest-object values drawn from them. It must not leak
   through a count before it appears in a slot.
5. **Per viewer.** The arena and training build one observation context per
   frame for both ships (`build_observation_context`). The delay filters per
   viewer, as each ship's enemy objects differ. Visibility depends only on the
   viewer, so the shared context still saves the work.
6. **Game rules are unchanged.**
   - Targeting inside the game rules reads the live world as today. The Chmmr
     satellites (`ChmmrSatellite._select_laser_target`) and the Earthling
     point defence choose their targets as now: with Earthling A2 held, it
     shoots down a 1 HP projectile younger than the delay.
   - Homing weapons, fighters and marines steer as now.
   - What the delay changes is the controller's decisions. A baseline or
     network deciding whether to hold Earthling A2 doesn't know of an enemy
     shot younger than the delay.
7. **Who is affected.**
   - Every controller that decides from the encoded observation: baseline
     strategies (trainee phases, training opponents, battle opponents, the
     arena) and trained networks.
   - Human players don't use the encoder, so human play is unchanged.
   - Verify that no baseline path reads the live world for enemy objects
     instead of the observation.
8. **Models.**
   - The training metadata records the delay (`perception_delay_frames`),
     as it records `ship_directions` (`model_registry`).
   - A model saved without it was trained with a delay of 0.
   - Resuming or loading a model with a different delay is handled as a
     ship-direction mismatch is handled today.
9. **Arena.** `--perception-delay` sets the delay, default 5, independent of
   the settings file so runs repeat. The report states the delay used.

### Measurement

- **Shot outcomes:** the Spathi against the Arilou, Yehat and Utwig, and
  laser ships against them, at delays 0 and 5. Hit, absorbed, teleported and
  shot-down shares for each.
- **Every ship:** every ship against every opponent, 5 seeds × 4 cells, at
  delays 0 and 5. Harm dealt and taken per ship, to show which ships lose most
  of their defence. Expect the Arilou, Yehat and Utwig to lose most, and rule
  1 dodges and shoot-downs of close shots to weaken for every ship.
- **Tuned rules:** recheck the Phase 2 and Phase 6 rules that rest on
  reactive defence. Examples: the Utwig and Yehat against the Chmmr, the
  Phase 3A shoot-downs, and the Spathi settings.
- **Other delays:** 3 and 8 frames, to show how sensitive the results are to
  the default.

### Tests

- An enemy object is absent from the observation for the delay, and present
  from the frame it reaches it.
- Own objects, ships, the planet, asteroids and crew pods are present at
  once.
- A delay of 0 gives observations identical to today's.
- An object under the delay leaks into no count or slot.
- Per-viewer filtering with one shared context: each ship sees its own
  objects at once and the other's after the delay.
- A respawned enemy's new objects wait for the delay.
- Chmmr satellites still shoot a projectile younger than the delay, and a
  held Earthling A2 still shoots one down.
- Settings: default 5, range 0–24, saved and loaded; an old settings file
  loads 5.
- Model metadata records the delay; a model without it reads as 0.

### Order

Phase 6B is implemented before the Spathi's settings are confirmed. It changes
how every opponent defends against the Spathi's shots, and how the Spathi
defends. Screening already flown at delay 0 shortlists settings, and the
shortlist is screened again at the default delay before confirmation.

---

## Decisions

Recorded 2026-11-20.

| # | Topic | Decision |
|---|---|---|
| D1 | Damage normalisation | Keep the 0.5 weight on crew damage. A full crew kill gives 0.5 D, and limpets make up the rest. |
| D2 | Penalty weights | The same formulas as the rewards, with their own weights set separately, as today. Default 0. |
| D3 | Pkunk | Halve both damage and kills: damage given to a Pkunk and damage a Pkunk trainee takes. |
| D4 | Self-inflicted own losses | Penalised. |
| D5 | Opponent deaths the trainee didn't cause | No kill credit, as today. |
| D6 | Half-life range | 1–30 s, default 5 s, with a warning whenever the cutoff exceeds the match frame length set on the Regimen tab. |
| D7 | Satellite HP bar colours | Full white. Then 9 `#00FF00`, 5 `#FFFF00`, 1 `#FF0000`, interpolated between 1 and 9. |
| D8 | Supox end animation | Only on a hit. Other projectiles unchanged. |
| D9 | Reward schemes | Classic and Harm both kept, switched by a slider at the top of the Rewards pane. |
| D10 | Crew gains and losses | Opponent: for every opponent, the trainee is credited only for new lows it causes; gains and losses it didn't cause just move the low-water mark. Trainee: the penalty follows the crew level, so every loss is penalised and every gain credited, netting to the real change in crew. |
| D11 | Perception delay | Recorded 2026-11-21. Computer controllers see enemy objects only once they are N frames old: a Game Play setting, 0–24, default 5. Ships, own objects, neutral objects and own state are not delayed. Game-rule targeting (Chmmr satellites, Earthling point defence, homing) still sees the live world. Chosen over a delay on everything, which would mistime every aim and dodge, and over UQM's random gate, which would bind only the baseline and not trained networks. |

---

## Progress Tracker

| Phase | Item | Status | Evidence |
|---|---|---|---|
| 1 | Satellite HP bars | Implemented | `hp_bar_appearance` and `ChmmrSatellite._draw_hp_bar`; `SatelliteHpBarTests` in `tests/test_chmmr.py`; render checked at 10, 9, 7, 5, 3 and 1 HP. Full suite 1,502 tests OK. |
| 1 | Supox end animation on hit only | Implemented | `end_anim_on_hit_only` ability field, set for `SupoxA1`; `destroy_projectile(dealt_damage=...)` is false for laser, area and planet deaths and for pair contacts that dealt no damage; `tests/test_supox_end_animation.py`. |
| 2 | Shared satellite helpers | Implemented | Engine: `chmmr_satellites` (handles the arena wrap), `satellite_offset` (orbit prediction, exact against the game at 6 and 12 frames), `intercept`, `satellite_shot`, `satellite_ring`, `satellite_on_beam`, `satellites_in_reach_of_me`, `gun_error`/`aim_gun`, `closing_reach` and `keep_standoff` (run early from a fast closer allowing for turn time and smoothed acceleration; approach below a speed cap). Arena reports satellite HP removed and fight length. Tests: `ChmmrSatelliteHelperTests`, `ChmmrMatchupMoreTests`. |
| 2.1 | Thraddash | Implemented | It never fired before and dashed through satellite range. Now: no dash; hold 1,000–1,450; run tail-first (laying flames) early from a fast closer; shoot satellites with orbit-predicted aim, or the Chmmr when lined up. 40 seeds: 0-3-37, crew dealt 9%, satellite HP 13.8, fight 96 s (before: 0-20-0, 1%, 0.1, 4 s). |
| 2.2 | Arilou | Implemented | Nose away from the Chmmr, thrust only inside 650; close in only while able to turn away in time; fire when a satellite will lie on the beam. 40 seeds: 0-24-16, crew dealt 2%, satellite HP 9.9, fight 58 s (before: 0-20-0, 2%, 0.0, 5 s). A 670 hold was worse (3.3 HP). |
| 2.3 | Supox | Implemented | Aims at the weakest satellite only while settled in its band and not faced; keeps its Chmmr-facing slides otherwise (aiming at satellites everywhere cut its life from 34 s to 9 s). Fires when on a satellite or the Chmmr. 40 seeds: 0-33-7, crew dealt 3%, satellite HP 0.8, fight 36 s (same seeds without it: 0-36-4, 2%, 0.1, 34 s). |
| 2.4 | Pkunk | Dropped | Tried circling at 620–680 and shooting passing satellites: 40 seeds 0-40-0, satellite HP 0.2, fight 11 s, no better than before. The tractor flings it to about 184 units/frame (mass 1), so it can't hold any band. Removed. |
| 2.5 | Vux | Implemented | Lasers a satellite predicted on the forward beam; limpets only while a satellite has the Vux in range (so satellites shoot the ship, not the limpets). 40 seeds: 0-40-0, crew dealt 47%, satellite HP 2.4, fight 4 s (before, 20 seeds: 47%, 2.1, 5 s). Little time alive to act. |
| 2.6 | Orz | Implemented | Marines only with ≤ 1 satellite left; standoff 900–1,250 with `keep_standoff`; turret at satellites. 40 seeds: 0-23-17, crew dealt 1%, satellite HP 0.0, fight 73 s (before, 20 seeds: 0-20-0, 4%, 0.0, 7 s). With marines ungated it dealt 4% but survived only 9 of 40. It spends most of the fight running, as the Chmmr matches its top speed; firing the turret backward while running is a possible next step. **Re-tuned for the 2.10a tractor:** the standoff moved to 1550–1800, beyond the 1500 pull band. 40 seeds against the new tractor: 0-27-13, 0% crew dealt, 78% taken with the old band; 0-14-26, 1%, 48%, 0.3 satellite HP, 82 s with the new one (1300–1550: 0-21-19; 1450–1700: 0-13-27; 700–1000: 0-40). From there the turret barely reaches a satellite, so the Orz mostly survives rather than deals harm. |
| 2.6a | Orz turret (all opponents) | Implemented | Fixed the observed turret angle: the encoder subtracted the hull rotation in degrees from the turret heading in ship directions, so `orz_turret_relative_sine/cosine` was wrong at any nonzero heading. This affected trained networks as well as the baseline. Baseline: `turn_turret` turns the turret (A2 + turn) only while the hull holds its heading and nothing fires. Strategy aims with the hull when steering is free (twice as quick) and with the turret while a rule holds the heading (approach, back-off, running). The Kohr-Ah and Chenjesu keep hull-only aiming, as their 10-HP shots must be dodged and the turret cost about 12 wins there. Kites (standoff 700–1,100, turret firing back) Arilou, Druuge, Kzer-Za, Orz, Pkunk, Shofixti, Spathi, Syreen and Yehat, chosen from a 40-seed kite-versus-default comparison. The evade rule can shoot a threat down with the turret, keeping course; the arena shows almost no effect from this. 20 seeds against all opponents: 341-109-1D-49T (before: 219-249-7D-25T); bug fix alone with hull aiming: 273-200. Chmmr: 0-10-10, crew dealt 4%, satellite HP 0.8. Shells (2 HP) are shot down by satellite lasers (one every 3 frames each) and the Chmmr laser before they land. A laser-survival filter was tried and dropped, as the shells still died. |
| 2.7 | Kzer-Za (confirm only) | Confirmed | Already launches fighters only with 0 satellites left. 20 seeds before: 0-6-14, crew dealt 25%, satellite HP 2.1. |
| 2.8 | Androsynth | Implemented | The blazer can't hit satellites: they collide only with projectiles, and in a simulated pass a blazer within 72 of one left it at 10 HP. Inside their ring the lasers take about 1 crew a frame, so the crew buys one ram. With satellites up: no bubbles, the nose kept on the Chmmr, blaze from 800 on a battery at least 90% full, then ram at the intercept point for the blazer's speed. 80 seeds: 0-80-0, crew dealt 38%, fight 3 s (old rules on the same seeds: 0-78-2T, 15%, satellite HP 0.4, 18 s). Blaze ranges 600–1,300 gave 33–39% with leading; steering straight at the Chmmr gave 26–28%. |
| 2.8a | Blazer intercept steering, all opponents but the Supox and Mmrnmrhm | Implemented | `ANDROSYNTH_STRAIGHT_RAM_OPPONENTS` keeps the Supox and Mmrnmrhm on a straight ram; every other blazer steers at `_blazer_intercept_bearing`. 20 seeds against all opponents: 209-169-3D-119T (before: 177-186-6D-131T). Thraddash 13-5 → 16-0, Melnorme 3-17 → 10-10, Vux 6-0 → 10-0, Syreen 12-8 → 15-5; Supox and Mmrnmrhm unchanged; Slylandro 16-2 → 15-4. Tests: `AndrosynthBlazerSteeringTests`. |
| 2.8b | Blazer damages satellites, then re-evaluate the Androsynth against the Chmmr | Implemented | Rule: `ChmmrSatellite.should_collide_with_ship` accepts only an enemy blazer; `ram_contact_allowed` allows one ram (3 HP) per contact, checked in `_apply_ramming_damage_to_projectile_like`; the blazer bounces off the satellite as off a planet (`bounce_off_static_body`, velocity kept in `collision_velocity`) and takes no damage; the satellite keeps its orbit. Other hulls and the base Androsynth pass through. Tests: `BlazerRamsSatelliteTests` in `tests/test_chmmr.py`. Re-evaluation with the bounce, 80 seeds, scored as 0.5 × crew dealt + 0.5 × satellite HP ÷ 30: ram the Chmmr 39% crew, 0.9 satellite HP (0.21); ram the satellite met soonest 8%, 5.5 (0.13); prograde 10%, 2.5 (0.09); retrograde 6%, 3.4 (0.09). Prograde and retrograde steer at a point 200 ahead of or behind the satellite on its orbit (relative to the Chmmr), then straight at it within 300; measured at contact, prograde rams met the satellite's orbital motion at a median 50° and retrograde at 150°. The blazer lands only one or two rams before the lasers kill it, so the Chmmr ram is kept. |
| 2.9 | Ilwrath | Investigated, no change | 40 seeds against the Chmmr: today 0% crew dealt, 0 satellite HP, 12 s alive. Every flame (1 HP) dies on its first frame: the satellites shoot enemy objects with the lowest HP first, and three of them, each firing every 3 frames, match the Ilwrath's one flame a frame; head-on, the Chmmr laser hits the flame in its line before the hull. The Chmmr baseline also sees the cloaked Ilwrath (observations carry a cloaked ship's position) and lasers it from 500–600, so the cloak doesn't protect it. Variants tried: nose on the Chmmr while closing, no holding back (1% crew, 0.2 satellite HP, 5 s alive); the same without recloaking once in reach (0%, 0.2, 5 s). Even against a Chmmr that holds fire on a cloaked target the best variant deals 9% (6 s alive), and against one fully blind to it 4% (0.5 satellite HP). Point blank, 40 seeds each: stay cloaked until the next decision would be within 150, 200 or 250; uncloak by the first flame (which snaps the nose to the 4-frame look-ahead) or by A2; then flame every frame without recloaking, with or without thrusting back in after the bounce off the Chmmr. All twelve deal 0–2% crew and 0–0.2 satellite HP, and the Ilwrath dies in 5 s: it arrives with about half its crew after the Chmmr laser, the bounce (about 45 outward, against 5 thrust a frame) carries it out of reach, and it fires only 2–7 flames before dying, most shot down by the satellites. No variant raises harm without shortening the time alive, so the default strategy is kept. |
| 2.10 | Utwig | Implemented | `_utwig_vs_chmmr` in `strategies.py`, used against the Chmmr with or without satellites. It holds the Chmmr between 800 and 1050 (`keep_standoff`): beyond its 600 laser, under the 1000 at which the Chmmr tractors, and within the bolts' 1080. From the band it fires at the Chmmr. While running from it within 650 (the Chmmr laser plus 50, which also covers the satellites: orbit 256 plus laser 256) it holds the shield, so the lasers charge its battery instead of taking crew; the Chmmr drains its own battery on the shield and is then open to the bolts. Other shield triggers (imminent hit, crew lost) are kept. Other opponents keep `_shield_rules`. 80 seeds: 80-0, 100% crew dealt, 0% taken, 7.1 satellite HP, 27 s, against 0-80, 15%, 1.4 and 11 s before. Measured on the way (80 seeds): the band alone, shielding only on imminent hits or lost crew, 36-44 and 50% crew dealt (700–1000: 31-49; 850–1080: 32-48; 900–1080: 28-52; 1000–1250, beyond the bolts: 0-38); the band without any shield 20-60; firing at satellites instead of the Chmmr 14-66 with 9.9 satellite HP; waiting without thrust beyond the band (40 seeds) 5-35, as a still Utwig can't build speed to run. Tests: `UtwigChmmrTests` in `tests/test_baseline_strategy.py`. **Re-tuned for the 2.10a tractor:** the 800–1050 band now lies inside the pull band, so the tractor held the Utwig on the run, never firing (0-77-3 over 80 seeds). It now holds 650–1050, runs from the Chmmr only while the Chmmr's battery is at least 70% (`UTWIG_CHMMR_RUN_BATTERY`), and within 650 shields while running or while the Chmmr has the 2 battery a laser shot costs, firing otherwise. 80 seeds: 51-29, 88% crew dealt, 53% taken, 6.1 satellite HP, 22 s. Run thresholds of 50, 60, 80, 90% and never running gave 39–45 wins; 5–20% gave 0–1; the old 800 inner edge did worse at every threshold tried. |
| 2.10a | Chmmr tractor: pull mid-range opponents in, keep close ones from escaping | Implemented | `_chmmr_tractor_helps` in `strategies.py`: with at least 30% battery (`CHMMR_TRACTOR_BATTERY`), the Chmmr holds the tractor on a visible, inertial opponent (`ship_spec(...).inertial`, which also covers the Arilou, Slylandro and blazer and replaces `TRACTOR_IMMUNE_SHIPS`) that is either between laser reach plus 100 and 1500 (`CHMMR_TRACTOR_FAR`), or within laser reach plus 100 (`CHMMR_TRACTOR_ESCAPE_MARGIN`) and moving away. No pull beyond 1500. Tuning, 20 seeds against every opponent, as the evaluated ship (W-L-T) and as the opponent (wins of the ships facing it, their mean crew dealt): today's rule (beyond 1000 only) 355-42-102, 41, 21.2%; band to 1200 plus escape 405-25-69, 21, 19.2%; the same plus pulling beyond 1200 385-18-97; band to 900 396-25-79; band to 1500 412-21-67, 23, 18.2%; band to 1800 413-23-63; band to 2200 390-23-87; escape only 378-32-90; band only 375-24-101; escape at 10% battery, or both uses at 20% or 40%, within noise of 30%. Re-tuned against it: the Utwig (row 2.10) fell from 20-0 to 0-19, held on the run by the tractor and never firing, and the Orz (row 2.6) from 0-10-10 to 0-12-8 with 0% crew dealt; see those rows. Final, 20 seeds: the Chmmr 392-32-76 as the evaluated ship; facing it, ships win 34 and deal a mean 21.7% crew. Tests: `ChmmrTractorTests` in `tests/test_baseline_strategy.py`. |
| 2.11 | Earthling | Implemented, small gain | Two changes in `earthling()` (`strategies.py`). (1) Nukes launch at 40 without the ship's speed and may hit their own ship (`hit_parent`), so it holds them while flying forward at `EARTHLING_NUKE_OVERRUN_SPEED` (30) or faster, against every opponent. Before, the tractor dragged it toward the Chmmr at about 73 and its first nuke hit itself for 4 in 20 of 20 fights; now 0 of 20. Against all opponents (20 seeds) 154-284 against 151-288. (2) Against the Chmmr, between nukes it steers 90° off the Chmmr's bearing (`EARTHLING_CHMMR_ORBIT_OFFSET`) instead of holding its standoff band, which kept it inside the tractor's reach and let it be dragged through the satellites at twice its top speed. 80 seeds: 0-40 with 40 timeouts, 2% crew dealt, 70% taken, 2.1 satellite HP, 58 s, against 0-80, 1%, 100%, 1.8 and 11 s before. Offsets of 70° and 110° (40 seeds) were about the same. Damage stays small: it orbits at 1500–1900, and the satellites shoot down every nuke (1 HP) about 250 from a satellite, before it gets within 300 of the Chmmr. The point-defense lasers (A2) do hit satellites and the Chmmr, but reach only 400, inside the Chmmr laser's 600. Against the 2.10a tractor (20 seeds): 0-14-6, 5% crew dealt, 76% taken, 2.1 satellite HP, 58 s, so the orbit is kept. Tests: `EarthlingNukeAndChmmrTests` in `tests/test_baseline_strategy.py`. |
| 2.12 | Umgah | Implemented, small gain | `_umgah_vs_chmmr` in `strategies.py`: face the Chmmr without thrust, keep the cone (A1, free) on throughout, and hop backward (A2) within 120 (`UMGAH_CHMMR_HOP_DISTANCE`) to stay in cone reach instead of bouncing off the hull. The cone does damage satellites: one held in it went from 10 HP to destroyed within 12 frames. The tractor pulls 12 ÷ mass, and the Umgah's mass is 1, so it is dragged in at about 100 a frame, faster than any hop could keep it away; it loses about half its crew to the lasers on the way in and has a few frames in cone reach. 80 seeds: 0-80, 17% crew dealt, 0.7 satellite HP, 7 s, against 0-80, 0%, 0.0 and 4 s before. Cone only while the Chmmr was seen in reach: 3% (hover at 120) or 1% (no hop); holding the cone from 400: 10%, 700: 13%, 1000 or more: 17%, as at about 100 a frame the Chmmr crosses the cone's reach between two decisions. With the cone gated on reach, hopping back at 200 instead of 120 gave 3%, and hopping to brake whenever closing faster than 30 within 700 gave 1%, though alive 18 s. Tests: `UmgahChmmrTests` in `tests/test_baseline_strategy.py`. |
| 2.13 | Slylandro, ZoqFotPik, Spathi, Syreen, Yehat | Implemented for four; ZoqFotPik unchanged | Each applies only against the Chmmr (`_<ship>_vs_chmmr` in `strategies.py`). 80 seeds each, all still 0 wins; before → after as crew dealt, satellite HP, time alive. **Slylandro** 6%, 1.6, 6 s → 8%, 3.4, 31 s: the tractor can't pull it and it outpaces the Chmmr, so it holds 700–850 (`SLYLANDRO_CHMMR_NEAR`/`FAR`) with `keep_standoff`, circles inside the band, and fires lightning within 1,000 (`SLYLANDRO_CHMMR_FIRE`); the bolt seeks the Chmmr by itself and strikes satellites in its path. It harvests asteroids only below 10% battery. Bands of 620–720 (4%, 9 s) and 800–1,000 (6%, 1.8 HP) were worse. **Spathi** 1%, 0.1, 12 s → 3%, 2.2, 90 s (64 of 80 reach the time limit): holds 1,000–1,300 (`SPATHI_CHMMR_NEAR`/`FAR`), shooting the satellite `satellite_shot` picks, or the Chmmr when lined up. No rear missiles: the satellites shoot them all down, and at 3 battery each they kept the battery near empty (with them, at 1,000–1,300: 2%, 1.2 HP). Nearer bands died sooner (650–1,200: 22 s; 800–1,300: 49 s). **Syreen** 9%, 0.0, 3 s → 14%, 0.0, 8 s: the tractor pulls the mass-2 Syreen in from 1,500 faster than it can thrust away, so no band holds (a 650–780 ring with songs from 800 dealt 6%). It faces the Chmmr without thrust and sings within 200 (`SYREEN_CHMMR_SONG_DISTANCE`), where a song takes the most crew. Singing within 150, 250 or 300 dealt 10–12%; charging in and singing within 200 dealt 12% but lived 3 s. **Yehat** 6%, 1.1, 14 s → 9%, 1.5, 21 s: lasers give no warning, so within the Chmmr laser's 600 (which covers the satellites' reach) it refreshes the shield as soon as it drops (`damage_shield_active`), and fires between with at least 1 battery left after the shot (`YEHAT_CHMMR_FIRE_RESERVE`). It still approaches to gun range. Charging in dealt 3–6%; waiting shielded survived longest (77 s) but dealt 5%; a band of 650–950 dealt 0%. **ZoqFotPik** unchanged at 5%, 0.1, 3 s: its 10 crew die to the Chmmr laser on the way in before the tongue lands, and its 1-HP shots are shot down. Bands (650–1,300) lived 14–48 s but dealt 1% and at most 0.3 satellite HP; waiting nose-on for the tongue dealt 4%. Tests: `SlylandroChmmrTests`, `SpathiChmmrTests`, `SyreenChmmrTests`, `YehatChmmrTests` in `tests/test_baseline_strategy.py`. |
| 3 | Planet lookahead by agility | Implemented | `planet_horizon_frames` in `engine.py` replaces `PLANET_AVOID_HORIZON_SECONDS`: `turn_about_frames` (8 heading steps, the first at once, then one every `turn_wait + 1`) plus `top_speed_frames` (pushes from rest to `max_thrust`, the first at once, then one every `thrust_wait + 1`; 0 for the Arilou and Slylandro) plus `PLANET_HORIZON_MARGIN_FRAMES` (12), clamped to 1–4 s. Horizons: Arilou, Slylandro and Pkunk 1 s (the floor); Supox 1.33 s; Orz, Spathi 1.4 s; Chenjesu 4 s (the cap). Measured over 6 seeds of every ship against the Spathi, Earthling, Kzer-Za and Pkunk (1,152 ship-fights), counting each ship's new planet contacts and the frames rule 0 claims: fixed 4 s, 71 hits, 2 planet deaths, rule 0 on 2.6% of frames; margin 0, 73, 0, 1.4%; margin 12, 70, 0, 1.5%; margin 24, 67, 1, 1.7%. Rule 0 frames for the Arilou 5.6% → 0.8%, Pkunk 3.0% → 1.1%, Supox 2.1% → 0.6% (margin 12). Hit counts per ship are small and move by 1–2 either way; the Pkunk's 21 hits are the same under every horizon, so they don't come from rule 0. Tests: four horizon tests in `PlanetRuleTests` (`tests/test_baseline_strategy.py`). |
| 3A | Pre-emptible held weapons (Chenjesu, Kohr-Ah, Melnorme) | Implemented | `HELD_WEAPON_SHIPS` in `engine.py`. <ul><li>**Value of the held shot** (`held_shot_value`): a crystal or saw in flight is worth its damage when its straight-line closest pass of the enemy comes within the enemy's hull reach, falling to 0 at twice that, and 0 when receding or more than 72 frames away (`held_pass_share`). The Melnorme charge is worth its current damage when the enemy is in range along the nose, else half of it (`MELNORME_UNAIMED_CHARGE_WEIGHT`).</li><li>**Pre-emption:** a rule 1 shoot-down adds the held shot's value to its cost.</li><li>**Release mechanics:** with A1 held, the shoot-down plan releases on frame 0 and fires from frame 1, and keeps A1 released until the nose is on, so the strategy can't fire early. The Melnorme shoot-down turns with the charge held and lets it fly, at its current damage, once on the threat.</li><li>**Release effects and shots already out** (`stopped_threats`): for these ships, rule 1 drops a threat that one of my shots (a crystal, shard, flying or parked saw, or the Melnorme charge at the nose) is due to meet, and can destroy, before the threat arrives. A crystal or saw is kept held while it is due to meet such a threat (`held_shot_meets_threat`). Shards a release would make are not predicted.</li><li>**Shoot-down exclusions:** the Melnorme is removed. The Kohr-Ah stays excluded: over 30 seeds, its shoot-downs cut crew lost by 0.9/min but crew dealt by 1.3/min.</li><li>**Dropped:** letting go of a crystal or saw that was closing but due to miss, once a new shot was lined up. Over 30 seeds it took the Chenjesu from 135-74 to 130-78 and the Kohr-Ah's crew dealt from 47.1 to 42.6/min (7 timeouts against the Druuge).</li></ul>Measured over 60 seeds against the Druuge, Kzer-Za, Orz, Spathi, Chenjesu, Mycon and Earthling (420 fights per ship), before → after:<ul><li>Chenjesu: W-L 269-148 → 265-152; crew lost to shootable objects 33.5 → 33.2/min; crew dealt 33.5 → 33.7/min. Flat overall: the Chenjesu mirror improved (lost 32.9 → 30.3/min) and the Orz matchup got worse (9-51 → 4-56).</li><li>Kohr-Ah: W-L 363-56 → 358-61; lost 23.4 → 23.5/min; dealt 45.7 → 45.7/min. Unchanged except against the changed Chenjesu.</li><li>Melnorme: W-L 251-165 → 259-157; lost 26.6 → 25.8/min; dealt 41.8 → 42.6/min.</li></ul>Tests: `HeldWeaponTests` in `tests/test_baseline_strategy.py`. |
| 4 | Discount half-life setting | Implemented | <ul><li>**Conversions** in `rewards.py`: `gamma_for_half_life`, `half_life_for_gamma` and `half_life_cutoff_frames`, with `DISCOUNT_HALF_LIFE_MIN_SECONDS` (1), `DISCOUNT_HALF_LIFE_MAX_SECONDS` (30) and `DEFAULT_DISCOUNT_HALF_LIFE_SECONDS` (5). The training pipeline still takes gamma, which the setting derives.</li><li>**Setting:** the Regimen slider "Discount half-life (s)" replaces "Gamma" in `train_ai.py`. It runs 1.0–30.0 s in 0.1 s steps, and its value shows the cutoff, e.g. "5.0 (cutoff 33 s)". `TrainingUIState.discount_half_life` replaces the `gamma` field, and `TrainingUIState.gamma` is now derived from it. The console's "Gamma" row shows "Half-life" instead.</li><li>**Cutoff warning:** `_regimen_control_warnings` adds `DISCOUNT_CUTOFF_WARNING` when the cutoff exceeds the match frame length. It shows like the other Regimen warnings: a warning-coloured value, with the text in the tooltip. It is recomputed from both settings every frame and doesn't block saving or starting.</li><li>**Saved configs:** the regimen metadata stores `discount_half_life`, plus the derived `gamma` for older readers. `_with_discount_half_life_key` turns an old gamma-only config into the half-life it implies, rounded to 0.1 s and kept within the range (0.99 → 2.9 s; 0.95 → 1.0 s). It is applied to saved UI state, to model conditions loaded into the state or the sliders, and to the loaded-settings comparison. Where a half-life is stored, the gamma beside it is ignored.</li><li>**Coordinated runs:** the UI compares `discount_half_life` among the Regimen fields, and `_coordinated_regimen_signature` compares the half-life derived from each config's gamma.</li></ul>Tests: `DiscountHalfLifeTests` in `tests/test_training_rewards.py`; the cutoff warning, cutoff suffix, old-config loading and settings-match tests in `tests/test_train_ai_ui.py`; and `test_coordinated_run_requires_equal_discount_half_lives` in `tests/test_coordinated_training.py`. |
| 4A | Reward magnitude steps ±0.001–1.024 | Implemented | <ul><li>**Steps:** `REWARD_WEIGHT_STEPS` in `rewards.py` holds the 23 values; `REWARD_VALUES` in `train_ai.py` is now that tuple. Labels show three decimals (`_format_reward`).</li><li>**Classic defaults:** Kill enemy 1.024, Reduce enemy crew 0.128, Get debuffed -0.032, Lose crew -0.064, Die -0.512. The Harm scheme doesn't exist yet, so its defaults (D_enemy and K_enemy 1.024, D_self and K_self 0) are set in Phase 5.</li><li>**Scale version:** `REWARD_SCALE_VERSION` (2) is saved as `reward_scale_version` in the model's training metadata and in the saved UI state. Settings without it are version 1.</li><li>**Old configs:** `with_current_reward_scale` rescales version 1 rewards through `legacy_reward_weight_to_step` (divide by 40, nearest step on the log scale, sign kept; 0.01 and 0.02 become 0.001). It is applied to saved UI state, to model conditions loaded into the state or the sliders, and to the loaded-settings comparison, so an old model whose rewards map onto today's sliders counts as matching. `changed_training_groups` now compares whole normalised settings, so the version travels with the rewards.</li><li>**Resume warning:** `legacy_reward_scale_resume` is true for a model with completed batches saved under version 1. Starting one asks first ("…was trained with rewards 40 times today's scale, so its value estimates will take time to relearn. Continue?") and, on confirm, saves its settings under version 2. Start all and Start synced add a sentence counting such models to their prompt, or prompt for it alone. Models being reset aren't counted.</li><li>**Learner:** the value loss is MSE and the optimiser Adam, so nothing else depends on the reward scale; gradient-norm clipping will trigger less often.</li></ul>Tests: `RewardWeightStepTests` in `tests/test_training_rewards.py`; `RewardScaleMigrationTests`, the updated `RewardSliderTests` and defaults test, and `test_individual_start_warns_before_resuming_an_old_reward_scale_model` in `tests/test_train_ai_ui.py`. Full suite 1,597 tests OK. |
| 4B | Decay half-lives in batches | Implemented | <ul><li>**Conversions** in the new `src/training/decay.py`: `decay_factor_for_half_life` (0.5 ^ (1 ÷ n); Off gives 1), `half_life_for_decay_factor` (infinite at 1), `nearest_decay_half_life` (nearest setting on the log scale; 1 is Off) and `normalize_decay_half_life`. `DECAY_HALF_LIFE_VALUES` holds 10–100,000 batches and `"off"`; defaults are `DEFAULT_EPSILON_DECAY_HALF_LIFE` (300) and `DEFAULT_BASELINE_DECAY_HALF_LIFE` (100).</li><li>**Settings:** the Regimen sliders "Epsilon decay half-life" and "Baseline decay half-life" show "300 batches" or "Off". `TrainingUIState.epsilon_decay_half_life` and `baseline_decay_half_life` replace the factor fields; `epsilon_decay` and `baseline_decay` are now properties derived from them, so training and the decay code are unchanged. The baseline label drops "use" so the longest value fits without shrinking the Regimen font; the tooltips follow the spec.</li><li>**Saved configs:** the regimen stores both half-lives, plus the derived factors for older readers (the sessions keep writing the factors). `_with_decay_half_life_keys` turns an old factor-only config into the nearest half-life (0.998 → 300, 0.995 → 100, 0.99 → 100, 0.95 → 10, 1.000 → Off). It is applied with the discount conversion (`_with_half_life_keys`) to saved UI state, to model conditions loaded into the state or the sliders, and to the loaded-settings comparison. Where a half-life is stored, the factor beside it is ignored.</li><li>**Console:** the "Eps decay" row shows the half-life the session's factor implies, e.g. "100 batches".</li><li>**Go to epsilon:** the no-decay warning fires when epsilon decay is Off, and reads "Epsilon decay is Off…".</li><li>**Coordinated runs:** the UI compares the half-life fields among the Regimen fields, and `_coordinated_regimen_signature` compares the half-lives derived from each config's factors.</li></ul>Tests: `tests/test_decay_half_life.py` (values, the factor table, round trip for every value, old factors, normalising); the decay label, old-config loading and settings-match tests and the updated defaults, warning, console and config tests in `tests/test_train_ai_ui.py`; `test_coordinated_run_requires_equal_decay_half_lives` in `tests/test_coordinated_training.py`; the updated baseline settings test in `tests/test_baseline_strategy.py`. Full suite 1,606 tests OK. |
| 5 | Harm reward scheme | Implemented | <ul><li>**Formulas** in the new `src/training/harm.py`: `HarmProfile` (s, HP, satHP, satellite count, p from the catalog; Vmax is the highest catalog cost, 30) and `HarmTracker`, which turns each frame's ledger events into the four components `Damage to enemy`, `Enemy kill`, `Damage taken` and `Own death`. The kill divides satellite kills by the ship's satellite count (3 for the Chmmr).</li><li>**Crew:** effective crew is tracked from crew events. Launches and recoveries emit none (escrow debits and refunds are skipped), and a unit lost for good is a 1-crew loss. Opponent damage counts new lows the trainee causes; the trainee's damage follows its crew level. "Starting crew" is the crew a ship spawns with (training spawns are randomised); HP stays the normal starting crew.</li><li>**Attribution:** harm counts as the trainee's when the event's source, followed up its parents, is the trainee ship. Units left by an earlier trainee life have no parent once it dies, so they earn the current life nothing; in a 2,400-frame Kzer-Za vs Mycon round the game credited such a fighter with a kill that harm didn't.</li><li>**Lives:** the tracker lives on the round's `BattleEventLedger` (`harm_tracker`), keyed by ship object, so opponent low-water marks outlast trainee-life trajectories. Respawns are new objects and start fresh; a Pkunk rebirth restarts tracking. The pre-step crew and limpet counts come from new `RewardDecisionFrame` fields.</li><li>**Placement:** `StagedTrajectoryPipeline._add_harm_components` puts each amount on the effect frame. In causal (and shadow) placement, trainee-caused opponent harm and self-inflicted own losses move to the origin frames of the event's causal credit when it is open; other penalties and all gains stay. In legacy placement everything stays on the effect frame.</li><li>**Schemes:** `REWARD_COMPONENTS` is now the classic components plus the four harm ones. The UI passes training only the active scheme's weights (`effective_reward_weights`), so the training config is unchanged and the tracker runs only when a harm weight is nonzero.</li><li>**UI and storage:** a "Reward scheme" slider (Classic/Harm) heads the Rewards pane, which shows the active scheme's sliders. Harm weights are stored in the rewards under their own keys, defaulting to 1.024, 1.024, 0 and 0; the scheme is stored as `reward_scheme` (default and old configs: classic) and propagates to other instances like other settings. Settings saved before the scheme compare as classic with default harm weights. The console's reward table lists the active scheme's components.</li><li>**Deferred:** the console shows each harm component's mean weighted return per sample, as it does for classic components, so a zero-weight penalty shows 0. Raw per-fight totals (`HarmTracker.totals`) aren't carried to the session status.</li></ul>Real rounds (`run_training_round`, random trainee actions, baseline opponents, 2,400 frames) for the Earthling, Kzer-Za, Chmmr and Vux against the Spathi, Mycon and Chmmr ran without errors in legacy and causal placement, with equal totals in both; e.g. six Earthling deaths gave Own death 3.633 = 6 × √(11/30). Tests: `tests/test_harm_reward.py` (formulas for a 6-crew and a 30-cost ship, the Chmmr, the Vux example, limpet series, Pkunk per life for both sides, the Mycon and no-gain new-low cases, unattributed and self-harm losses, launched crew, the Syreen example, per-frame sums, causal placement, tracking across trajectories); `RewardSchemeTests` and the Rewards pane screen test in `tests/test_train_ai_ui.py`. Full suite 1,631 tests OK. |
| 6 | Shared behaviours, per ship | In progress: shared helpers, schema and harm evaluation (6A) done; pass 1 (see "Passes") done: all 25 ships moved (the Thraddash, Umgah and Vux as whole-rule specials, split up in pass 2); irregular fire against shields checked, no share below 1.0 kept (Vux pending pass 2); pass 2 next | <ul><li>**Measuring:** the arena now reports crew dealt and lost per minute (`dealt_per_minute`, `lost_per_minute`) and still frames (`still_share`: frames alive below `STILL_SPEED_FRACTION`, 10%, of top speed). Fights are deterministic per seed, so a pure refactor is checked by identical results: every ship against every opponent, one seed (625 fights), matched exactly before and after the table and the helpers.</li><li>**Table:** the new `src/training/baseline/behaviours.py` holds `ShipBehaviour` per ship (`SHIP_BEHAVIOURS`, `ship_behaviour`), read through `BaselineContext.behaviour`. It replaces `SHOOT_DOWN_EXCLUDED_SHIPS`, `EVADE_NO_HEADING_SHIPS`, `KEEP_MOVING_EXCLUDED_SHIPS`, `KEEP_MOVING_AIM_SHIPS` and `COAST_ABOVE_TOP_SPEED_SHIPS`, and adds `evade`, `run_from_threats`, `planet_horizon_frames` (None: the Phase 3 horizon), `recharge_battery` and `band`. Opponent-dependent gates (`EVADE_GATES`) stay functions in `strategies.py`; weapon mechanics such as `HELD_WEAPON_SHIPS` stay facts in the engine.</li><li>**Helpers** in `engine.py`: `run_bearing` and the rule 1 run option (with `run_from_threats`, rule 1 also weighs flying straight away from the threats, each weighted by its crew cost over its distance); `recharging` and `recharge_retreat` (below `recharge_battery`, run once the opponent would come within its reach plus a hull and `RECHARGE_REACH_MARGIN`, 150, before the nose could turn away; release 200 farther; never close in); `hold_band` (`keep_standoff` on the given band or the table's); `enemy_reach`, moved from `strategies.py`. </li><li>**Schema:** extended after reading every strategy (see "Schema" under Phase 6): per-opponent overrides (`opponents`, applied by `ship_behaviour(ship, opponent)`), `Band` (as far as possible, outside the opponent's reach, approach speed, posture inside, run angle), recharge hysteresis and run angle, `WeaponUse` per weapon (`aimed`, `launch`, `defend`, `special`) run by `fire_weapon`, and the `specials` and `form_switch` slots (`strategies.SPECIAL_RULES`, `FORM_SWITCHES`). `strategies.shared` runs a ship wholly from its entry.</li><li>**Candidate entries:** an `ArenaMatch` can carry a table entry (`entry`), which the ship then flies through `strategies.shared` (`BaselineController(..., entry=)`, `BaselineContext(entry=)`, `ship_behaviour(..., entry)`); λ still comes from the ship's own entry. `screen(entries=)` screens named entries, so settings are compared without adding strategy variants.</li><li>**Spathi, pass 1** (at perception delay 5): the earlier settings were chosen by wins and losses at delay 0. Band screen, seeds 0–9 × 4 cells × 25 opponents: 756–1,080 closing in at up to 60, with no overrides, led (harm dealt 1.122 per fight against 0.988 for the earlier entry); every other band and approach speed (500–900, 300–756, 900–1,300; 15 or 60) was clearly behind. Per opponent, the earlier entry was clearly ahead only against the Vux (its override) and the Mycon (left for pass 2); the Chmmr and Utwig overrides had stalled (harm dealt 0.01 and 0.02 per fight, against 0.31 and 0.21). Recharge screen with the Vux override kept: no recharging led (1.329) over recharging below 2, 3 (to 6), 5, 5 (to 8) and 7, all clearly behind. Confirmed on seeds 100–119 (2,000 paired fights): harm dealt +0.349 per fight [+0.314, +0.385], λ score +0.263 [+0.225, +0.300]; harm taken 0.488 → 0.890 per fight, an extra 0.40 for 0.35 more dealt. Without the Vux override, harm dealt against the Vux fell by 0.198 [−0.337, −0.059], so it stays: 300–756 at full approach speed, running 110° off its bearing. Entry now: band 756–1,080 at approach 60, no recharging, gun aimed, rear missile launched, λ 0.25, Vux override. The earlier Chmmr rule stays in the `approach` variant.</li><li>**Specials:** every ship's unique rules are named specials (`strategies.SPECIAL_RULES`, `FORM_SWITCHES`), and a `late_specials` slot runs after recharge or the band, before the weapons, for rules that depend on whether movement claimed the steering (the Orz turret; the Melnorme, Mycon, Pkunk, Syreen and Zoq-Fot-Pik guns). The Thraddash, Umgah and Vux run their whole strategies as one special for now (fight for fight as before); their own movement is split up in pass 2. Tests: `tests/test_baseline_specials.py`.</li><li>**Androsynth, pass 1:** the blazer is the mode switch (`androsynth_blazer`: blaze within 1,000 on half a battery, ram while it lasts), bubbles launch keeping 9 battery back, and the Chmmr rule is a special. Screen, 11 settings × 25 opponents, dropping as they fell behind (to 80 seeds): far bands of 600, 1,000, 1,500 and 2,400 (running 135° or 180° off the bearing) × recharging below 2.4 or 12. Every setting recharging only below 2.4 was clearly behind, as were the earlier rules; 1,000 at 135° led over 180° (1.060 against 1.051 harm dealt per fight, within noise). Confirmed on seeds 100–119 (2,000 paired fights): harm dealt +0.061 per fight [+0.022, +0.101], λ score +0.172 [+0.129, +0.214]; harm taken 1.33 → 1.02 per fight. Clearly behind the earlier rules against the Chenjesu, Druuge, Melnorme, Mmrnmrhm and Mycon (pass 2). Entry: band at least 1,000 off, running 135°, recharging below 12 of 24, λ 0.5 (slow in its normal form, it escapes only by blazing, which spends the bubbles' battery). The earlier rules stay as the `earlier` variant.</li><li>**Chenjesu, pass 1:** Dogi and the held crystal are specials. In the first screen every far band flew identically: the crystal special aims the nose every decision and specials run before the band, so no band steered. That (aiming first, no band) dealt 2.481 harm per fight against 1.161 for the earlier rules, which fled at 135° and then aimed. A second screen aimed the crystal after far bands (2,400, 1,000 or 1,000–2,000 off, 600–1,200, 300–800) instead: all clearly behind both, at 0.21–0.88, as moving away leaves the crystal pointing away. Confirmed on seeds 100–119 (2,000 paired fights): harm dealt +1.289 per fight [+1.233, +1.344], λ score +1.574; harm taken 1.65 → 1.00; clearly behind against no opponent. Entry: specials only, no band, λ 1.0 (costly and outrunning nothing, it can't stall). The earlier rules stay as the `earlier` variant.</li><li>**Arilou, pass 1:** the laser launches keeping a teleport's 3 battery, the teleport defends (0.1 s), and the Chmmr beam rule is a special. Two screens of ordinary bands (280–400, 200–380, 150–300, 300–500 at approach 15 or 60; then full speed, running 110° or 180°, orbiting) were all clearly behind the earlier rules (best 2.094 harm dealt per fight against 2.473). Bands that never run did nearly as well as the best, so the earlier rule's back-off was what paid: inside 280 it steered 110° off the bearing whether or not the opponent closed, which `keep_standoff`, running only from a predicted approach, can't express. Hence `Band.back_off` (`BaselineContext._back_off_band`): beyond the far edge close in at an angle, inside the near edge steer `run_offset` off the bearing, no prediction or release. A 280–400 back-off band at 110° flew the earlier rules fight for fight. Third screen of back-off bands: 200–400 at 110° led (2.511 over 40 seeds) and 280–400 at 135° was level; 150–300 and 200–300 at 90° were clearly behind. Confirmed on seeds 100–119 (2,000 paired fights): equal, harm dealt −0.005 [−0.041, +0.032], λ score −0.007. Entry: back-off band 200–400 at full speed, 110°; λ 0.5 (it escapes only by teleporting from a hit about to land). The earlier rules stay as the `earlier` variant.</li><li>**Druuge, pass 1:** the recoil escape and crew for battery are specials (against the Chenjesu also shooting the nearest Dogi), and the cannon is aimed; the earlier rules had no movement rule. With no band the entry flew every screening fight exactly as the earlier rules (merged, 1,000 fights). Every band was clearly behind: 600–1,500, 1,000–2,000 and 1,500–2,160 at approach 15 or 60 dealt 0.13–0.55 harm per fight against 1.41 and took 2.7–3.3 against 0.48. Entry: no band, λ 1.0 (slow, it can't outrun anything). Tested fight for fight against the earlier rules (`MovedShipTests`).</li><li>**Chmmr, pass 1:** the tractor is a special and the laser is aimed; the earlier rules flew straight at the opponent throughout. First screen (ordinary bands with near edge 0 and far 0, 300, 450 or 600, facing or free): 600 facing led (2.052 harm dealt per fight over 40 seeds against 1.933) but on seeds 100–119 was only +0.080 [+0.005, +0.154], under 5%, and behind on the λ score, so not accepted. A predictive band runs from an opponent closing fast enough even with a near edge of 0, turning the Chmmr from rammers. Second screen with back-off bands and a new `chase` posture (`BAND_INSIDE_CHASE`: fly straight at the opponent), which with no edges flew the earlier rules fight for fight: closing in at full speed to 600 and facing inside it led (2.457 against 1.875; chasing with a back-off inside 200 or 300 was clearly behind, as was chasing from 450). Confirmed on seeds 100–119 (2,000 paired fights): harm dealt +0.498 per fight [+0.423, +0.574], λ score +0.286 [+0.209, +0.364]; harm taken 0.90 → 1.29. Clearly behind the earlier rules against the Androsynth, Ilwrath, Melnorme, Orz, Pkunk, Shofixti, Spathi, Thraddash and Zoq-Fot-Pik (pass 2). Entry: back-off band 0–600 at full speed, facing inside; λ 1.0 (the costliest ship, it never runs). The earlier rules stay as the `earlier` variant.</li><li>**Earthling, pass 1:** point defence, the nuke (from a standstill turn, clear of its minimum distance) and the planet whip away from the opponent are specials; coasting above top speed moves from the strategy to the table. The earlier band was 1,200–2,000. Screen (bands 1,200–2,000 facing or free, 800–1,600, 1,500–2,160 and 1,000–2,160 facing, all veering 135°): 1,000–2,160 facing led (1.342 harm dealt per fight over 40 seeds) and the earlier rules fell out in round 1. Confirmed on seeds 100–119 (2,000 paired fights): harm dealt +0.084 per fight [+0.049, +0.120], λ score +0.047 [+0.005, +0.090]; harm taken 1.26 → 1.35. Against the Chmmr it still circles across the bearing (an orbit band with no edges) with point defence and nukes. λ 0.5 (slow, it keeps its distance only by turning early). The earlier rules stay as the `earlier` variant.</li><li>**Ilwrath, pass 1:** flame and cloak, waiting for a faster opponent that isn't coming, and following one pulling out of reach are specials. Screen: ordinary bands closing to 200, 250 or 300 (at approach 60 or full speed) were all clearly behind (0.48 harm dealt per fight against 0.85), as was dropping the hang-back; back-off bands to the flame's 300 reach (no near edge, as `approach_range` with no inner edge; and 210–300, which flies the same) were level with the earlier rules. Confirmed on seeds 100–119 (2,000 paired fights): equal, harm dealt +0.001 [−0.011, +0.013]. Entry: back-off band 0–300 at full speed; λ 0.5. The earlier rules stay as the `earlier` variant.</li><li>**Kohr-Ah, pass 1:** gas and the held saw are specials. As for the Chenjesu, the saw aims every decision, so it is compared both as an early special with no band and after far bands (2,400, 1,500, 1,000 and 600 off, 1,000–2,000, 600–1,200) in place of avoid_opponent. Aiming first led (3.281 harm dealt per fight against 1.931 for the earlier rules); every band was clearly behind (0.92–2.07). Confirmed on seeds 100–119 (2,000 paired fights): harm dealt +1.310 per fight [+1.248, +1.372], λ score +1.504; harm taken 0.95 → 0.54; clearly behind against no opponent. Entry: specials only, no band, no shoot-downs, λ 1.0 (costly and slow). The earlier rules stay as the `earlier` variant.</li><li>**Kzer-Za, pass 1:** fighters (kept at 8 out, except against their counters and a Chmmr with satellites) are a special and the gun is aimed; the earlier rules flew straight at the opponent. Screen: ordinary bands (0, 0–600 and 0–1,000 facing, 500–1,200 and 0–1,000 free) and back-off bands (chasing from 0 or 300, 0–600 and 0–1,000 facing). A back-off chase with no edges flew the earlier rules fight for fight (merged); closing in at full speed to 1,000 and facing inside it led (2.078 harm dealt per fight over 40 seeds against 1.950). Confirmed on seeds 100–119 (2,000 paired fights): equal, harm dealt +0.064 [+0.009, +0.119] (3%, under 5%), λ score +0.026 [−0.075, +0.128]; harm taken 2.48 → 2.42. Lower against the Ilwrath (2.68 → 1.62 per fight), Slylandro, Umgah, Thraddash, Earthling, Chmmr and Supox (pass 2). Entry: back-off band 0–1,000 at full speed, facing inside; λ 1.0 (the slowest ship, with the most crew, it can't stall). The earlier rules stay as the `earlier` variant.</li><li>**Thraddash, Umgah and Vux, pass 1:** moved to the table unchanged, each running its whole earlier rules as one special (`thraddash_rules`, `umgah_rules`, `vux_rules`); every opponent on seed 0 in all four cells (300 paired fights) flew fight for fight as the `earlier` variant. Their movement is split into the schema in pass 2. λ 0.25 for the Thraddash (its afterburner outruns most ships), 0.5 for the Umgah (it escapes only by hopping backwards), 1.0 for the Vux (slow, it can't run).</li><li>**Melnorme, pass 1:** the charged shot is a special (keep charging, release on target in range) aimed late, and confusion fires when aimed without turning the nose. Screen: ordinary bands 1,134–1,620, 800–1,400, 1,300–1,800 and 600–1,134 at approach 15 or 60 were all clearly behind the earlier rules (0.48–1.29 harm dealt per fight against 1.61), the nearer the better; a back-off band at the earlier 1,134–1,620 (`approach_range` to the gun's reach) flew the earlier rules fight for fight (merged), and one at 800–1,400 led (2.137). Confirmed on seeds 100–119 (2,000 paired fights): harm dealt +0.542 per fight [+0.489, +0.595], λ score +0.602 [+0.541, +0.663]; harm taken 1.84 → 1.64; lower only against the Vux (−0.04). Entry: back-off band 800–1,400 at full speed, 110°; λ 0.5 (it keeps its range but can't outrun the fast ships). The earlier rules stay as the `earlier` variant.</li><li>**Mycon, pass 1:** regeneration is a special, and the planet whip when slow and the plasmoid run late. Far bands (2,400, 1,500, 1,000 and 600 off; 1,000–2,000, 600–1,200) were all clearly behind the earlier rules (1.76–1.93 harm dealt per fight against 1.96): a band runs from anything inside its near edge, while the earlier rules fled only a closing opponent within 2,000 (`avoid_opponent`). Second screen: fleeing that way flew the earlier rules fight for fight (merged); fleeing within `avoid_opponent`'s own 2,400, or with a 1,000–2,000 band besides, did no better; no movement rule at all (keep moving does the rest) led (1.994 over 20 seeds against 1.977). Confirmed on seeds 100–119 (2,000 paired fights): tied on harm dealt, +0.019 [−0.028, +0.066], ahead on the λ score, +0.150 [+0.087, +0.213]; harm taken 2.38 → 2.07. Lower against the Pkunk (3.20 → 2.54 per fight) and Utwig (pass 2). Entry: specials only, no band, no shoot-downs, λ 1.0 (slow, it can't run). The earlier rules stay as the `earlier` variant.</li><li>**Mmrnmrhm, pass 1:** the forms are the mode switch (the X-form within its laser's reach, the Y-wing beyond), the X-form lasers flying at the opponent and the Y-wing missiles are a special, and it faces the opponent late. Far bands (2,400, 1,500, 1,000 and 600 off; 1,000–2,000, 600–1,200) were all clearly behind the earlier rules (0.41–0.67 harm dealt per fight against 0.93). Second screen: fleeing a closing opponent as the earlier Y-wing did (`avoid_opponent`), alone or with a band at least 600 off, was behind too; no movement rule at all led (1.639 against 0.926). Confirmed on seeds 100–119 (2,000 paired fights): harm dealt +0.728 per fight [+0.687, +0.769], λ score +0.223 [+0.165, +0.281] (at λ 0.5); harm taken 0.49 → 1.51, as the earlier Y-wing kept away and dealt little. Ahead against every opponent. Entry: no band, λ 0.25 (the Y-wing outruns most ships, so running away mustn't pay). The earlier rules stay as the `earlier` variant.</li><li>**Orz, pass 1:** marines are a special and the turret runs late; kiting opponents are held at 700–1,100 and a Chmmr at 1,550–1,800, as per-opponent bands. Screen: ordinary bands 907–1,296, 700–1,100 and 500–900, each with or without the kiting overrides, were all clearly behind the earlier rules (1.89–2.15 harm dealt per fight against 2.45); a back-off band at 907–1,296 (`approach_range` to the gun's reach) was level (2.442 against 2.442 over 20 seeds). Confirmed on seeds 100–119 (2,000 paired fights): equal, harm dealt −0.000 [−0.002, +0.001], the same fight for fight except against the Chmmr, whose band now holds with or without satellites (0.317 → 0.309). Entry: back-off band 907–1,296 at full speed, 110°; λ 0.5 (it can't outrun the fast ships). The earlier rules stay as the `earlier` variant.</li><li>**Pkunk, pass 1:** insults (its only battery source) and passing beside the opponent while charged are specials, and the nearest gun is aimed late; below the battery threshold it recharges outside the opponent's reach (`recharge_retreat`) in place of the earlier station-keeping. Screen: recharging below 4 or 6 of 12, resuming at 9, or running 135° off the bearing were all level with the earlier rules (0.540–0.548 harm dealt per fight over up to 40 seeds); bands at 302–432 and 200–350 closing at full speed were clearly behind (0.13 and 0.23). Confirmed recharging below 4 on seeds 100–119 (2,000 paired fights): equal, harm dealt +0.003 [−0.002, +0.008], λ score +0.001; no opponent moved by more than 0.04 per fight. It deals little (0.57 per fight) and takes much (2.88): left for pass 2. Entry: recharge below 4, no band; λ 0.25 (among the fastest ships, it can always run). The earlier rules stay as the `earlier` variant.</li><li>**Shofixti, pass 1:** the glory device is a special and the gun is aimed; from an empty battery it recharges outside the opponent's reach until full (4), where the earlier rules fled a closing opponent (`avoid_opponent`). Screen: ordinary bands 605–864, 400–700 and 700–1,000 at approach 15 or 60, and 605–864 recharging only to 1, were all clearly behind the earlier rules (1.59–3.34 harm dealt per fight against 7.22); a back-off band at 605–864 (`approach_range` to the gun's reach) was level, and one at 400–700 led (7.486). Confirmed on seeds 100–119 (2,000 paired fights): equal, harm dealt +0.257 [+0.116, +0.398] (4%, under 5%), λ score +0.164 [+0.037, +0.291]; harm taken 6.27 → 6.44. Higher against the Umgah (+2.33 per fight) and Vux, slightly lower against the Thraddash, Kzer-Za, Utwig and Yehat. Entry: back-off band 400–700 at full speed, 110°; λ 0.25 (fast, and its glory blast spends its own crew on purpose). The earlier rules stay as the `earlier` variant.</li><li>**Supox, pass 1:** the slides are its movement and fire, as one special. Screen of table bands, which `supox` reads in place of its own: 600–900 with no reach margin was level with the earlier rules (3.116 harm dealt per fight over 20 seeds against 3.102, slightly behind on the λ score); 600–900, 500–800 and 700–1,000 with a margin of 150, and 600–900 with 50, were all clearly behind (1.53–1.73). The earlier band moves outside a shorter-reaching opponent only while its own gun still reaches (0.95 of it, less the band's width), and otherwise holds 600–900; `band_limits` has no such cap, so against long reaches it held the Supox out of range. Entry: no table band, so the special holds the earlier band; every opponent on seed 0 in all four cells (100 paired fights) flew fight for fight as the `earlier` variant. λ 0.5 (it slides out of the way but can't outrun the fast ships). Capping `band_limits` at a ship's own reach is for pass 2.</li><li>**Slylandro, pass 1:** harvesting asteroids is a special (flying at the nearest below 10% battery), and the lightning, which seeks its target, launches within its 1,152 reach; against the Chmmr the earlier rule is an override from the schema (the low-battery harvest, a 700–850 band orbiting inside, lightning within 1,000). A back-off band at 806–1,152 (`approach_range` to the lightning's reach) flew the earlier rules fight for fight (seed 0 against every opponent, 100 paired fights). Screens on Windows (`%USERPROFILE%\StarAI-measurements\phase6\`, outside the repository; the Linux scripts are gone). First: back-off bands 600–1,000 and 400–800, the earlier band orbiting inside, and predictive bands at 806–1,152 and 600–1,000 orbiting; the nearer the better (400–800: 2.369 harm dealt per fight against 1.208), the predictive bands, which run from any closer, worst (0.13–0.14). Second, nearer back-off bands: 150–300 led (3.394 over 40 seeds), tied with 0–300 (3.357); 200–400 was level at 20 seeds; 200–500, 300–600 and 400–800 were clearly behind. Third, on 150–300: firing only within 0.6 or 0.35 of the reach was clearly behind (3.18, 2.69), and running 90° or 135° off the bearing was tied with 110°. With the lightning off, harm dealt fell to 0.003 per fight: the harvest deals none, and the band pays because the lightning lands far more often from close in. Confirmed on seeds 100–119 (2,000 paired fights): harm dealt +2.183 per fight [+2.119, +2.247], λ score +1.914 [+1.833, +1.996]; harm taken 3.12 → 4.16. Ahead against every opponent but the Chmmr, where it flies as before. Entry: back-off band 150–300 at full speed, 110°; λ 0.25 (among the fastest, and the tractor can't hold it). The earlier rules stay as the `earlier` variant.</li><li>**Syreen, pass 1:** singing within 300 when the song can take crew, collecting crew pods, and closing in to sing with the song ready unless that is too risky while the crew is small are specials, and the gun is aimed late, keeping the song's 5 battery spare; rule 1's gate (`_syreen_may_evade`) is kept. Below the song's cost it keeps out of reach (`recharge_retreat`), where the earlier rules fled a closing opponent (`avoid_opponent`). Against the Chmmr the earlier rule is a whole-rule special (`syreen_chmmr`), fight for fight as before. First screen: the planned entry (back-off band 756–1,080, `approach_range` to the gun's reach), with no recharging, recharging to 10, or running 135° off while recharging, were all clearly behind the earlier rules (0.970–0.975 harm dealt per fight against 1.011); a back-off band at 500–900 led (1.080) and 300–600 was behind. Second screen: 500–900 led over 40 seeds (1.075), tied with the same without recharging (1.073); 600–1,000 was level at 20 seeds, and 400–750, facing inside 500–900, and a predictive 500–900 band were clearly behind. The band steers only while the song isn't the goal (not viable, or too risky), so the settings differ little. Confirmed on seeds 100–119 (2,000 paired fights): harm dealt +0.117 per fight [+0.082, +0.152], λ score +0.155 [+0.102, +0.208]; harm taken 4.21 → 4.14. Ahead against the Arilou, Slylandro, Mycon, Umgah, Zoq-Fot-Pik and Spathi and level against the rest. It still deals little (1.10 per fight) and takes much: the song and close-in rules are for pass 2. Entry: recharge below 5, back-off band 500–900 at full speed, 110°; λ 0.5 (it can't outrun the fast ships). The earlier rules stay as the `earlier` variant.</li><li>**Utwig, pass 1:** the shield defends (a hit due within 0.2 s, or crew lost), against the Shofixti also wherever its blast would kill (`shield_glory`, an override), and the bolts are aimed; rule 1's gate (`_utwig_may_evade`) is kept, and the Chmmr rule is a whole-rule special. The earlier rules pressed the shield even without the battery for it, which the defend rule checks; that alone made 6 of 100 fights on seed 0 differ (with the check off, none), and the two screened level (merged). First screen: back-off bands at 500–900 led (2.171 harm dealt per fight against 1.603 for the earlier rules); 300–700, the entry at 756–1,080 (`approach_range` to the bolts' reach), the shield at 0.1 or 0.4 s, a predictive 756–1,080 band and 900–1,080 were all clearly behind. Second screen: 500–900 led over 40 seeds (2.185), tied with 600–1,000 (2.127); running 135° off was level at 20 seeds; 400–800 and 550–800 were behind, and facing (1.29) or orbiting (0.15) inside the band, which leaves the forward bolts off the opponent, far behind. Confirmed on seeds 100–119 (2,000 paired fights): harm dealt +0.539 per fight [+0.488, +0.589]; harm taken 2.01 → 2.03. Ahead against 20 opponents, level against the Androsynth, Chmmr, Druuge and Shofixti, but behind against the Spathi (1.04 against 2.45 harm dealt, taking 0.93 against 0.26), which holds 756–1,080 itself: against it the earlier band is kept as an override, fight for fight as before (80 of 80 on the confirmation seeds, on which the override was chosen). Entry: back-off band 500–900 at full speed, 110°; λ 0.5 (heavy and no faster than most, it can't run). The earlier rules stay as the `earlier` variant.</li><li>**Yehat, pass 1:** as the Utwig, the shield defends (a hit due within 0.2 s, or crew lost) and the guns are aimed, with `shield_glory` against the Shofixti and the Chmmr rule a whole-rule special. Checking the battery before shielding changed 14 of 100 fights on seed 0 and screened level with the earlier rules. Four band screens, each clearly ahead of the last: 350–650 (1.167 harm dealt per fight against 0.706 for the earlier rules; nearer 250–500 and 150–400, the entry at 504–720, `approach_range` to the guns' reach, the shield at 0.1 or 0.4 s and a predictive band were all behind), then 300–650 (1.297), 250–700 (1.500 over 20 seeds), and 100–700 (1.632 over 40 seeds), tied with 200–720 (1.595); wider bands reaching to the guns did better each time. Confirmed on seeds 100–119 (2,000 paired fights): harm dealt +0.902 per fight [+0.853, +0.951], 1.66 against 0.76; harm taken 1.69 → 1.52. Ahead against 20 opponents and level against the Chmmr, Kzer-Za, Orz and Spathi, but behind against the Shofixti (0.49 harm dealt against 0.65, taking 2.98 against 2.51): closer in, its glory blast costs more. Against it the earlier band is kept as an override, chosen on the confirmation seeds (78 of 80 confirmation fights identical, the rest level). Entry: back-off band 100–700 at full speed, 110°; λ 0.5 (slow, it can't run). The earlier rules stay as the `earlier` variant.</li><li>**Zoq-Fot-Pik, pass 1:** the tongue (lick when aimed within its reach) and the lunge at an opponent it can reach within a second are specials, and the gun is aimed late, keeping the tongue's 7 battery spare while the tongue can land. Below 30% battery it keeps out of reach (`recharge_retreat`), where the earlier rules fled a closing opponent (`avoid_opponent`). No Chmmr rule (row 2.13). First screen: the planned entry (back-off band 605–864, `approach_range` to the gun's reach) was level with the earlier rules (1.936 harm dealt per fight against 1.921), as were recharging to 7 and not recharging; a back-off band at 100–864 led (2.328), with 300–864, 200–500 and 400–700 clearly behind. Second screen: every variant was within 3% (100–800, 100–700, 0–864, 200–800, no recharging, 135°); 100–864 led over 40 seeds (2.333), tied with 100–800 (2.344). Confirmed on seeds 100–119 (2,000 paired fights): harm dealt +0.402 per fight [+0.338, +0.467], 2.33 against 1.93; harm taken 3.76 → 3.70. Ahead against 9 opponents and level against 14, but behind against the Orz (1.04 harm dealt against 1.33, both taking far more than they deal) and on the λ score against the Spathi, against which both deal next to nothing (0.06–0.08 per fight; pass 2). Against the Orz the earlier band is kept as an override, chosen on the confirmation seeds: 1.323 against 1.326 for the earlier rules, tied. Entry: recharge below 3, back-off band 100–864 at full speed, 110°; λ 0.5 (it outruns only the slow ships). The earlier rules stay as the `earlier` variant.</li><li>**Settling after pass 1** (every ship against every opponent, seeds 0–4 × 4 cells, 12,500 fights; 34 minutes on Windows; results in `settle_pass1.json` beside the screens). Harm dealt / taken per fight, each ship's own fights: Androsynth 1.26 / 1.09, Arilou 2.47 / 3.10, Chenjesu 2.54 / 1.21, Chmmr 2.48 / 1.35, Druuge 1.47 / 0.60, Earthling 1.44 / 1.81, Ilwrath 0.98 / 2.20, Kohr-Ah 3.47 / 0.70, Kzer-Za 2.13 / 2.78, Mmrnmrhm 1.69 / 1.75, Melnorme 2.16 / 1.84, Mycon 2.02 / 2.23, Orz 2.48 / 2.23, Pkunk 0.51 / 3.20, Shofixti 7.74 / 6.58, Slylandro 3.44 / 4.66, Spathi 1.34 / 1.16, Supox 3.27 / 3.30, Syreen 1.09 / 4.44, Thraddash 1.01 / 3.77, Umgah 0.54 / 3.41 (still 43% of frames), Utwig 2.22 / 2.13, Vux 1.18 / 1.92, Yehat 1.60 / 1.55, Zoq-Fot-Pik 2.29 / 3.66. Pass 2's settling run is compared with this one.</li></ul>Tests: `SharedBehaviourTests`, `BehaviourSchemaTests` and `SpathiBandTests` in `tests/test_baseline_strategy.py`; the per-minute and still-frame tests in `tests/test_baseline_arena.py`. |
| 6A | Harm evaluation in the arena | Implemented | <ul><li>**Fights:** `src/training/baseline/arena.py` now runs fixed 1,200-frame fights (`DEFAULT_FRAMES`) in the battle's training mode, which respawns a destroyed ship. An `ArenaSimulation` subclass sets each spawn's crew and satellites from its side's condition. The conditions are `STRONG` (normal crew, all satellites) and `WEAK` (a quarter of the crew rounded up, `WEAK_SATELLITES` = 1). They are drawn from a key per match, side and life, not the battle's random stream, so every variant meets the same starts. A respawned ship's controller is reset. Spawned ships get training's starting velocities, so they don't start at rest.</li><li>**Matrix:** `build_matches` flies each seed in the 4 cells (`CELLS`: SS, SW, WS, WW, mine then theirs).</li><li>**Harm:** a `HarmTracker` runs on each frame's ledger events. `ArenaResult` records damage and kill harm dealt and taken, kills, deaths, still frames and the λ in force; `score` is 0 without harm dealt.</li><li>**λ:** `ShipBehaviour.harm_taken_weight`, 0.5 until a ship moves; the Spathi has 0.25.</li><li>**Reports:** `tally` groups by ship and variant, then by all, opponent and cell. The report shows total and per-fight harm dealt, harm taken and λ score, plus kills, deaths, the share of fights with no harm, and still frames.</li><li>**Ranking:** `paired_difference` (95% interval, Student's t), `compare` (harm dealt first, then the λ score) and `rank_order` (lead group by λ score, then by harm dealt). The CLI compares each variant with the default.</li><li>**Screening:** `screen` and `--screen` drop settings as they fall behind, flying only the new seeds each round. After each round they merge settings whose fights all came out identical (`_merge_identical`), drop settings clearly behind the leader, keep at most the better half, and stop once one is left without flying it further. Against the Chenjesu, for example, the leader was 1.076 against the next setting's 0.765 at 20 seeds, yet the old screen flew on to 100 seeds (1,240 fights).</li><li>**CLI:** `--first-seed`, `--frames`, `--cells` and `--by-cell`.</li><li>**Speed:** see "Speed" under Phase 6 for the savings made and the worker measurements.</li></ul>Tests: `tests/test_baseline_arena.py` (conditions and respawns, repeatability, shared observations, scoring, paired intervals, ranking, screening, CLI). |
| 6B | Perception delay for computer control | Implemented and measured; default 5 kept | <ul><li>**Ages:** `World` keeps `frame_id`, which `BattleSimulation` sets each step, and stamps `world_entry_frame` on every object `add` or `add_all` takes in. Objects with no stamp are never held back.</li><li>**Encoder:** `encode_observation` filters the shared `ObservationFrameContext` per viewer (`_context_for_viewer` in `observation.py`). An object on the enemy side (`_enemy_side`: owned by the enemy ship, an earlier life of it, or an unowned ability of the other player) younger than `const.PERCEPTION_DELAY_FRAMES` is left out of the slots and of the enemy's satellite, fighter, Dogi, saw and floating-marine counts. Ships, own objects, the planet, asteroids, Syreen crew pods, marines aboard the viewer and shields (ship state, which the ship block's shield flag already shows) are never held back. At 0 the shared context is used unchanged.</li><li>**Game rules:** satellites, point defence and homing read the ships' tracking lists, which the delay doesn't touch. Baseline strategies decide from the observation alone; `guard_computer_controls` reads only the two ships.</li><li>**Setting:** `perception_delay_frames` in `DEFAULT_GAMEPLAY` and `GameSettings` (0–24, default 5; out-of-range stored values fall back to 5), applied by `apply_game_settings`, and the Game Play slider "Perception delay (frames)".</li><li>**Processes:** spawned processes import `const` afresh and so started from the default game-play settings: coordinated workers flew 16 directions whatever the setting while the model metadata recorded the setting. `const.runtime_game_settings()` (asteroids, directions, repeat-key delay, perception delay) now goes to coordinated workers in `StartRunCommand.game_settings` and to `independent_training_process_main` as `game_settings`; each applies it with `const.apply_runtime_game_settings` before building a battle. Checked with a real spawned worker: a parent at 32 directions gave 22.5° heading steps before, 11.25° after, with display frames drawn.</li><li>**Models:** `current_game_settings_metadata` records `perception_delay_frames`; `validate_model_metadata` reads a model without it as 0 and warns on a difference, as for ship directions.</li><li>**Arena:** `--perception-delay` (default 5, `DEFAULT_PERCEPTION_DELAY`), stated at the start and end of the report.</li><li>**Shot outcomes** (`src/training/baseline/shot_outcomes.py`: `ShotTracker` follows each damaging projectile or laser to hit, absorbed, teleported, shot down or missed; `arena.run_match` takes `on_step`). Spathi, Arilou, Slylandro and Vux against the Arilou, Yehat and Utwig, 5 seeds × 4 cells, delays 0 / 3 / 5 / 8. Spathi A1 at the Arilou: hit 0.0 / 0.3 / 18.2 / 60.6%, teleported 59.7 / 45.5 / 33.1 / 0.2%. At the Yehat: hit 1.4 / 0.2 / 3.1 / 28.2%, absorbed 61.6 / 58.1 / 54.7 / 30.6%. At the Utwig: hit 1.7 / 1.0 / 8.7 / 57.5%, absorbed 73.3 / 77.3 / 63.0 / 6.9%, shot down 18.3 / 16.5 / 26.1 / 28.7%. Lasers arrive the frame they are fired, so the delay leaves them unchanged: Arilou A1 hits the Utwig 97–100% and the Yehat 49–53% (44–48% absorbed) at every delay.</li><li>**Every ship** against every opponent, 5 seeds × 4 cells (12,500 fights per delay; results and `analyse.py` kept outside the repository). Harm taken per fight, the ship's fights as evaluated ship and as opponent pooled, delay 0 → 5 (3, 8): Supox 1.63 → 2.57 (2.28, 2.72), Arilou 1.29 → 2.21 (1.83, 2.57), Thraddash 2.04 → 2.78, Utwig 1.02 → 1.55 (1.27, 1.81), Slylandro 2.10 → 2.62, Zoq-Fot-Pik 2.65 → 3.07, Yehat 0.85 → 1.21 (1.04, 1.61). Every other ship moved by 0.15 or less; harm taken rose for 23 of 25 ships (Druuge flat, Syreen 3.21 → 3.15). As expected, the Arilou, Utwig and Yehat lost much of their defence, and so did the ships whose rule 1 dodges close shots (Supox, Thraddash, Slylandro, Zoq-Fot-Pik). The Chmmr, Kohr-Ah, Spathi and Mmrnmrhm barely changed.</li><li>**Tuned rules**, harm dealt / taken per fight, delay 0 → 5 (8): Utwig vs Chmmr 1.62/0.91 → 1.90/1.07 (2.01/1.09) and Yehat vs Chmmr 0.53/2.69 → 0.63/2.66 (0.70/2.74), so both Chmmr rules still hold. Arilou vs Spathi 0.02/0.00 → 0.06/1.10 (0.06/4.75; 0 → 22 → 101 deaths), Yehat vs Spathi 0.00/0.01 → 0.00/0.07 (0.00/0.79), Utwig vs Spathi 0.11/0.01 → 0.07/0.02 (0.12/0.19). The Spathi's settings were chosen at delay 0 and are retuned at 5 next. The Phase 3A shoot-downs were checked only through the Utwig's share of Spathi shots, which rose.</li><li>**Sensitivity:** at 3 the Spathi still can't hit the Arilou (0.3%) and nothing changes much. At 8 the shields and teleport fail against slow shots (the Utwig absorbs 6.9% of Spathi A1 bolts). Five frames is the point where escapes and shields start to be beatable without being broken, so the default stays 5.</li><li>**Memory:** each arena worker cached the assets of every ship type it met, up to about 3 GB at 64 directions, and an all-pairs sweep on 24 workers exhausted 61 GB. `ARENA_CACHED_SHIP_TYPES` (6) now bounds the cache: past it the worker drops its asset manager and, on Linux, returns the freed memory with `malloc_trim`. Fight results are unchanged. The four sweeps peaked at 10 GB in total, about 25 minutes each.</li></li></ul> Spathi screening at delay 0 shortlists settings; the shortlist is rescreened at the default delay before the Spathi is confirmed. Tests: `tests/test_perception_delay.py`. |
