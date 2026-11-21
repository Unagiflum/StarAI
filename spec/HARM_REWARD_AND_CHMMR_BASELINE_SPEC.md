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
- Training statistics show the harm totals per fight: damage, kill and the two
  penalties.

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

### Process

1. Extract shared behaviours as engine helpers.
2. Add a per-ship parameter table: distance band, recharge threshold, gates and
   planet horizon.
3. Move one ship at a time. Compare crew lost and dealt per minute, still
   frames, and W-L before and after.
4. Commit each ship separately. Revert any ship that regresses.

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
| 4 | Discount half-life setting | Not started | |
| 4A | Reward magnitude steps ±0.001–1.024 | Not started | |
| 4B | Decay half-lives in batches | Not started | |
| 5 | Harm reward scheme | Not started | |
| 6 | Shared behaviours, per ship | Not started | |
